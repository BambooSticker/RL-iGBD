import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete
import numpy as np
from pyomo.environ import *
from pyomo.opt import TerminationCondition
from tools import solve_mp_milp
from dynamic_opt import transition_block_rule, transition_init_block_rule
from schedule_opt import scheduling_milp_block_rule
from itertools import permutations
import math

products = ['A', 'B', 'C', 'D', 'E']
assignment_map = {}
for i, assignment in enumerate(permutations(products, len(products))):
    assignment_map[assignment] = i

class InexactGBDEnv(gym.Env):
    """
    A Gym environment where the agent provides only the first iteration warm-start
    for a Benders decomposition.  Mixed-integer projection is enforced via MIQP.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_iter=50, initial_gap=None, t_ref=0.2):
        super().__init__()
        # self.mpc       = mpc_model
        self.mp_solver = SolverFactory('gurobi')
        self.sp_solver = SolverFactory('ipopt')

        self.max_iter = max_iter        
        self.t_ref = t_ref

        self.K = 11

        # Agent outputs in [-1, 1]; we will rescale to [1e-3, 0.3] in step()
        self.action_space = spaces.Discrete(self.K)

        # Observation: [cinit] + demand for each product + [MIPGap, Benders gap, UB-LB]
        # 1 (cinit) + 5 (demand) + 4 (solve-info) = 11
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1 + len(products) + 6,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.convergence = False
        self.itr = 0

        # initialize with finite values to avoid inf in observations
        self.TUB = np.inf
        self.LB = - np.inf
        self.TUB_list = [0.0]
        self.LB_list = [0.0]
        
        self.mp_time_list = []
        self.sp_time_list = []

        self.true_gap = 1.0
        self.true_gap_prev = 1.0

        self.true_gap_list = [self.true_gap]

        self.cinit = np.random.uniform(low=0.8, high=1.2)

        # 1. Configure the model.
        def model_configuration(cinit):
            model = ConcreteModel()

            # Set the products for this transition.
            # Create an empty block.
            model.I = Set(initialize=products)      # Products
            model.K = Set(initialize=[1, 2, 3, 4, 5])                # Slots
            
            print("Configuring transition blocks...")
            def transition_block_indexed_rule(b, sp, ep):
                if sp == ep:
                    return
                else:
                    b.start_product = sp
                    b.end_product   = ep
                    transition_block_rule(b)
            model.transitions = Block(model.I, model.I, rule=transition_block_indexed_rule)
            
            print("Configuring initial transition blocks...")
            def transition_init_block_indexed_rule(b, ep):
                b.end_product = ep
                transition_init_block_rule(b)
                b.cinit.set_value(cinit)
            model.transitions_init = Block(model.I, rule=transition_init_block_indexed_rule)
            
            print("Configuring scheduling block...")
            def scheduling_block_rule(b):
                b.cinit = Param(initialize=cinit, doc='Initial concentration')
                scheduling_milp_block_rule(b)
            
            scheduling = ConcreteModel(rule=scheduling_block_rule)

            return model, scheduling
        
        self.sp_model, self.scheduling = model_configuration(self.cinit)

        # ------------------ Force the lower bound in MP -----------------
        self.scheduling.lower_bound = Param(initialize=0, mutable=True)
        # self.scheduling.obj_lower_bound = Constraint(expr=self.scheduling.obj_expr >= self.scheduling.lower_bound - 1) # Slack for avoiding numerical issue
        # ---------------------------------------------------------

        # 2. Retrieve the demand data
        self.demand = {i: self.scheduling.d[i].value for i in products}

        # Build initial state for Gym reset
        # Context: [cinit] + demand values (all as float)
        # Solve-info: [iteration, gap_mp, gap] at start
        initial_context = [float(self.cinit)] + [float(self.demand[p]) for p in products]
        initial_solve_info = [0, 0.3, 0.3, self.true_gap, 0.0, 0.0] # [Itr, gap_mp, gap_mp_true, gap, log_gap, d_gap]
        state = np.array(initial_context + initial_solve_info, dtype=np.float32)
        info = {'TUB': self.TUB, 'LB': self.LB, 'true gap': self.true_gap, 
                'mp_time': 0.0, 'sp_time': 0.0, 'iteration': 0}

        return state, info

    def step(self, action):
        """
        Decision at each iteration of Benders:
        - The policy outputs the factor
        - Update the MIPGap
        - Solve the master problem and transition times
            - Record the MP runtime (Penalty 1)
        - Solve the subproblems and generate a new cut
        - Add the cut to master 
        - Update the global gap:
            - Record the gap improvement (Reward 2)
        - run full Benders decomposition
        - reward = -iterations
        """
        # Action: factor of gap

        # -------------------------------------------------------------
        self.itr += 1
        print('Iteration:', self.itr)
        
        raw_idx = int(action)
        raw = -1.0 + 2.0 * raw_idx / float(self.K - 1)
        mp_gap_tol_u = 0.3
        mp_gap_tol_u_l = 1e-3

        factor = mp_gap_tol_u_l + (raw + 1.0) * 0.5 * (self.true_gap - mp_gap_tol_u_l)
        # Clip to original bounds
        self.gap_mp_set = float(np.clip(factor, mp_gap_tol_u_l, mp_gap_tol_u))

        best_assign, self.gap_mp, mp_time, mp_result = solve_mp_milp(self.scheduling, MIPFocus=1, MIPGap=self.gap_mp_set)
        # mp_time is captured here
        # If the master problem is infeasible or unbounded, skip this episode
        if mp_result['Solver'][0]['Termination condition'] == TerminationCondition.infeasibleOrUnbounded:
            obs, info = self.reset()
            return obs, -1.0, True, False, info

        assignment = best_assign
        
        self.TUB = value(self.scheduling.obj) * (1 + self.gap_mp)
        if self.itr > 1 and self.TUB > self.TUB_list[-1]:
            self.TUB = self.TUB_list[-1]
        
        # Get transition schedule
        trans_pairs = list(assignment.keys())
        
        self.LB = 0
        sp_time = 0
        for i, ip, k in trans_pairs:
            # 1. fix the transition time
            trans_time_ijk = assignment[i, ip, k]
            if k == 0:
                if hasattr(self.sp_model.transitions_init[ip], 'trans_time_fix'):
                    self.sp_model.transitions_init[ip].del_component(self.sp_model.transitions_init[ip].trans_time_fix)
                self.sp_model.transitions_init[ip].trans_time_fix = Constraint(expr=self.sp_model.transitions_init[ip].trans_time==trans_time_ijk)
                # 2. solve the subproblem
                result_sp_init = self.sp_solver.solve(self.sp_model.transitions_init[ip], tee=False)

                if result_sp_init['Solver'][0]['Termination condition'] != TerminationCondition.optimal:
                    obs, info = self.reset()
                    return obs, -1.0, True, False, info
                
                # record solver-reported subproblem runtime
                sp_time += result_sp_init.solver.time
                # 3. extract the dual value
                dual_val_init = - self.sp_model.transitions_init[ip].dual[self.sp_model.transitions_init[ip].trans_time_fix]
                # print("\tDual value:", dual_val_init)
                # 4. add this benders cut to all slots
                phi_init = value(self.sp_model.transitions_init[ip].obj)
                cut_init_expr = self.scheduling.eta_init[ip] >= (phi_init) - dual_val_init * (self.scheduling.tt_init[ip] - trans_time_ijk)
                self.scheduling.benders_cuts.add(cut_init_expr)
                # print("\tTransition cost:", phi_init)
                # 5. lower bound
                self.LB += - phi_init
            else:
                if hasattr(self.sp_model.transitions[i, ip], 'trans_time_fix'):
                    self.sp_model.transitions[i, ip].del_component(self.sp_model.transitions[i, ip].trans_time_fix)
                self.sp_model.transitions[i, ip].trans_time_fix = Constraint(expr=self.sp_model.transitions[i, ip].trans_time==trans_time_ijk)
                # 2. solve the subproblem
                result_sp = self.sp_solver.solve(self.sp_model.transitions[i, ip], tee=False)

                if result_sp['Solver'][0]['Termination condition'] != TerminationCondition.optimal:
                    obs, info = self.reset()
                    return obs, -1.0, True, False, info
                
                # record solver-reported subproblem runtime
                sp_time += result_sp.solver.time
                # 3. extract the dual value
                dual_val = - self.sp_model.transitions[i, ip].dual[self.sp_model.transitions[i, ip].trans_time_fix]
                # 4. add this benders cut to all slots
                phi_ij = value(self.sp_model.transitions[i, ip].obj)
                # print("\tDual value:", dual_val)
                for kk in self.scheduling.Kt:
                    cut_expr = self.scheduling.eta[i, ip, kk] >= (phi_ij) - dual_val * (self.scheduling.tt[i, ip, kk] - trans_time_ijk)
                    
                    self.scheduling.benders_cuts.add(cut_expr)
                # 5. lower bound
                self.LB += - phi_ij

        self.LB += value(self.scheduling.obj) + value(self.scheduling.expr_var_transition_cost) + value(self.scheduling.expr_var_transition_init_cost)
        if self.itr > 1 and self.LB < self.LB_list[-1]:
            self.LB = self.LB_list[-1]
        self.scheduling.lower_bound = self.LB
 
        self.true_gap_prev = self.true_gap
        self.true_gap = (self.TUB - self.LB) / abs(self.LB)

        delta_gap = math.log(self.true_gap_prev / max([1e-3, self.true_gap]))

        # penalize iteration count, master problem solve time, and gap‐reduction
        reward_t = - mp_time/self.t_ref
        reward_g = 2 * delta_gap
        reward_e = -1
        
        reward = reward_t + reward_g + reward_e

        if self.true_gap <= 1e-3 or self.itr >= self.max_iter:
            self.convergence = True

        # Build state with explicit floats to avoid sequences
        context = [float(self.cinit)] + [float(self.demand[p]) for p in products]
        solve_info = [
            float(self.itr),
            float(self.gap_mp_set), # mp gap tol
            float(self.gap_mp), # real mp gap
            float(self.true_gap), 
            float(np.log(self.true_gap+1e-3)), # +1e-3 to avoid numerical issues
            float(delta_gap) # gap improvement
        ]
        state = np.array(context + solve_info, dtype=np.float32)

        done = self.convergence

        self.mp_time_list.append(mp_time)
        self.sp_time_list.append(sp_time)
        self.LB_list.append(self.LB)
        self.TUB_list.append(self.TUB)
        self.true_gap_list.append(self.true_gap)

        # Rebuild info dict to include mp_time and iteration
        info = {
            'TUB': self.TUB,
            'LB': self.LB,
            'true gap': self.true_gap,
            'master_gap_tol': self.gap_mp_set, 
            'mp_time': mp_time,
            'sp_time': sp_time,
            'iteration': self.itr
        }

        return state, reward, done, False, info

    def render(self, mode='human'):
        pass
