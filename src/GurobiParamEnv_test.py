"""
Module defining the InexactGBDEnv for a Gymnasium environment that uses reinforcement
learning to guide Generalized Benders Decomposition by controlling the master problem
gap tolerance.
"""
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
    A Gymnasium environment where the agent selects the gap tolerance factor for the
    master problem in each iteration of a Benders decomposition loop.

    Attributes:
        mp_solver: Gurobi solver for the master problem.
        sp_solver: Ipopt solver for the subproblems.
        max_iter: Maximum number of Benders iterations before termination.
        t_ref: Reference CPU time for reward normalization.
        action_space: Continuous Box space for agent actions (gap factor).
        observation_space: Continuous Box space for observations.
    """
    def __init__(self, max_iter=50, t_ref=0.2):
        """
        Initialize the InexactGBDEnv environment.

        Parameters:
            max_iter (int): Maximum number of Benders iterations allowed.
            initial_gap: Placeholder for an initial gap tolerance (not currently used).
            t_ref (float): Reference time for normalizing the runtime penalty in rewards.
        """
        super().__init__()
        # Initialize solvers
        self.sp_solver = SolverFactory('ipopt')

        self.max_iter = max_iter        
        self.t_ref = t_ref

        self.K = 11

        # Define agent action space: raw output in [-1,1], to be rescaled to a gap tolerance factor
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define environment observation space: [cinit] + demands + solve-info features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1 + len(products) + 6,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment state, configure the optimization models, and
        return the initial observation and info dictionary.
        """

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.convergence = False
        self.itr = 0

        # initialize with finite values to avoid inf in observations
        self.IUB = np.inf # inexact upper bound
        self.TUB = np.inf # true upper bound
        self.LB = - np.inf
        self.IUB_list = [0.0]
        self.TUB_list = [0.0]
        self.LB_list = [0.0]
        
        self.mp_time_list = []
        self.sp_time_list = []
        self.max_sp_time_list = []

        self.true_gap = 1.0
        self.inexact_gap = 1.0
        self.true_gap_prev = 1.0
        self.inexact_gap_prev = 1.0

        self.true_gap_list = [self.true_gap]
        self.inexact_gap_list = [self.inexact_gap]

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

        # 2. Retrieve the demand data
        self.demand = {i: self.scheduling.d[i].value for i in products}

        # Build initial state for Gym reset
        initial_context = [float(self.cinit)] + [float(self.demand[p]) for p in products]
        initial_solve_info = [0, 0.3, 0.3, self.true_gap, 0.0, 0.0] # [Itr, gap_mp, gap_mp_true, gap, log_gap, d_gap]
        state = np.array(initial_context + initial_solve_info, dtype=np.float32)
        info = {'IUB': self.IUB, 'TUB': self.TUB, 'LB': self.LB, 
                'true gap': self.true_gap, 'inexact gap': self.inexact_gap, 
                'mp_time': 0.0, 'sp_time': 0.0, 'iteration': 0}

        return state, info

    def step(self, action):
        """
        Environment for testing the policies. The input is the optimality tolerance direcely.
        """
        # Increment the iteration counter
        self.itr += 1
        print('Iteration:', self.itr)
        
        # Clip to original bounds
        self.gap_mp_set = float(np.clip(action, 1e-3, 0.3))
        
        best_assign, self.gap_mp, mp_time, mp_result = solve_mp_milp(self.scheduling, MIPFocus=1, MIPGap=self.gap_mp_set)
        # If the master problem is infeasible or unbounded, skip this episode
        if mp_result['Solver'][0]['Termination condition'] == TerminationCondition.infeasibleOrUnbounded:
            obs, info = self.reset()
            return obs, -1.0, True, False, info

        assignment = best_assign
        
        self.IUB = value(self.scheduling.obj)
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
                sp_time.append(result_sp_init.solver.time)
                # 3. extract the dual value
                dual_val_init = - self.sp_model.transitions_init[ip].dual[self.sp_model.transitions_init[ip].trans_time_fix]
                # 4. add this benders cut to all slots
                phi_init = value(self.sp_model.transitions_init[ip].obj)
                cut_init_expr = self.scheduling.eta_init[ip] >= (phi_init) - dual_val_init * (self.scheduling.tt_init[ip] - trans_time_ijk)
                self.scheduling.benders_cuts.add(cut_init_expr)
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
                sp_time.append(result_sp.solver.time)
                # 3. extract the dual value
                dual_val = - self.sp_model.transitions[i, ip].dual[self.sp_model.transitions[i, ip].trans_time_fix]
                # 4. add this benders cut to all slots
                phi_ij = value(self.sp_model.transitions[i, ip].obj)
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
        self.inexact_gap_prev = self.inexact_gap
        self.true_gap = (self.TUB - self.LB) / abs(self.LB)
        self.inexact_gap = (self.IUB - self.LB) / abs(self.LB)

        delta_gap = math.log(self.true_gap_prev / max([1e-3, self.true_gap]))

        # Compute reward terms: runtime penalty, gap improvement incentive, iteration penalty
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
            float(np.log(self.true_gap+1e-3)), 
            float(delta_gap) # gap improvement
        ]
        state = np.array(context + solve_info, dtype=np.float32)

        done = self.convergence

        self.mp_time_list.append(mp_time)
        self.sp_time_list.append(sum(sp_time))
        self.max_sp_time_list.append(max(sp_time))
        self.LB_list.append(self.LB)
        self.IUB_list.append(self.IUB)
        self.TUB_list.append(self.TUB)
        self.true_gap_list.append(self.true_gap)
        self.inexact_gap_list.append(self.inexact_gap)

        # Rebuild info dict to include mp_time and iteration
        info = {
            'IUB': self.IUB,
            'TUB': self.TUB,
            'LB': self.LB,
            'true gap': self.true_gap,
            'inexact gap': self.inexact_gap,
            'master_gap_tol': self.gap_mp_set, 
            'mp_time': mp_time,
            'sp_time': sum(sp_time),
            'max_sp_time': max(sp_time),
            'iteration': self.itr
        }

        return state, reward, done, False, info

    def render(self, mode='human'):
        pass

