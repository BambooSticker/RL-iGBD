from pyomo.environ import *
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
import matplotlib.pyplot as plt
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
from dynamic_opt import transition_block_rule, transition_init_block_rule, transition_block_ttbound, transition_init_block_ttbound
# from schedule_opt import scheduling_milp_block_rule

p_list = ['A', 'B', 'C', 'D', 'E']

def get_trans_time_bound(initial_trans=True, cinit=0.5):
    """
    Compute the lower bound of transition time, which should be set in both 
    scheduling problem and each dynamic subproblem. 

    Arg:
        initlial_trans [Boolean]: If it's getting bounds for initial transition.
        If initlial_trans=True, returned dict: {'A': theta}; 
        If initlial_trans=False, returned dict: {('A', 'B'): theta}; 
    """
    model = ConcreteModel()

    # Set the products for this transition.
    # Create an empty block.
    model.I = Set(initialize=p_list)      # Products
    sp_solver = SolverFactory('ipopt')

    if not initial_trans:
        def trans_time_bound_cal(b, sp, ep):
            if sp == ep:
                return
            else:
                b.start_product = sp
                b.end_product = ep
                transition_block_ttbound(b)
        model.transitions = Block(model.I, model.I, rule=trans_time_bound_cal)

        tt_bounds_dict = {}
        for i in model.I:
            for ip in model.I:
                if i != ip:
                    results = sp_solver.solve(model.transitions[i, ip], tee=False)
                    if results.solver.termination_condition == TerminationCondition.optimal:
                        tt_min = model.transitions[i, ip].trans_time.value
                    else:
                        print("Minimum transition time too short! Eps is assigned")
                        tt_min = 1e-5
                    tt_bounds_dict[i, ip] = (tt_min, 20)
    else:
        # Define the block
        def trans_time_init_bound_cal(b, ep):
            b.end_product = ep
            transition_init_block_ttbound(b)
            b.cinit.set_value(cinit)

        model.transitions_init_tt = Block(model.I, rule=trans_time_init_bound_cal)
        
        tt_bounds_dict = {}
        for i in model.I:
            results = sp_solver.solve(model.transitions_init_tt[i], tee=False)
            if results.solver.termination_condition == TerminationCondition.optimal:
                tt_min = model.transitions_init_tt[i].trans_time.value
            else:
                print("Minimum transition time too short! Eps is assigned")
                tt_min = 1e-5
            tt_bounds_dict[i] = (tt_min, 20)

    return tt_bounds_dict
        
def get_trans_cost_bound(initial_trans=False, cinit=0.5):
    """
    Compute the lower bound of transition cost, which should be set in both 
    scheduling problem and each dynamic subproblem. 

    Arg:
        initlial_trans [Boolean]: If it's getting bounds for initial transition.
        If initlial_trans=True, returned dict: {'A': theta}; 
        If initlial_trans=False, returned dict: {('A', 'B'): theta}; 
    """
    model = ConcreteModel()

    # Set the products for this transition.
    # Create an empty block.
    model.I = Set(initialize=p_list)      # Products
    sp_solver = SolverFactory('ipopt')
    trans_time_bounds = get_trans_time_bound(initial_trans, cinit)
    # print("Initial transition time bounds:", trans_time_bounds)
    eta_bounds_dict = {}

    if initial_trans == False: # Regular transition blocks
        def transition_block_indexed_rule(b, sp, ep):
            if sp == ep:
                return
            else:
                b.start_product = sp
                b.end_product   = ep
                transition_block_rule(b)
                b.trans_time.setlb(trans_time_bounds[sp, ep][0])
                
        model.transitions = Block(model.I, model.I, rule=transition_block_indexed_rule)
        
        for i in model.I:
            for ip in model.I:
                if i != ip:
                    # Solve for lower bound
                    # print(f"Lower bound of eta from {i} to {ip}...")
                    sp_solver.solve(model.transitions[i, ip], tee=False)
                    eta_min = value(model.transitions[i, ip].obj) * 0.9

                    # Solve for upper bound (at the minimum transition time)
                    # print(f"Upper bound of eta from {i} to {ip}...")
                    model.transitions[i, ip].trans_time.fix(trans_time_bounds[i, ip][0]) # 
                    sp_solver.solve(model.transitions[i, ip], tee=False)
                    eta_max = value(model.transitions[i, ip].obj) * 1.5
                    eta_bounds_dict[i, ip] = (eta_min, eta_max)

    else: 
        def transition_block_indexed_rule(b, ep):
            b.end_product = ep
            transition_init_block_rule(b)
            b.trans_time.setlb(trans_time_bounds[ep][0])
            b.cinit.set_value(cinit)
            # b.qinit.set_value(qinit)

        # print("cinit=", cinit)
        model.transitions_init = Block(model.I, rule=transition_block_indexed_rule)
        
        for i in model.I:
            # Solve for lower bound
            sp_solver.solve(model.transitions_init[i], tee=False)
            eta_init_min = value(model.transitions_init[i].obj) * 0.9

            # Solve for upper bound (at the minimum transition time)
            model.transitions_init[i].trans_time.fix(trans_time_bounds[i][0])
            sp_solver.solve(model.transitions_init[i], tee=False)
            eta_init_max = value(model.transitions_init[i].obj) * 1.5
            eta_bounds_dict[i] = (eta_init_min, eta_init_max)

    return eta_bounds_dict


def solve_mp_milp(scheduling, PoolSolutions=1, PoolSearchMode=2, MIPFocus=None, MIPGap=None):
    '''
    Solve the MILP master problem using gurobi_persistent, obtaining multiple solutions
    (transition schedule) and returning the best solution (highest objective value) along 
    with the pool of all solutions.
    
    Parameters:
        scheduling: The Pyomo (or similar) model instance representing the master problem.
        PoolSolutions: Number of solutions to store in the pool.
        PoolSearchMode: How the solution pool is constructed (Gurobi parameter).

    Returns:
        best_assign: Dictionary corresponding to the best solution (highest objective value).
        pool: A list of dictionaries representing the transition assignments for each solution.
    '''
    
    # Initialize and attach the instance to the solver
    opt = SolverFactory('gurobi_persistent')
    
    opt.set_instance(scheduling)
    opt.set_gurobi_param('OutputFlag', 0)
    
    # Set Gurobi parameters for generating the solution pool
    opt.set_gurobi_param('PoolSolutions', PoolSolutions)
    # opt.set_gurobi_param('FeasibilityTol', 1e-8)
    # opt.set_gurobi_param('PoolSearchMode', PoolSearchMode)
    if MIPFocus is not None:
        opt.set_gurobi_param('MIPFocus', MIPFocus)
    if MIPGap is not None:
        opt.set_gurobi_param('MIPGap', MIPGap)

    # now solve quietly
    results = opt.solve(report_timing=True, tee=False)

    # Print Gurobi-reported solve time (wall-clock)
    gurobi_model = opt._solver_model
    mp_time = gurobi_model.Runtime
    
    # Retrieve the number of solutions in the pool
    sol_count = gurobi_model.SolCount
    
    pool = []
    best_assign = None
    best_obj = -float('inf')
    
    # Iterate over each solution in the pool
    for num in range(sol_count):
        # Set the current solution from the pool
        gurobi_model.setParam('SolutionNumber', num)
        
        # Retrieve the objective value for the current solution
        obj_val = gurobi_model.PoolObjVal
        
        # Retrieve variable assignments for the current solution
        trans_time = {}
        for k in scheduling.Kt:
            for i in scheduling.I:
                for ip in scheduling.I:
                    # Handle initialization for the first time slot
                    if k == 1:
                        z_init_name = opt.get_var_attr(scheduling.z_init[i], 'VarName')
                        z_init = gurobi_model.getVarByName(z_init_name).Xn
                        if z_init >= 0.9:
                            tt_init_name = opt.get_var_attr(scheduling.tt_init[i], 'VarName')
                            trans_time[('*', i, 0)] = gurobi_model.getVarByName(tt_init_name).Xn
                    # Retrieve decision variables from the main schedule
                    z_name = opt.get_var_attr(scheduling.z[i, ip, k], 'VarName')
                    z = gurobi_model.getVarByName(z_name).Xn
                    if z >= 0.9:
                        tt_name = opt.get_var_attr(scheduling.tt[i, ip, k], 'VarName')
                        trans_time[(i, ip, k)] = gurobi_model.getVarByName(tt_name).Xn
        
        # Add the current solution to the pool list
        pool.append(trans_time)
        
        # Check if the current solution is the best so far.
        if obj_val > best_obj:
            best_obj = obj_val
            best_assign = trans_time

    # Extract Gurobi's reported optimality gap (relative)
    try:
        gap = gurobi_model.MIPGap
    except:
        gap = 0.0
            
    return best_assign, gap, mp_time, results


# def generate_sample(scheduling, sp_model, cinit, assignment_map, MIPFocus, MIPGap):
#     sp_solver = SolverFactory('ipopt')
#     convergence = False
#     itr = 0
#     UB_list = []
#     LB_list = []
#     gap_list = []
#     gap = 1e4
#     mp_time_list = []
#     while not convergence:
#         itr += 1
#         best_assign, gap_mp, mp_time, _ = solve_mp_milp(scheduling, MIPFocus=MIPFocus, MIPGap=MIPGap)
#         assignment = best_assign
#         mp_time_list.append(mp_time)

#         upper_bound = value(scheduling.obj)
#         UB_list.append(upper_bound)
        
#         # Get transition schedule
#         trans_pairs = list(assignment.keys())
        
#         lower_bound = 0
#         sp_sol_time = 0
#         for i, ip, k in trans_pairs:
#             # 1. fix the transition time
#             trans_time_ijk = assignment[i, ip, k]
    
#             if k == 0:
#                 if hasattr(sp_model.transitions_init[ip], 'trans_time_fix'):
#                     sp_model.transitions_init[ip].del_component(sp_model.transitions_init[ip].trans_time_fix)
#                 sp_model.transitions_init[ip].trans_time_fix = Constraint(expr=sp_model.transitions_init[ip].trans_time==trans_time_ijk)
#                 # 2. solve the subproblem
#                 result_sp_init = sp_solver.solve(sp_model.transitions_init[ip], tee=False)
#                 # 3. extract the dual value
#                 dual_val_init = - sp_model.transitions_init[ip].dual[sp_model.transitions_init[ip].trans_time_fix]
#                 # 4. add this benders cut to all slots
#                 phi_init = value(sp_model.transitions_init[ip].obj)
#                 cut_init_expr = scheduling.eta_init[ip] >= phi_init - dual_val_init * (scheduling.tt_init[ip] - trans_time_ijk)
#                 scheduling.benders_cuts.add(cut_init_expr)
#                 # 5. lower bound
#                 lower_bound += - phi_init
#             else:
#                 if hasattr(sp_model.transitions[i, ip], 'trans_time_fix'):
#                     sp_model.transitions[i, ip].del_component(sp_model.transitions[i, ip].trans_time_fix)
#                 sp_model.transitions[i, ip].trans_time_fix = Constraint(expr=sp_model.transitions[i, ip].trans_time==trans_time_ijk)
#                 # 2. solve the subproblem
#                 result_sp = sp_solver.solve(sp_model.transitions[i, ip], tee=False)
#                 # 3. extract the dual value
#                 dual_val = - sp_model.transitions[i, ip].dual[sp_model.transitions[i, ip].trans_time_fix]
#                 # 4. add this benders cut to all slots
#                 phi_ij = value(sp_model.transitions[i, ip].obj)
#                 for kk in scheduling.Kt:
#                     cut_expr = scheduling.eta[i, ip, kk] >= phi_ij - dual_val * (scheduling.tt[i, ip, kk] - trans_time_ijk)
                    
#                     scheduling.benders_cuts.add(cut_expr)
#                 # 5. lower bound
#                 lower_bound += - phi_ij
        
#         lower_bound += value(scheduling.obj) + value(scheduling.expr_var_transition_cost) + value(scheduling.expr_var_transition_init_cost)

#         LB_list.append(lower_bound)
#         gap = (upper_bound - lower_bound) / abs(lower_bound)
#         gap_list.append(gap)
#         print(f"Iteration {itr}: UB={upper_bound:.2f}, LB={lower_bound:.2f}, MP gap={gap_mp:.6f}, Benders gap={gap:.6f}")
    
#         # Check the convergence
#         if gap <= 1e-4 or itr >= 150:
#             convergence = True
#             print('*** Congratulations!!! ***')

#     best_assign = sorted([ind for ind, var in scheduling.z.items() if var.value > 0.9], key=lambda x: x[2])
#     print(best_assign)
#     permutation = tuple([best_assign[i][0] for i in range(len(best_assign))] + [best_assign[-1][1]])

#     print('Building features...')
#     norm_demands = np.array([value(scheduling.d[i]) for i in sp_model.I])
#     print(norm_demands)
#     features = np.concatenate((norm_demands, np.array([cinit]))) 
#     label = assignment_map[permutation]
#     print(f"New data point: X={features}, y={label}")

#     print(f"Total master problem runtime: {sum(mp_time_list)}s")

#     obj_value = upper_bound
    
#     return features, label, mp_time_list, obj_value



