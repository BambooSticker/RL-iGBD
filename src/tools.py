from pyomo.environ import *
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
import matplotlib.pyplot as plt
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
from dynamic_opt import transition_block_rule, transition_init_block_rule, transition_block_ttbound, transition_init_block_ttbound

p_list = ['A', 'B', 'C', 'D', 'E']

def get_trans_time_bound(initial_trans=True, cinit=0.5):
    """
    Compute the lower bound of transition time, which should be set in both 
    scheduling problem and each dynamic subproblem. 

    Args:
        initlial_trans [Boolean]: whether getting bounds for initial transition.

    Returns:
        tt_bounds_dict: Dictionary containing bounds of transition times.
            - If initlial_trans=True, returned dict: {'A': theta}; 
            - If initlial_trans=False, returned dict: {('A', 'B'): theta}; 
    """
    model = ConcreteModel()

    # Set the products for this transition.
    # Create an empty block.
    model.I = Set(initialize=p_list)
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

    Args:
        initlial_trans [Boolean]: whether getting bounds for initial transition.

    Returns:
        eta_bounds_dict: Dictionary containing the bounds of transition costs.
            - If initlial_trans=True, returned dict: {'A': theta}; 
            - If initlial_trans=False, returned dict: {('A', 'B'): theta}; 
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
    
    Args:
        scheduling: The Pyomo (or similar) model instance representing the master problem.
        PoolSolutions: Number of solutions to store in the pool.
        PoolSearchMode: How the solution pool is constructed (Gurobi parameter).

    Returns:
        best_assign: Dictionary corresponding to the best solution (highest objective value).
        gap: Float presenting the realized optimality gap of the master problem.
        pool: List of dictionaries representing the transition assignments for each solution.
        results: Solution log returned by Gurobi.
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




