from pyomo.environ import *
from pyomo.opt import SolverFactory
from src.tool import get_trans_time_bound, get_trans_cost_bound
import numpy as np

def scheduling_milp_block_rule(block):
    # ========================
    # 1. Sets and Parameters
    # ========================
    block.I = Set(initialize=['A', 'B', 'C', 'D', 'E'])      # Products
    block.K = Set(initialize=[1, 2, 3, 4, 5])                # Slots
    block.Kt = Set(initialize=[1, 2, 3, 4])                # Index for slots with transition

    block.Tc = Param(initialize=7*24, doc='Cycle time')

    # Transition costs table ct[i,j]
    ctrans_data = {('A','A'): 0, ('A','B'): 1140, ('A','C'): 960, ('A','D'): 1240, ('A','E'): 1450, 
                   ('B','A'): 880, ('B','B'): 0, ('B','C'): 470, ('B','D'): 420, ('B','E'): 460,
                   ('C','A'): 1400, ('C','B'): 450, ('C','C'): 0, ('C','D'): 1390, ('C','E'): 360, 
                   ('D','A'): 530, ('D','B'): 490, ('D','C'): 460, ('D','D'): 0, ('D','E'): 320, 
                   ('E','A'): 1370, ('E','B'): 850, ('E','C'): 480, ('E','D'): 520, ('E','E'): 0}
    block.ctrans = Param(block.I, block.I, initialize=ctrans_data)

    print("Adding noise...")
    d_noise = lambda scale: np.random.uniform(low=-scale, high=scale)
        
    d_data    = {'A': 3000+d_noise(300.0), 
                 'B': 5000+d_noise(500.0), 
                 'C': 6000+d_noise(600.0), 
                 'D': 10000+d_noise(1000.0),       
                 'E': 14000+d_noise(1400.0)}      # Demand rate [mol/week]
    pr_data   = {'A': 300, 
                 'B': 500, 
                 'C': 200, 
                 'D': 100,
                 'E': 150}    # Price [$ / mol]
    coper_data   = {'A': 13, 
                    'B': 22, 
                    'C': 35, 
                    'D': 29,
                    'E': 25}     # Operation cost
    qrate_data= {'A': 200, 'B': 100, 'C': 400, 'D': 1000, 'E': 2500}    # Flowrate [lt/h]
    conc_data = {'A': 0.24, 'B': 0.2, 'C': 0.3032, 'D': 0.393, 'E': 0.50}   # Concentration

    block.d     = Param(block.I, initialize=d_data, mutable=True)
    block.pr    = Param(block.I, initialize=pr_data)
    block.cinv  = Param(initialize=0.5)
    block.coper = Param(block.I, initialize=coper_data)
    block.qrate = Param(block.I, initialize=qrate_data)
    block.conc  = Param(block.I, initialize=conc_data)
    block.Cf    = Param(initialize=1)

    # ========================
    # 2. Variables
    # ========================
    # Transition times table tt[i, j, k]
    block.tt = Var(block.I, block.I, block.Kt, initialize=5,
                   doc='Transition time from i to j in slot k')
    
    # ----- ***** -----
    block.tt_init = Var(block.I, bounds=(0, 20), initialize=5,
                        doc='Transition time from immediate state to i')
    
    # Predefined bounds
    print("In scheduling_opt, calculating transition time bounds...")

    # !! Please use this function for the bounds of transition time if you've changed any parameter !!
    # trans_time_bounds = get_trans_time_bound(initial_trans=False)

    trans_time_bounds = {('A', 'B'): (1.9301999767653213, 20),
                        ('A', 'C'): (0.09999999250601427, 20),
                        ('A', 'D'): (0.2542104008407287, 20),
                        ('A', 'E'): (0.5560439545622674, 20),
                        ('B', 'A'): (0.09999999173782066, 20),
                        ('B', 'C'): (0.14611884648351978, 20),
                        ('B', 'D'): (0.3075880122737712, 20),
                        ('B', 'E'): (0.6096577067532046, 20),
                        ('C', 'A'): (1.640646236265163, 20),
                        ('C', 'B'): (3.5789266678101095, 20),
                        ('C', 'D'): (0.1611620765626465, 20),
                        ('C', 'E'): (0.4625660238059599, 20),
                        ('D', 'A'): (2.763930209937099, 20),
                        ('D', 'B'): (4.7101465649075145, 20),
                        ('D', 'C'): (1.1136808973025318, 20),
                        ('D', 'E'): (0.30060149153394167, 20),
                        ('E', 'A'): (3.4024951116303175, 20),
                        ('E', 'B'): (5.355457654674393, 20),
                        ('E', 'C'): (1.7448614836029075, 20),
                        ('E', 'D'): (0.6257523835692085, 20)}
    
    print("In scheduling_opt, calculating initial transition time bounds...")
    trans_init_time_bounds = get_trans_time_bound(initial_trans=True, cinit=block.cinit.value)
    for i in block.I:
        block.tt_init[i].setlb(trans_init_time_bounds[i][0])
        for ip in block.I:
            if i != ip:
                for k in block.Kt:
                    block.tt[i, ip, k].setlb(trans_time_bounds[i, ip][0]+1e-4) # Slacks for avoiding numerical issues

    # Scheduling and profit variables
    block.t  = Var(block.I, within=PositiveReals, doc='Processing time of product i')
    block.w  = Var(block.I, within=PositiveReals, doc='Amount of product i produced')
    block.ts = Var(block.K, within=NonNegativeReals, doc='Start times of slot k')
    block.te = Var(block.K, within=NonNegativeReals, doc='End times of slot k')
    block.process_time = Var(block.K, within=NonNegativeReals, doc='Process times of slot k')
    block.theta = Var(block.I, block.K, within=NonNegativeReals, doc='Time for manufacturing product i in slot k')

    block.transition_time = Var(block.Kt, within=NonNegativeReals)
    block.production_rate = Var(block.I, within=NonNegativeReals)
    block.production_quan = Var(block.I, block.K, within=NonNegativeReals)
    block.inv = Var(block.I, block.K, within=NonNegativeReals)

    # Binary variables for scheduling
    block.y  = Var(block.I, block.K, within=Binary)  # product assignment
    block.z  = Var(block.I, block.I, block.Kt, within=Binary)  # transition indicator

    # ----- ***** -----
    block.z_init  = Var(block.I, within=Binary)  # initial transition indicator

    block.eta = Var(block.I, block.I, block.Kt, within=NonNegativeReals)
    block.eta_init = Var(block.I, within=NonNegativeReals)
    # Predefined bounds
    print("Configuring transition cost bounds...")

    # !! Please use this function for the bounds of transition cost if you've changed any parameter !!
    # trans_cost_bounds = get_trans_cost_bound(initial_trans=False)

    trans_cost_bounds = {('A', 'B'): (246.89527896509455, 14416.180581799312),
                        ('A', 'C'): (543.5908413453586, 1351250.8389042884),
                        ('A', 'D'): (5272.1681731460985, 3025604.723892082),
                        ('A', 'E'): (26196.1841982973, 2593922.9501959155),
                        ('B', 'A'): (95.33519404539788, 474345.70311388606),
                        ('B', 'C'): (1280.5885287543379, 2299613.800081855),
                        ('B', 'D'): (6970.422889875159, 3661068.534460131),
                        ('B', 'E'): (29489.45100852032, 2844924.7943670675),
                        ('C', 'A'): (1798.1821124802823, 49014.30175265682),
                        ('C', 'B'): (1994.1890814452443, 27624.790628871753),
                        ('C', 'D'): (2722.8308269974996, 1918002.550520132),
                        ('C', 'E'): (20565.30906237096, 2156579.4594117254),
                        ('D', 'A'): (11307.50605511965, 87754.49848975886),
                        ('D', 'B'): (14106.20253298135, 46953.764028541525),
                        ('D', 'C'): (3647.314149208716, 133780.73645405564),
                        ('D', 'E'): (9582.74197401395, 1399436.8696313147),
                        ('E', 'A'): (62131.86578124682, 157469.287660578),
                        ('E', 'B'): (75435.41952856425, 136225.50422700332),
                        ('E', 'C'): (36184.63315093652, 231683.83973322122),
                        ('E', 'D'): (12799.846196867009, 469802.0387461736)}
    print("Configuring initial transition cost bounds...")
    trans_init_cost_bounds = get_trans_cost_bound(initial_trans=True, cinit=block.cinit.value)
                    
    block.sell = Var(block.I, block.K, within=NonNegativeReals)

    block.bi_tt = Var(block.I, block.I, block.Kt, within=NonNegativeReals)
    block.bi_tt_init = Var(block.I, within=NonNegativeReals)

    block.bi_eta = Var(block.I, block.I, block.Kt, within=NonNegativeReals)
    block.bi_eta_init = Var(block.I, within=NonNegativeReals)

    # ========================
    # 3. Objective Expressions
    # ========================
    block.expr_revenue = Expression(expr=sum(block.pr[i] * block.sell[i, k]
                                              for i in block.I for k in block.K))
    block.expr_operation_cost = Expression(expr=sum(block.coper[i] * block.production_quan[i, k]
                                                     for i in block.I for k in block.K))
    block.expr_inventory_cost = Expression(expr=sum(block.cinv * block.inv[i, k]
                                                     for i in block.I for k in block.K))
    block.expr_fixed_transition_cost = Expression(
        expr=sum(block.ctrans[i, ip] * block.z[i, ip, k]
                 for i in block.I for ip in block.I for k in block.Kt))
    block.expr_var_transition_cost = Expression(
        expr=sum(block.bi_eta[i, ip, k]
                 for i in block.I for ip in block.I for k in block.Kt))
    
    block.expr_var_transition_init_cost = Expression(
        expr=sum(block.bi_eta_init[i]
                 for i in block.I))

    block.obj_expr = Expression(expr=block.expr_revenue - block.expr_operation_cost - block.expr_inventory_cost - \
             block.expr_fixed_transition_cost - block.expr_var_transition_cost - block.expr_var_transition_init_cost)
    # Final objective: maximize profit
    block.obj = Objective( 
        expr=block.obj_expr,
        sense=maximize)

    # ========================
    # 4. Constraints
    # ========================

    # (*) Linearization of bilinear term
    def transition_cost_linear_rule1(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_eta[i, ip, k] >= trans_cost_bounds[i, ip][0] * block.z[i, ip, k]
    block.transition_cost_linear_con1 = Constraint(block.I, block.I, block.Kt, rule=transition_cost_linear_rule1)
    
    def transition_cost_linear_rule2(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_eta[i, ip, k] <= trans_cost_bounds[i, ip][1] * block.z[i, ip, k]
    block.transition_cost_linear_con2 = Constraint(block.I, block.I, block.Kt, rule=transition_cost_linear_rule2)
    
    def transition_cost_linear_rule3(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_eta[i, ip, k] >= block.eta[i, ip, k] - trans_cost_bounds[i, ip][1] * (1 - block.z[i, ip, k])
    block.transition_cost_linear_con3 = Constraint(block.I, block.I, block.Kt, rule=transition_cost_linear_rule3)
    
    def transition_cost_linear_rule4(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_eta[i, ip, k] <= block.eta[i, ip, k] - trans_cost_bounds[i, ip][0] * (1 - block.z[i, ip, k])
    block.transition_cost_linear_con4 = Constraint(block.I, block.I, block.Kt, rule=transition_cost_linear_rule4)
    
    def transition_init_cost_linear_rule1(block, i):
        return block.bi_eta_init[i] >= trans_init_cost_bounds[i][0] * block.z_init[i]
    block.transition_init_cost_linear_con1 = Constraint(block.I, rule=transition_init_cost_linear_rule1)
    
    def transition_init_cost_linear_rule2(block, i):
        return block.bi_eta_init[i] <= trans_init_cost_bounds[i][1] * block.z_init[i]
    block.transition_init_cost_linear_con2 = Constraint(block.I, rule=transition_init_cost_linear_rule2)
    
    def transition_init_cost_linear_rule3(block, i):
        return block.bi_eta_init[i] >= block.eta_init[i] - trans_init_cost_bounds[i][1] * (1 - block.z_init[i])
    block.transition_init_cost_linear_con3 = Constraint(block.I, rule=transition_init_cost_linear_rule3)
    
    def transition_init_cost_linear_rule4(block, i):
        return block.bi_eta_init[i] <= block.eta_init[i] - trans_init_cost_bounds[i][0] * (1 - block.z_init[i])
    block.transition_init_cost_linear_con4 = Constraint(block.I, rule=transition_init_cost_linear_rule4)
    # ----------------
    def transition_time_linear_rule1(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_tt[i, ip, k] >= trans_time_bounds[i, ip][0] * block.z[i, ip, k]
    block.transition_time_linear_con1 = Constraint(block.I, block.I, block.Kt, rule=transition_time_linear_rule1)
    
    def transition_time_linear_rule2(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_tt[i, ip, k] <= trans_time_bounds[i, ip][1] * block.z[i, ip, k]
    block.transition_time_linear_con2 = Constraint(block.I, block.I, block.Kt, rule=transition_time_linear_rule2)
    
    def transition_time_linear_rule3(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_tt[i, ip, k] >= block.tt[i, ip, k] - trans_time_bounds[i, ip][1] * (1 - block.z[i, ip, k])
    block.transition_time_linear_con3 = Constraint(block.I, block.I, block.Kt, rule=transition_time_linear_rule3)
    
    def transition_time_linear_rule4(block, i, ip, k):
        if i == ip:
            return Constraint.Skip
        return block.bi_tt[i, ip, k] <= block.tt[i, ip, k] # - trans_time_bounds[i, ip][0] * (1 - block.z[i, ip, k])
    block.transition_time_linear_con4 = Constraint(block.I, block.I, block.Kt, rule=transition_time_linear_rule4)
    
    def transition_init_time_linear_rule1(block, i):
        return block.bi_tt_init[i] >= trans_init_time_bounds[i][0] * block.z_init[i]
    block.transition_init_time_linear_con1 = Constraint(block.I, rule=transition_init_time_linear_rule1)
    
    def transition_init_time_linear_rule2(block, i):
        return block.bi_tt_init[i] <= trans_init_time_bounds[i][1] * block.z_init[i]
    block.transition_init_time_linear_con2 = Constraint(block.I, rule=transition_init_time_linear_rule2)
    
    def transition_init_time_linear_rule3(block, i):
        return block.bi_tt_init[i] >= block.tt_init[i] - trans_init_time_bounds[i][1] * (1 - block.z_init[i])
    block.transition_init_time_linear_con3 = Constraint(block.I, rule=transition_init_time_linear_rule3)
    
    def transition_init_time_linear_rule4(block, i):
        return block.bi_tt_init[i] <= block.tt_init[i] - trans_init_time_bounds[i][0] * (1 - block.z_init[i])
    block.transition_init_time_linear_con4 = Constraint(block.I, rule=transition_init_time_linear_rule4)

    # (a) Product assignment constraints
    def switch_rule(block, i, ip, k):
        return block.z[i, ip, k] >= block.y[i, k] + block.y[ip, k+1] - 1
    block.switch_con = Constraint(block.I, block.I, block.Kt, rule=switch_rule)

    def switch_rule2(block, i, ip, k):
        return block.z[i, ip, k] <= block.y[i, k]
    block.switch2_con = Constraint(block.I, block.I, block.Kt, rule=switch_rule2)

    def switch_rule3(block, i, ip, k):
        return block.z[i, ip, k] <= block.y[ip, k+1]
    block.switch3_con = Constraint(block.I, block.I, block.Kt, rule=switch_rule3)

    def oneSlot_rule(block, i):
        return sum(block.y[i, k] for k in block.K) >= 1
    block.oneSlot_con = Constraint(block.I, rule=oneSlot_rule)

    def oneTask_rule(block, k):
        return sum(block.y[i, k] for i in block.I) == 1
    block.oneTask_con = Constraint(block.K, rule=oneTask_rule)

    def z_init_rule(block, i):
        return block.z_init[i] == block.y[i, min(block.K)]
    block.z_init_con = Constraint(block.I, rule=z_init_rule)

    # (b) Timing constraints
    def endstartSlot_rule(block, k):
        if k == max(block.K):
            return block.te[k] == block.ts[k] + block.process_time[k]
        return block.te[k] == block.ts[k] + block.process_time[k] + block.transition_time[k]
    block.endstartSlot_con = Constraint(block.K, rule=endstartSlot_rule)

    def procTime_rule(block, k):
        return block.process_time[k] == sum(block.theta[i, k] for i in block.I)
    block.procTime_con = Constraint(block.K, rule=procTime_rule, doc='Process time in slot k')

    def Processtime_rule(block, i):
        return block.t[i] == sum(block.theta[i, k] for k in block.K)
    block.Processtime_con = Constraint(block.I, rule=Processtime_rule)

    def supportEq_rule(block, i, k):
        return block.theta[i, k] <= block.y[i, k] * block.Tc
    block.supportEq_con = Constraint(block.I, block.K, rule=supportEq_rule)

    def transtimes_rule(block, k):
        if k == 1:
            return block.transition_time[k] == sum(block.tt[i, ip, k] * block.z[i, ip, k]
                                                    for i in block.I for ip in block.I) + \
                                                sum(
                                                    block.tt_init[i] * block.z_init[i]
                                                    for i in block.I
                                                )
        else:
            return block.transition_time[k] == sum(block.tt[i, ip, k] * block.z[i, ip, k]
                                                    for i in block.I for ip in block.I)
    block.transtimes_con = Constraint(block.Kt, rule=transtimes_rule)

    # (c) Production and inventory constraints
    def Production_rule(block, i):
        return block.w[i] == block.production_rate[i] * sum(block.theta[i, k] for k in block.K)
    block.Production_con = Constraint(block.I, rule=Production_rule)

    def defpreqn_rule(block, i):
        return block.production_rate[i] == block.qrate[i] * (block.Cf - block.conc[i])
    block.defpreqn_con = Constraint(block.I, rule=defpreqn_rule)

    def production_quan_rule(block, i, k):
        return block.production_quan[i, k] == block.production_rate[i] * block.theta[i, k]
    block.production_quan_con = Constraint(block.I, block.K, rule=production_quan_rule)

    def inventory_rule(block, i, k):
        if k == 1:
            return block.inv[i, k] == block.production_rate[i] * block.theta[i, k] - block.sell[i, k]
        return block.inv[i, k] == block.inv[i, k-1] + block.production_rate[i] * block.theta[i, k] - block.sell[i, k]
    block.inventory_con = Constraint(block.I, block.K, rule=inventory_rule)

    def demand_satis_rule(block, i, k):
        if k < max(block.K):
            return block.sell[i, k] == 0
        else:
            return block.sell[i, max(block.K)] >= block.d[i]
    block.demand_satis_con = Constraint(block.I, block.K, rule=demand_satis_rule)

    # (d) Slot timing linking
    def endStartPN_rule(block, k):
        if k == 1:
            return Constraint.Skip
        return block.ts[k] == block.te[k-1]
    block.endStartPN_con = Constraint(block.K, rule=endStartPN_rule)

    block.startA_con = Constraint(expr = block.ts[1] == 0)

    def Cyctime_rule(block, k):
        return block.te[k] <= block.Tc
    block.Cyctime_con = Constraint(block.K, rule=Cyctime_rule)

    # A constrant list for storing Benders cuts
    block.benders_cuts = ConstraintList()
