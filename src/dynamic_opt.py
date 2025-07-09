from pyomo.environ import *
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
import matplotlib.pyplot as plt
import numpy as np

max_Q = 5000
nfe = 120
penalty_trans_cost = 0.5
qrate_data= {'A': 200, 'B': 100, 'C': 400, 'D': 1000, 'E': 2500}  # Flowrate [lt/h]
conc_data = {'A': 0.24, 'B': 0.2, 'C': 0.3032, 'D': 0.393, 'E': 0.50}  # Concentration
# ------------------------------------------------------------------------------
# Transition block rule: encapsulates one transition
# ------------------------------------------------------------------------------
def transition_block_rule(b):
    '''
    Transition dynamics from i to j in slot k (k!=1).
    '''
    # Expect that b.start_product and b.end_product are set externally.
    sp = b.start_product
    ep = b.end_product

    b.dual = Suffix(direction = Suffix.IMPORT)

    b.cinit = Param(initialize=conc_data[sp])
    b.qinit = Param(initialize=qrate_data[sp])
    b.cend  = Param(initialize=conc_data[ep])
    b.qend  = Param(initialize=qrate_data[ep])

    # CSTR process parameters
    b.Cf = Param(initialize=1)      # Feed concentration
    b.v  = Param(initialize=5000)   # Reactor volume
    b.kr = Param(initialize=2)      # Reaction rate constant
    
    # Transition time
    b.trans_time = Var(initialize=5, within=NonNegativeReals)
    
    # Define a normalized time set tau in [0, 1]
    b.tau = ContinuousSet(bounds=(0, 1))
    
    # Define state (C) and control (Q) variables over tau
    b.C = Var(b.tau, bounds=(1e-5, 1.5))
    b.Q = Var(b.tau, bounds=(0, max_Q))
    
    # Define the derivative of C with respect to tau
    b.dCdtau = DerivativeVar(b.C, wrt=b.tau)
    
    # ODE: dC/dtau = T * (Q*(Cf - C)/v - kr * C^3)
    def _ode_rule(bm, tau):
        return bm.dCdtau[tau] == bm.trans_time * (bm.Q[tau]*(bm.Cf - bm.C[tau])/bm.v - bm.kr * bm.C[tau]**3)
    b.ODE = Constraint(b.tau, rule=_ode_rule)
    
    # Boundary conditions: at tau=0 and tau=1
    b.C_init = Constraint(expr = b.C[b.tau.first()] == b.cinit)
    b.C_final = Constraint(expr = b.C[b.tau.last()]  == b.cend)
    b.Q_init = Constraint(expr = b.Q[b.tau.first()] == b.qinit)
    b.Q_final = Constraint(expr = b.Q[b.tau.last()]  == b.qend)
    
    # Define the running cost as the integral of deviations over tau
    def cost_rule(bm, tau):
        # return (bm.C[tau] - bm.cend)**2
        return 1 * (bm.Q[tau] - bm.qend)**2
    b.running_cost = Integral(b.tau, rule=cost_rule)
    
    # Overall objective: minimize the transition cost
    b.obj = Objective(expr = penalty_trans_cost * b.trans_time * b.running_cost, sense=minimize)
    
    # Discretize the block using finite differences (Backward Euler)
    TransformationFactory('dae.finite_difference').apply_to(b, nfe=nfe, scheme='BACKWARD')


def transition_init_block_rule(b):
    '''
    Transition dynamics from intermediate state to i in slot 1.
    '''
    # Expect that b.start_product and b.end_product are set externally.
    # sp = b.start_product
    ep = b.end_product

    b.dual = Suffix(direction = Suffix.IMPORT)

    # Set inlet and outlet conditions from global data:
    b.cinit = Param(initialize=0.5, within=NonNegativeReals, mutable=True, doc='Initial concentration')
    b.qinit = Param(initialize=500, within=NonNegativeReals, doc='Initial flowrate')
    b.cend  = Param(initialize=conc_data[ep])
    b.qend  = Param(initialize=qrate_data[ep])

    # CSTR process parameters (can be adjusted if needed)
    b.Cf = Param(initialize=1)      # Feed concentration
    b.v  = Param(initialize=5000)   # Reactor volume
    b.kr = Param(initialize=2)      # Reaction rate constant
    
    # Decision variable for the transition time (T)
    b.trans_time = Var(initialize=5, within=NonNegativeReals)
    
    # Define a normalized time set tau in [0, 1]
    b.tau = ContinuousSet(bounds=(0, 1))
    
    # Define state (C) and control (Q) variables over tau
    b.C = Var(b.tau, bounds=(1e-5, 1.5))
    b.Q = Var(b.tau, bounds=(0, max_Q))
    
    # Define the derivative of C with respect to tau
    b.dCdtau = DerivativeVar(b.C, wrt=b.tau)
    
    # ODE: dC/dtau = T * (Q*(Cf - C)/v - kr * C^3)
    def _ode_rule(bm, tau):
        return bm.dCdtau[tau] == bm.trans_time * (bm.Q[tau]*(bm.Cf - bm.C[tau])/bm.v - bm.kr * bm.C[tau]**3)
    b.ODE = Constraint(b.tau, rule=_ode_rule)
    
    # Boundary conditions: at tau=0 and tau=1
    b.C_init = Constraint(expr = b.C[b.tau.first()] == b.cinit)
    b.C_final = Constraint(expr = b.C[b.tau.last()]  == b.cend)
    b.Q_init = Constraint(expr = b.Q[b.tau.first()] == b.qinit)
    b.Q_final = Constraint(expr = b.Q[b.tau.last()]  == b.qend)
    
    # Define the running cost as the integral of deviations over tau
    def cost_rule(bm, tau):
        return 1 * (bm.Q[tau] - bm.qend)**2
    b.running_cost = Integral(b.tau, rule=cost_rule)
    
    # Overall objective: minimize the transition cost
    b.obj = Objective(expr = penalty_trans_cost * b.trans_time * b.running_cost, sense=minimize)
    
    # Discretize the block using finite differences (Backward Euler)
    TransformationFactory('dae.finite_difference').apply_to(b, nfe=nfe, scheme='BACKWARD')


def transition_block_ttbound(b):
    """
    Model for obtaining the bound of transition time between products.
    """
    # Expect that b.start_product and b.end_product are set externally.
    sp = b.start_product
    ep = b.end_product

    b.dual = Suffix(direction = Suffix.IMPORT)

    b.cinit = Param(initialize=conc_data[sp])
    b.qinit = Param(initialize=qrate_data[sp])
    b.cend  = Param(initialize=conc_data[ep])
    b.qend  = Param(initialize=qrate_data[ep])
    
    # CSTR process parameters
    b.Cf = Param(initialize=1)      # Feed concentration
    b.v  = Param(initialize=5000)   # Reactor volume
    b.kr = Param(initialize=2)      # Reaction rate constant

    # Transition time
    b.trans_time = Var(bounds=(0.1, 30), initialize=1)
    
    # Define a normalized time set tau in [0, 1]
    b.tau = ContinuousSet(bounds=(0, 1))
    
    # Define state (C) and control (Q) variables over tau
    b.C = Var(b.tau, bounds=(1e-5, 1.5))
    b.Q = Var(b.tau, bounds=(0, max_Q))
    
    # Define the derivative of C with respect to tau
    b.dCdtau = DerivativeVar(b.C, wrt=b.tau)
    
    # ODE: using chain rule, dC/dtau = T * (Q*(Cf - C)/v - kr * C^3)
    def _ode_rule(bm, tau):
        return bm.dCdtau[tau] == bm.trans_time * (bm.Q[tau]*(bm.Cf - bm.C[tau])/bm.v - bm.kr * bm.C[tau]**3)
    b.ODE = Constraint(b.tau, rule=_ode_rule)
    
    # Boundary conditions: at tau=0 and tau=1
    b.C_init = Constraint(expr = b.C[b.tau.first()] == b.cinit)
    b.C_final = Constraint(expr = b.C[b.tau.last()]  == b.cend)
    b.Q_init = Constraint(expr = b.Q[b.tau.first()] == b.qinit)
    b.Q_final = Constraint(expr = b.Q[b.tau.last()]  == b.qend)
    
    # Overall objective: minimize the transition time
    b.obj = Objective(expr = b.trans_time, sense=minimize)
    
    # Discretize the block using finite differences (Backward Euler)
    TransformationFactory('dae.finite_difference').apply_to(b, nfe=nfe, scheme='BACKWARD')


def transition_init_block_ttbound(b):
    """
    Model for obtaining the bound of transition time from the intermediate state to a product.
    """
    # Expect that b.start_product and b.end_product are set externally.
    ep = b.end_product

    b.dual = Suffix(direction = Suffix.IMPORT)

    # Set inlet and outlet conditions
    b.cinit = Param(initialize=0.5, within=NonNegativeReals, mutable=True, doc='Initial concentration')
    b.qinit = Param(initialize=0, doc='Initial flowrate')
    b.cend  = Param(initialize=conc_data[ep])
    b.qend  = Param(initialize=qrate_data[ep])
    
    # CSTR process parameters
    b.Cf = Param(initialize=1)      # Feed concentration
    b.v  = Param(initialize=5000)   # Reactor volume
    b.kr = Param(initialize=2)      # Reaction rate constant

    # Transition time
    b.trans_time = Var(bounds=(1e-5, 30), initialize=1)
    
    # Define a normalized time set tau in [0, 1]
    b.tau = ContinuousSet(bounds=(0, 1))
    
    # Define state (C) and control (Q) variables over tau
    b.C = Var(b.tau, bounds=(1e-5, 1.5))
    b.Q = Var(b.tau, bounds=(0, max_Q))
    
    # Define the derivative of C with respect to tau
    b.dCdtau = DerivativeVar(b.C, wrt=b.tau)
    
    # ODE: using chain rule, dC/dtau = T * (Q*(Cf - C)/v - kr * C^3)
    def _ode_rule(bm, tau):
        return bm.dCdtau[tau] == bm.trans_time * (bm.Q[tau]*(bm.Cf - bm.C[tau])/bm.v - bm.kr * bm.C[tau]**3)
    b.ODE = Constraint(b.tau, rule=_ode_rule)
    
    # Boundary conditions: at tau=0 and tau=1
    b.C_init = Constraint(expr = b.C[b.tau.first()] == b.cinit)
    b.C_final = Constraint(expr = b.C[b.tau.last()]  == b.cend)
    b.Q_init = Constraint(expr = b.Q[b.tau.first()] == b.qinit)
    b.Q_final = Constraint(expr = b.Q[b.tau.last()]  == b.qend)
    
    # Overall objective: minimize the transition time
    b.obj = Objective(expr = b.trans_time, sense=minimize)
    
    # Discretize the block using finite differences (Backward Euler)
    TransformationFactory('dae.finite_difference').apply_to(b, nfe=nfe, scheme='BACKWARD')