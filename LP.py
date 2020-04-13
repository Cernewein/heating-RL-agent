import pandas as pd
import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB


def heat_pump_power(phi_e, ambient_temperature):
    """Takes an electrical power flow and converts it to a heat flow.

    :param phi_e: The electrical power
    :type phi_e: Float
    :return: Returns the heat flow as an integer
    """
    return phi_e * (0.0606 * ambient_temperature + 2.612)

TIME_STEP_SIZE = 60*60# How many seconds are in one of our timeteps? For example if we want every minute, set this to 60
NUM_HOURS = 31*24
NUM_TIME_STEPS = int(NUM_HOURS*3600//TIME_STEP_SIZE) # A total of 12 hours computed every second
T_MIN = 19.5 # Minimum temperature that should be achieved inside of the building
T_MAX = 22.5 # Maximum temperature that should be achieved inside of the building
C_I = 2.07*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
R_IA = 5.29e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
A_w = 7.89 # Window surface area
NOMINAL_HEAT_PUMP_POWER = 2000 # 2kW based on some quick loockup of purchaseable heat pumps
COMFORT_PENALTY = 10
STORAGE_CAPACITY = 4000 # Number of Watts that can be stored in the battery
C_MAX = 2750 * TIME_STEP_SIZE / 3600# Power in watt that the charging can provide divided by the time step size
D_MAX = 2750 * TIME_STEP_SIZE / 3600# Power in watt that the discharging can provide divided by the time step size
ETA = 0.95 # Charging and discharging efficiency of the battery
MIN_STORAGE = 0.1 * STORAGE_CAPACITY
INITIAL_STORAGE = 1000 # Set here what is initially stored inside of the battery
SELL_PRICE_DISCOUNT = 0.9 # Percentage of buying price, so selling price = SELL_PRICE_DISCOUNT*buying_price
PV_EFFICIENCY = 0.15
PV_SURFACE = 30
T = NUM_TIME_STEPS
set_T = range(0,T-1)

# Create models
m = gp.Model('MIP')

initial_day = 0
# Create Variables
ambient_temperatures = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[initial_day*24:initial_day*24+NUM_HOURS+1,2].reset_index(drop=True)

sun_powers = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[initial_day*24:initial_day*24+NUM_HOURS+1,3].reset_index(drop=True)

prices = pd.read_csv('data/environment/2014_DK2_spot_prices.csv',
                                  header = 0).iloc[initial_day*24:initial_day*24+NUM_HOURS+1,1].reset_index(drop=True)


T_a = {t: ambient_temperatures[(t * TIME_STEP_SIZE)//3600] for t in set_T}
Phi_s = {t: sun_powers[(t * TIME_STEP_SIZE)//3600] for t in set_T}
P_s = {t: sun_powers[(t * TIME_STEP_SIZE)//3600] * PV_EFFICIENCY * PV_SURFACE * TIME_STEP_SIZE/3600  for t in set_T} # Power generation of PV
print(P_s)
P = {t: prices[(t * TIME_STEP_SIZE)//3600] for t in set_T}


# Defining decision variables

x_vars = {t:m.addVar(vtype=GRB.CONTINUOUS,lb=0, ub=NOMINAL_HEAT_PUMP_POWER, name="x_{}".format(t)) for t in set_T}#
T_i = {t:m.addVar(vtype=GRB.CONTINUOUS, name="T_{}".format(t)) for t in range(0,T)} #, lb = T_MIN, ub= T_MAX
nu = {t:m.addVar(vtype=GRB.CONTINUOUS, name="nu_{}".format(t)) for t in range(0,T)}
f = {t:m.addVar(vtype=GRB.CONTINUOUS, lb=-D_MAX, ub = C_MAX, name="f_{}".format(t)) for t in set_T}
#d = {t:m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub = D_MAX, name="d_{}".format(t)) for t in set_T}
B = {t:m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub = STORAGE_CAPACITY,name="B_{}".format(t)) for t in range(0,T)}
phi = {t:m.addVar(vtype=GRB.CONTINUOUS,ub = C_MAX + NOMINAL_HEAT_PUMP_POWER*TIME_STEP_SIZE//3600, name="phi_{}".format(t)) for t in set_T}
i = {t:m.addVar(vtype=GRB.BINARY, name="i_{}".format(t)) for t in set_T} # 1 if power from grid is negative 0 otherwise
#Defining the constraints

# <= contraints

constraints_less_eq = {t: m.addConstr(
    lhs = T_MIN,
    sense = GRB.LESS_EQUAL,
    rhs=T_i[t] + nu[t],
    name='max_constraint_{}'.format(t)
) for t in range(0,T)}

constraints_less_eq_power_from_grid = {t: m.addConstr(
    lhs = -phi[t]/10000,
    sense = GRB.LESS_EQUAL,
    rhs=i[t],
    name='power_from_grid_les_eq_{}'.format(t)
) for t in set_T}


# >= contraints

constraints_greater_eq = {t: m.addConstr(
    lhs = T_MAX,
    sense = GRB.GREATER_EQUAL,
    rhs=T_i[t] - nu[t],
    name='min_constraint_{}'.format(t)
) for t in range(0,T)}

constraints_greater_eq_power_from_grid = {t: m.addConstr(
    lhs =1-phi[t]/10000,
    sense = GRB.GREATER_EQUAL,
    rhs= i[t],
    name='power_from_grid_greater_eq_{}'.format(t)
) for t in set_T}

constraints_greater_eq_battery = {t: m.addConstr(
    lhs = B[t],
    sense = GRB.GREATER_EQUAL,
    rhs= MIN_STORAGE,
    name='battery_greater_eq_{}'.format(t)
) for t in range(0,T)}

# == contraints

constraints_eq = {t: m.addConstr(
    lhs = T_i[t],
    sense = GRB.EQUAL,
    rhs= T_i[t-1] + TIME_STEP_SIZE*(1 / (R_IA * C_I) * (T_a[t-1] - T_i[t-1]) + \
                heat_pump_power(x_vars[t-1], T_a[t-1])/C_I + A_w*Phi_s[t-1]/C_I),
    name='equality_constraint_{}'.format(t)
) for t in range(1,T)}

constraints_eq[0] = m.addConstr(
    lhs = T_i[0],
    sense = GRB.EQUAL,
    rhs= 21,
    name='equality_constraint_{}'.format(0)
)

constraints_eq_power = {t: m.addConstr(
    lhs = phi[t],
    sense = GRB.EQUAL,
    rhs= f[t] + x_vars[t]*TIME_STEP_SIZE/3600 - P_s[t],
    name='equality_constraint_power_{}'.format(t)
) for t in set_T}

constraints_eq_battery = {t: m.addConstr(
    lhs = B[t],
    sense = GRB.EQUAL,
    rhs= B[t-1] + f[t-1],
    name='equality_constraint_battery_{}'.format(t)
) for t in range(1,T)}

constraints_eq_battery[0] = m.addConstr(
    lhs = B[0],
    sense = GRB.EQUAL,
    rhs= 1000,
    name='equality_battery_{}'.format(0)
)
# Objective

objective = gp.quicksum(f[t]/1e6 * TIME_STEP_SIZE/3600 + phi[t]*P[t]*(1-i[t]*SELL_PRICE_DISCOUNT)/1e6 + COMFORT_PENALTY*nu[t] for t in set_T)
m.ModelSense = GRB.MINIMIZE
m.setObjective(objective)
m.optimize()

opt_df = pd.DataFrame.from_dict(x_vars, orient='index', columns= ["variable_object"])

cost=0
for t,varname in enumerate(phi.values()):
    cost+=m.getVarByName(varname.VarName).x*P[t]

print(cost/1e6)
