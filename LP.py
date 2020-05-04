import pandas as pd
import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB
import pickle as pkl


def heat_pump_power(phi_e, ambient_temperature):
    """Takes an electrical power flow and converts it to a heat flow.

    :param phi_e: The electrical power
    :type phi_e: Float
    :return: Returns the heat flow as an integer
    """
    return phi_e * (0.0606 * ambient_temperature + 2.612)

TIME_STEP_SIZE = 10*60# How many seconds are in one of our timeteps? For example if we want every minute, set this to 60
NUM_HOURS = 31*24
NUM_TIME_STEPS = int(NUM_HOURS*3600//TIME_STEP_SIZE) # A total of 12 hours computed every second
T_MIN = 19.5 # Minimum temperature that should be achieved inside of the building
T_MAX = 22.5 # Maximum temperature that should be achieved inside of the building
C_I = 2.07*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
R_IA = 5.29e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
A_w = 7.89 # Window surface area
NOMINAL_HEAT_PUMP_POWER = 2000 # 2kW based on some quick loockup of purchaseable heat pumps
COMFORT_PENALTY = 10
T = NUM_TIME_STEPS
set_T = range(0,T-1)

# Create models
m = gp.Model('MIP')

# Create Variables
ambient_temperatures = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[0:NUM_HOURS+1,2]

sun_powers = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[0:NUM_HOURS+1,3]

prices = pd.read_csv('data/environment/2014_DK2_spot_prices.csv',
                                  header = 0).iloc[0:NUM_HOURS+1,1]

T_a = {t: ambient_temperatures[(t * TIME_STEP_SIZE)//3600] for t in set_T}
Phi_s = {t: sun_powers[(t * TIME_STEP_SIZE)//3600] for t in set_T}
P = {t: prices[(t * TIME_STEP_SIZE)//3600] for t in set_T}


# Defining decision variables

x_vars = {t:m.addVar(vtype=GRB.CONTINUOUS,lb=0, ub=1, name="x_{}".format(t)) for t in set_T}#
T_i = {t:m.addVar(vtype=GRB.CONTINUOUS, name="T_{}".format(t)) for t in range(0,T)} #, lb = T_MIN, ub= T_MAX
nu = {t:m.addVar(vtype=GRB.CONTINUOUS, name="nu_{}".format(t)) for t in range(0,T)}


#Defining the constraints

# <= contraints

constraints_less_eq = {t: m.addConstr(
    lhs = T_MIN,
    sense = GRB.LESS_EQUAL,
    rhs=T_i[t] + nu[t],
    name='max_constraint_{}'.format(t)
) for t in range(0,T)}

# >= contraints

constraints_greater_eq = {t: m.addConstr(
    lhs = T_MAX,
    sense = GRB.GREATER_EQUAL,
    rhs=T_i[t] - nu[t],
    name='min_constraint_{}'.format(t)
) for t in range(0,T)}


# == contraints

constraints_eq = {t: m.addConstr(
    lhs = T_i[t],
    sense = GRB.EQUAL,
    rhs= T_i[t-1] + TIME_STEP_SIZE*(1 / (R_IA * C_I) * (T_a[t-1] - T_i[t-1]) + \
                x_vars[t-1] * heat_pump_power(NOMINAL_HEAT_PUMP_POWER, T_a[t-1])/C_I + A_w*Phi_s[t-1]/C_I),
    name='equality_constraint_{}'.format(t)
) for t in range(1,T)}

constraints_eq[0] = m.addConstr(
    lhs = T_i[0],
    sense = GRB.EQUAL,
    rhs= 21,
    name='equality_constraint_{}'.format(0)
)
# Objective

objective = gp.quicksum(x_vars[t]*P[t]*NOMINAL_HEAT_PUMP_POWER/1e6*TIME_STEP_SIZE/3600 + COMFORT_PENALTY*nu[t] for t in set_T)
m.ModelSense = GRB.MINIMIZE
m.setObjective(objective)
m.optimize()

#opt_df = pd.DataFrame.from_dict(x_vars, orient='index', columns= ["variable_object"])

LP_solution = pd.DataFrame()

inside_temperatures = []
for t,varname in enumerate(T_i.values()):
    inside_temperatures.append(m.getVarByName(varname.VarName).x)

LP_solution['Inside Temperature'] = inside_temperatures

with open('data/output/DDPG/LP_output.pkl', 'wb') as f:
    pkl.dump(LP_solution,f)


power=0
for t,varname in enumerate(x_vars.values()):
    power+=m.getVarByName(varname.VarName).x*TIME_STEP_SIZE/3600*2000

print(power)
