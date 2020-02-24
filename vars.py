import torch
### General settings
TIME_STEP_SIZE = 10*60# How many seconds are in one of our timeteps? For example if we want every minute, set this to 60
NUM_HOURS = 31*24
NUM_TIME_STEPS = int(NUM_HOURS*3600//TIME_STEP_SIZE) # A total of 12 hours computed every second

##### RL Agent parameters
NUM_EPISODES = 500 # Number of episodes
EPSILON = 1 # For epsilon-greedy approach
EPS_DECAY = 0.9998
LEARNING_RATE = 0.00025
GAMMA = 0.99
TARGET_UPDATE = 10
BATCH_SIZE = 16
N_ACTIONS = 2
INPUT_DIMS = 4
FC_1_DIMS = 300
FC_2_DIMS = 600
FC_3_DIMS = FC_2_DIMS # If we don't want a third layer, set this to FC_2_DIMS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TAU = 0.01 # For soft update


##### Environment parameters
E_PRICE = 1 # Price per kwh (expressed in price/kwminute)
COMFORT_PENALTY = 2 # Penalty applied when going outside of "comfort" bounds
T_MIN = 19.5 # Minimum temperature that should be achieved inside of the building
T_MAX = 22.5 # Maximum temperature that should be achieved inside of the building
C_I = 2.07*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
C_E = 3.24*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
R_IA = 5.29e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_IE = 0.909e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_EA = 4.47e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
A_w = 7.89 # Window surface area
T_BOUND_MIN = 10 # We will not model anything under 15 in order to limit the number of spaces
T_BOUND_MAX = 30 # We will not model anything over 30 in order to limit the number of spaces
T_AMBIENT = 2 # Ambient temperature
NOMINAL_HEAT_PUMP_POWER = 2000 # 2kW based on some quick loockup of purchaseable heat pumps
TEMPERATURE_ROUNDING = 10000 #We are rounding up to 1/TEMPERATURE_ROUNDING
PRECISION = (T_BOUND_MAX-T_BOUND_MIN)*TEMPERATURE_ROUNDING

