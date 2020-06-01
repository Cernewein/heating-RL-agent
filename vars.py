import torch
### General settings
TIME_STEP_SIZE = 60*60# How many seconds are in one of our timeteps? For example if we want every minute, set this to 60
NUM_HOURS = 31*24
NUM_TIME_STEPS = int(NUM_HOURS*3600//TIME_STEP_SIZE) # A total of 12 hours computed every second

##### RL Agent parameters
NUM_EPISODES = 1000 # Number of episodes
EPSILON = 1 # For epsilon-greedy approach
EPS_DECAY = 0.997
LEARNING_RATE = 0.0001
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
TARGET_UPDATE = 10
BATCH_SIZE = 64
N_ACTIONS = 2
INPUT_DIMS = 5
FC_1_DIMS = 300
FC_2_DIMS = 600
FC_3_DIMS = FC_2_DIMS # If we don't want a third layer, set this to FC_2_DIMS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TAU = 0.001 # For soft update
MEMORY_SIZE = 400000

##### Environment parameters
COMFORT_PENALTY = 10 # Penalty applied when going outside of "comfort" bounds
BATTERY_DEPRECIATION = 1 # When using the battery some depreciation is created
T_MIN = 19.5 # Minimum temperature that should be achieved inside of the building
T_MAX = 22.5 # Maximum temperature that should be achieved inside of the building
C_I = 2.07*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
C_E = 3.24*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
R_IA = 5.29e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_IE = 0.909e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_EA = 4.47e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
A_w = 7.89 # Window surface area
NOMINAL_HEAT_PUMP_POWER = 2000 # 2kW based on some quick loockup of purchaseable heat pumps
STORAGE_CAPACITY = 4000 # Number of Watts that can be stored in the battery
C_MAX = 2750 * TIME_STEP_SIZE / 3600# Power in watt that the charging can provide divided by the time step size
D_MAX = 2750 * TIME_STEP_SIZE / 3600# Power in watt that the discharging can provide divided by the time step size
ETA = 0.95 # Charging and discharging efficiency of the battery
MIN_STORAGE = 0.1 * STORAGE_CAPACITY
INITIAL_STORAGE = 1000 # Set here what is initially stored inside of the battery
SELL_PRICE_DISCOUNT = 0.9 # Percentage of buying price, so selling price = SELL_PRICE_DISCOUNT*buying_price
PV_EFFICIENCY = 0.15
PV_SURFACE = 30