#### General settings
TIME_DIVISION = 1# How many timesteps do we have in 1s? For example if we want every minute, set this to 60
NUM_TIME_STEPS = 24*3600 # A total of 24 hours computed every minute

##### RL Agent parameters
NUM_EPISODES = 500 # Number of episodes
EPSILON = 0.1 # For epsilon-greedy approach
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPS_DECAY = 0.9998

##### Environment parameters
E_PRICE = 10 # Price per kwh (expressed in price/kwminute)
COMFORT_PENALTY = 10 # Penalty applied when going outside of "comfort" bounds
T_MIN = 19.5 # Minimum temperature that should be achieved inside of the building
T_MAX = 22.5 # Maximum temperature that should be achieved inside of the building
C_I = 2.07*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
C_E = 3.24*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
R_IA = 5.39e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_IE = 0.909e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_EA = 4.47e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
T_BOUND_MIN = 15 # We will not model anything under 15 in order to limit the number of spaces
T_BOUND_MAX = 30 # We will not model anything over 30 in order to limit the number of spaces
T_AMBIENT = 2 # Ambient temperature
NOMINAL_HEAT_PUMP_POWER = 5000 # 2kW based on some quick loockup of purchaseable heat pumps
TEMPERATURE_ROUNDING = 10000 #We are rounding up to 1/TEMPERATURE_ROUNDING
PRECISION = (T_BOUND_MAX-T_BOUND_MIN)*TEMPERATURE_ROUNDING

