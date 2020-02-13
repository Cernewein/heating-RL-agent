import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle as pkl
import time
from vars import *
from environment import Building
from agent import Agent
import progress

agent = Agent()
agent.train(NUM_EPISODES,NUM_TIME_STEPS)

plt.figure()
plt.plot(agent.episode_rewards)
plt.savefig(os.getcwd() + '/data/output/' + 'rewards_LR_'+ str(LEARNING_RATE) + '_G_' + str(DISCOUNT) + '_EPS_' + str(EPSILON) + '_TS_' + str(TIME_STEP_SIZE) +'.png')

plt.figure()
plt.plot(agent.temperature_evolutions[0])
plt.savefig(os.getcwd() + '/data/output/' + 'temperatures_initial_LR_' + str(LEARNING_RATE) + '_G_' + str(DISCOUNT) + '_EPS_' + str(EPSILON) + '_TS_' + str(TIME_STEP_SIZE) +'.png')

plt.figure()
plt.plot(agent.temperature_evolutions[-1])
plt.savefig(os.getcwd() + '/data/output/' + 'temperatures_trained_LR_' + str(LEARNING_RATE) + '_G_' + str(DISCOUNT) + '_EPS_' + str(EPSILON) + '_TS_' + str(TIME_STEP_SIZE) +'.png')

for t in range (T_BOUND_MIN,T_BOUND_MAX):
    print("\n Temperature {}".format(t))
    print(agent.q_table[t/1.])
