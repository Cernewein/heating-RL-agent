import numpy as np
#from PIL import Image
#import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
from vars import *


class Building:
    """ This class represents the building that has to be controlled. Its dynamics are modelled based on an RC analogy.
    When instanciated, it initialises the inside temperature to 21°C, the envelope temperature to 20, and resets the done
    and time variables.
    """
    def __init__(self):
        self.inside_temperature = 21.0 #np.random.randint(19,24)
        self.envelope_temperature = 20
        self.done = False
        self.time=0

    def heat_pump_power(self, phi_e):
        """Takes an electrical power flow and converts it to a heat flow.

        :param phi_e: The electrical power
        :type phi_e: Float
        :return: Returns the heat flow as an integer
        """
        return phi_e*(0.0606*T_AMBIENT+2.612)

    def step(self, action):
        """

        :param action: The chosen action - is the index of selected action from the action space.
        :type action: Integer
        :return: Returns the new state after a step, the reward for the action and the done state
        """
        #delta = 1/(R_IA*C_I) * (T_AMBIENT - self.inside_temperature) + 1/(R_IE*C_I)*(self.envelope_temperature - self.inside_temperature) + choice * self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER)/C_I

        #delta_envelope = 1/(R_IE*C_E) * (self.inside_temperature - self.envelope_temperature) + 1/(R_EA*C_E) * (T_AMBIENT - self.envelope_temperature)

        delta = 1 / (R_IA * C_I) * (T_AMBIENT - self.inside_temperature) + action * self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER)/C_I
        #self.envelope_temperature += delta_envelope* TIME_STEP_SIZE
        self.inside_temperature += delta * TIME_STEP_SIZE
        # If we are out of bounds, fix that issue
        if self.inside_temperature > T_BOUND_MAX:
            self.inside_temperature = T_BOUND_MAX
        elif self.inside_temperature < T_BOUND_MIN:
            self.inside_temperature = T_BOUND_MIN


        r = self.reward(action)

        self.time +=1

        if self.time >= NUM_TIME_STEPS:
            self.done = True

        return [self.inside_temperature], r, self.done

    def reward(self,action):
        """
        Returns the received value for the chosen action and transition to next state

        :param action: The selected action
        :return: Returns the reward for that action
        """
        penalty = 0
        if self.inside_temperature > T_MAX:
            penalty += COMFORT_PENALTY
        elif self.inside_temperature < T_MIN:
            penalty += COMFORT_PENALTY
        reward = -action*E_PRICE - penalty
        return reward

    def reset(self):
        """
        This method is resetting the attributes of the building.

        :return: Returns the resetted inside temperature
        """
        self.inside_temperature = 21
        self.done = False
        self.time=0
        return [self.inside_temperature]