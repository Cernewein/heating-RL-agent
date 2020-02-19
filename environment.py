import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
from vars import *
import random


class Building:
    """ This class represents the building that has to be controlled. Its dynamics are modelled based on an RC analogy.
    When instanciated, it initialises the inside temperature to 21°C, the envelope temperature to 20, and resets the done
    and time variables.
    """
    def __init__(self, dynamic=False):

        # If variable sun power, outside temperature, price should be used
        self.dynamic = dynamic
        ### Initiliazing the temperatures
        self.inside_temperature = 21.0 #np.random.randint(19,24)
        self.envelope_temperature = 20

        ### Selecting a random set for the outside temperatures based on a dataset
        self.random_day=random.randint(0,363)*24
        self.ambient_temperatures = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[self.random_day:self.random_day+NUM_HOURS+1,2]
        self.ambient_temperature=self.ambient_temperatures[self.random_day]

        ### Based on the same day, choose the sun irradiation for the episode

        self.sun_powers = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[self.random_day:self.random_day+NUM_HOURS+1,3]
        self.sun_power = self.sun_powers[self.random_day]


        self.done = False
        self.time=0

    def heat_pump_power(self, phi_e):
        """Takes an electrical power flow and converts it to a heat flow.

        :param phi_e: The electrical power
        :type phi_e: Float
        :return: Returns the heat flow as an integer
        """
        return phi_e*(0.0606*self.ambient_temperature+2.612)

    def step(self, action):
        """

        :param action: The chosen action - is the index of selected action from the action space.
        :type action: Integer
        :return: Returns the new state after a step, the reward for the action and the done state
        """
        #delta = 1/(R_IA*C_I) * (T_AMBIENT - self.inside_temperature) + 1/(R_IE*C_I)*(self.envelope_temperature - self.inside_temperature) + choice * self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER)/C_I

        #delta_envelope = 1/(R_IE*C_E) * (self.inside_temperature - self.envelope_temperature) + 1/(R_EA*C_E) * (T_AMBIENT - self.envelope_temperature)

        delta = 1 / (R_IA * C_I) * (self.ambient_temperature - self.inside_temperature) + \
                action * self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER)/C_I + A_w*self.sun_power
        #self.envelope_temperature += delta_envelope* TIME_STEP_SIZE
        self.inside_temperature += delta * TIME_STEP_SIZE
        # If we are out of bounds, fix that issue
        if self.inside_temperature > T_BOUND_MAX:
            self.inside_temperature = T_BOUND_MAX
        elif self.inside_temperature < T_BOUND_MIN:
            self.inside_temperature = T_BOUND_MIN

        r = self.reward(action)
        self.time +=1

        if self.dynamic:
            # Updating the outside temperature with the new temperature
            self.ambient_temperature = self.ambient_temperatures[self.random_day + (self.time * TIME_STEP_SIZE)//3600]
            self.sun_power = self.sun_powers[self.random_day + (self.time * TIME_STEP_SIZE)//3600]

        if self.time >= NUM_TIME_STEPS:
            self.done = True

        return [self.inside_temperature, self.ambient_temperature, self.sun_power,self.time], r, self.done

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

        :return: Returns the resetted inside temperature, ambient temperature and sun power
        """
        self.inside_temperature = 21
        self.ambient_temperature = self.ambient_temperatures[self.random_day]
        self.sun_power = self.sun_powers[self.random_day]

        self.done = False
        self.time = 0
        return [self.inside_temperature,self.ambient_temperature,self.sun_power,self.time]


class basicBuilding:
    """ This class represents the building that has to be controlled. Its dynamics are modelled based on an RC analogy.
    When instanciated, it initialises the inside temperature to 21°C, the envelope temperature to 20, and resets the done
    and time variables.
    """
    def __init__(self):

        ### Initiliazing the temperatures
        self.inside_temperature = 21.0 #np.random.randint(19,24)
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

        delta = 1 / (R_IA * C_I) * (T_AMBIENT - self.inside_temperature) + \
                action * self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER)/C_I
        #self.envelope_temperature += delta_envelope* TIME_STEP_SIZE
        self.inside_temperature += delta * TIME_STEP_SIZE
        # If we are out of bounds, fix that issue
        if self.inside_temperature > T_BOUND_MAX:
            self.inside_temperature = T_BOUND_MAX
        elif self.inside_temperature < T_BOUND_MIN:
            self.inside_temperature = T_BOUND_MIN

        self.time +=1

        if self.time >= NUM_TIME_STEPS:
            self.done = True

        return self.inside_temperature

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

        :return: Returns the resetted inside temperature, ambient temperature and sun power
        """
        self.inside_temperature = 21

        self.done = False
        self.time=0
        return self.inside_temperature