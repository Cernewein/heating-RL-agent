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
    def __init__(self, dynamic=False, eval=False, pv_gen=True):

        # If variable sun power, outside temperature, price should be used
        self.eval = eval
        self.dynamic = dynamic
        self.pv_gen= pv_gen

        ### Initiliazing the temperatures
        self.inside_temperature = 21.0 #np.random.randint(19,24)

        ### Selecting a random set for the outside temperatures based on a dataset

        # If we are in eval mode, select the month of january
        if self.eval:
            self.random_day = 0 # First day of the year
        else:
            # Else select November/December for training
            self.random_day=random.randint(304,365-NUM_HOURS//24-1)*24
        self.ambient_temperatures = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[self.random_day:self.random_day+NUM_HOURS+1,2]
        self.ambient_temperature=self.ambient_temperatures[self.random_day]

        ### Based on the same day, choose the sun irradiation for the episode

        self.sun_powers = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[self.random_day:self.random_day+NUM_HOURS+1,3] * 3
        self.sun_power = self.sun_powers[self.random_day]

        ### Based on the same day, choose the hourly prices for the episode

        self.prices = pd.read_csv('data/environment/2014_DK2_spot_prices.csv',
                                  header = 0).iloc[self.random_day:self.random_day+NUM_HOURS+1,1]
        self.price = self.prices[self.random_day]

        ### Defining the storage capacity
        self.storage = INITIAL_STORAGE

        ## How much power from the grid has been drawn?
        self.power_from_grid = 0
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

        delta = 1 / (R_IA * C_I) * (self.ambient_temperature - self.inside_temperature) + \
                self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER*action[0])/C_I + A_w*self.sun_power/C_I

        self.inside_temperature += delta * TIME_STEP_SIZE


        # Battery charging or discharging cannot be greater than the defined limit
        # It also cannot be greater than current charging state for the discharge
        # And current empty capacity for the charging
        if action[1] >= 0:
            battery_power = np.minimum(action[1] * C_MAX, (STORAGE_CAPACITY - self.storage) / ETA)
            self.storage += int(battery_power * ETA)
        else:
            battery_power = np.maximum(action[1] * D_MAX, (MIN_STORAGE - self.storage) * ETA)
            self.storage += int(battery_power / ETA)


        # After having updated storage, battery power is scaled to MW for price computation
        battery_power = battery_power / (1e6)
        # Heat pump power is adjusted so that the power is expressed in MW and also adjusted to the correct time slot size
        heat_pump_power = action[0] * NOMINAL_HEAT_PUMP_POWER / (1e6) * TIME_STEP_SIZE / 3600

        # Power drawn from grid is the sum of what is put into battery or drawn from battery and how much we are heating
        # It is also minus what has been produced by PV if PV mode is activated

        if self.pv_gen:
            pv_generation = self.sun_power * PV_EFFICIENCY * PV_SURFACE * TIME_STEP_SIZE/3600 / (1e6)
        else:
            pv_generation = 0
        self.power_from_grid = np.maximum(0,heat_pump_power + battery_power - pv_generation)

        #self.power_from_grid = np.maximum(0, heat_pump_power + battery_power)

        r = self.reward(self.power_from_grid, battery_power)
        self.time +=1

        if self.dynamic:
            # Updating the outside temperature with the new temperature
            self.ambient_temperature = self.ambient_temperatures[self.random_day + (self.time * TIME_STEP_SIZE)//3600]
            self.sun_power = self.sun_powers[self.random_day + (self.time * TIME_STEP_SIZE)//3600]
            self.price = self.prices[self.random_day + (self.time * TIME_STEP_SIZE) // 3600]

        if self.time >= NUM_TIME_STEPS:
            self.done = True

        return [self.inside_temperature, self.ambient_temperature, self.sun_power, self.price , self.storage,self.time % int(24*3600//TIME_STEP_SIZE) ], r, self.done #


    def reward(self,power_from_grid, battery_power):
        """
        Returns the received value for the chosen action and transition to next state

        :param action: The selected action
        :return: Returns the reward for that action
        """

        penalty = np.maximum(0,self.inside_temperature-T_MAX) + np.maximum(0,T_MIN-self.inside_temperature)
        penalty *= COMFORT_PENALTY

        battery_usage_penalty = np.abs(BATTERY_DEPRECIATION*battery_power) # Negative reward of BATTERY_DEPRECIATION per MWh

        if power_from_grid >=0:
            paid_price = - power_from_grid*self.price
        else:
            paid_price = - power_from_grid*self.price * SELL_PRICE_DISCOUNT


        reward =  paid_price - penalty - battery_usage_penalty

        return reward

    def reset(self):
        """
        This method is resetting the attributes of the building.

        :return: Returns the resetted inside temperature, ambient temperature and sun power
        """
        self.inside_temperature = 21

        if self.eval:
            self.random_day=0
        else:
            self.random_day = random.randint(304, 365 - NUM_HOURS // 24 - 1) * 24

        self.ambient_temperatures = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                                header=3).iloc[self.random_day:self.random_day + NUM_HOURS + 1, 2]


        self.ambient_temperature = self.ambient_temperatures[self.random_day]

        ## Resetting the sun power
        self.sun_powers = pd.read_csv('data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
                                      header=3).iloc[self.random_day:self.random_day + NUM_HOURS + 1, 3] * 3
        self.sun_power = self.sun_powers[self.random_day]

        ## Resetting the prices
        self.prices = pd.read_csv('data/environment/2014_DK2_spot_prices.csv',
                                  header=0).iloc[self.random_day:self.random_day + NUM_HOURS + 1, 1]
        self.price = self.prices[self.random_day]

        self.storage = 0

        self.done = False
        self.time = 0

        return [self.inside_temperature,self.ambient_temperature,self.sun_power,self.price, self.storage,self.time] #,self.time % int(24*3600//TIME_STEP_SIZE)

