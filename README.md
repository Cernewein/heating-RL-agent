# heating-RL-agent


This repo implements a deep reinforcement learning based home energy management system. It features both a [Deep Q-Learning](https://arxiv.org/abs/1312.5602) algorithm as well as a [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) solution. Built with Python & Pytorch.


It has been developped for my master thesis at the Technical University of Denmark.
A paper based on the work done during the master thesis has been published [here](https://doi.org/10.1145/3427773.3427862).


# Installation

In order to install the needed packages run:
```
pip install -r requirement.txt
```


# Environment

The environement (the house) is made up of following elements :

* Thermal dynamics modelled by an analogy with electrical RC systems (see figure)
* A heat pump which serves as heating element for the house and can be controlled by varying its input power
* A battery system which can be used for storing electrical energy
* A photovoltaic installation on the house generating electrical power

This house is subject to following elements :

* A varying outside temperature
* A varying base electrical load which cannot be controlled
* Varying sun radiation
* Varying electricity prices

![](/images/Ti.PNG)

The goal is to control the inside temperature while remaining within comfort bounds and minimizing heating costs.

# RL formulation

The agent can make use of following elements :
* The heat pump (continuous action between 0 and 1 translated into an electrical power between 0 W and P_max W at a certain time step, where P_max is the nominal heat pump power)
* The battery (one continuous variable modelling the charging or discharging power at a certain time step)

The agent needs to control the inside temperature of the house under following contraints :

* Keep within the temperature comfort bounds defined for the house
* Minimize the total cost of electricity
* Don't overuse the battery

Thus, the reward is modelled at each time step as the sum of :

* Negative price paid for the electricity used
* Negative reward proportional to the severity of the temperature bound trespassing
* Negative reward for battery activations

![](/images/RLFormulation.PNG)

The variables are the following :
* $T^a$ is the exterior temperature
* $T^i$ is the inside temperature
* $\Phi^s$ is the sun radiation
* soe is the state of energy of the battery
* $p$ is the electricity price
* TOD is the time of day
* $\phi$ is the electricity power
* $\lambda_P$ is a sensitivity variable for the power cost
* $\lambda_T$ is a sensitivity variable for the comfort disutility cost
* $\Psi$ is a sensitivity variable for the battery depreciation
* $\Delta_t$ is the time step size
* $\nu$ is the temperature trespassing of the upper or lower comfort bounds

# Data

The dataset used for modelling the environment contains three distinct parts (chosen year is 2014 but can of course be any):
* The historic electricity spot prices (region DK1 and DK2) for the year 2014 have been obtained on [NordPool](https://www.nordpoolgroup.com)
* The historic electricity loads (region DK1 and DK2) for the year 2014 have been obtained on [NordPool](https://www.nordpoolgroup.com)
* The historic weather conditions (temperature and sun) for Copenhagen in the year 2014 have been obtained on [RenawablesNinja](https://www.renewables.ninja/)


# Results

The benchmark for the RL agent is an optimal linear programming (LP) solution.
The figure below shows the RL agent managing the heating system and the battery over the course of january 2014 (which has not been used for training):

![](/images/DDPG_storage_eval.png)

The temperature comfort bounds are in red, the inside temperature evolution is shown next to the spot prices, the battery energy level, the outside temperature and the sun radiation. The agent manages to keep within the temperature bounds and makes use of the battery.

Zooming in, especially for the battery level, and comparing it to what the linear programming solution does shows that the solution adopted by the RL agent seems quite close to what the LP solution is :

![](/images/DDPG_storage_power_zoom_profile.png)

In terms of cost and power consumption, the RL agent performs quite well when compared to the LP solution ("sun" qualifies an alternative scenario with increased sun radiation):

![](/images/comparing_ddpg_vs_lp.png)