# heating-RL-agent


This repo implements a deep reinforcement learning based home energy management system. It features both a Deep Q-Learning algorithm as well as a Deep Deterministic Policy Gradient solution.
It has been developped for my master thesis at the Technical University of Denmark.
Built with ![](https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg.png)

A paper based on the work done during the master thesis has been published https://doi.org/10.1145/3427773.3427862


# Installation

In order to install the needed packages run:
```
pip install -r requirement.txt
```



# Data
The dataset contains three distinct parts (chosen year is 2014 but can of course be any):
* The historic electricity spot prices (region DK1 and DK2) for the year 2014 have been obtained on [NordPool](https://www.nordpoolgroup.com)
* The historic electricity loads (region DK1 and DK2) for the year 2014 have been obtained on [NordPool](https://www.nordpoolgroup.com)
* The historic weather conditions (temperature and sun) for Copenhagen in the year 2014 have been obtained on [RenawablesNinja](https://www.renewables.ninja/)

# Results

The benchmark for the RL agent is an optimal LP solution.
The RL agent in situation on the month of january 2014 which has not been used for training (the agent is controlling the heating source and the battery) :

![](/images/DDPG_storage_eval.png)

The temperature comfort bound are in red, the inside temperature evolution is shown next to the spot prices, the battery energy level, the outside temperature and the sun radiation.

Zooming in, especially for the battery level, and comparing it to what the linear programming solution does :

![](/images/DDPG_storage_power_zoom_profile.png)

In terms of cost and power consumption, the RL agent performs quite well when compared to the LP solution :

![](/images/comparing_ddpg_vs_lp.png)