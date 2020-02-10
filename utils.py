import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

rewards = pkl.load(open('rewards.pkl', "rb"))

rewards_df = pd.DataFrame()
rewards_df['reward'] = rewards
rewards_df['hour'] = pd.to_datetime(rewards_df.index, unit='s')

