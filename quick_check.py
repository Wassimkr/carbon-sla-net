# quick_fix_monitor.py — run once to extract eval data into a plottable CSV
import numpy as np, pandas as pd, pathlib

npz = np.load('results/ppo/evaluations.npz')
print('Keys:', list(npz.keys()))
print('timesteps shape:', npz['timesteps'].shape)
print('results shape:', npz['results'].shape)

# evaluations.npz contains: timesteps, results (n_evals x n_episodes), ep_lengths
timesteps = npz['timesteps']
mean_rewards = npz['results'].mean(axis=1)

df = pd.DataFrame({'timestep': timesteps, 'mean_reward': mean_rewards})
df.to_csv('results/ppo/eval_rewards.csv', index=False)
print(df)
print('Saved results/ppo/eval_rewards.csv')