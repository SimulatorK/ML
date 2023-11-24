## CS7641 Machine Learning - Assignment 3

from sklearn import preprocessing
from  sklearn import model_selection
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.metrics import silhouette_samples, silhouette_score
# Neural Network MLP
from sklearn.neural_network import MLPClassifier
from collections import Counter
import os
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

import itertools
import gym
#import gymnasium as gym
import pygame

from gymnasium.wrappers import TransformReward
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import random

from scipy.spatial.distance import cdist
from sklearn import mixture
import math
from matplotlib.patches import Ellipse
from PIL import Image
import glob
import shutil

verbose = True
###############################################################################
# Helper Functions
###############################################################################

def vPrint(text: str = '', verbose: bool = verbose):
    if verbose:
        print(text)

def printTime(start,end, verbose = verbose):
    
    vPrint('Elapsed Time = {0:.3f} sec'.format(end-start))

def make_gif(env, pi, savefilename = 'GIF'):
    savefile = os.path.join(os.getcwd(),savefilename) + '.gif'
    env.reset()
    terminated = truncated = False
    frame = 0
    images = []
    while not (terminated or truncated):
        frame_folder = f'Images/Frame_Folder'
        if not os.path.isdir(frame_folder):
            
            os.mkdir(frame_folder)

        next_state, reward, terminated, truncated, _ = env.step(pi(env.get_wrapper_attr('s')))
        
        img = Image.fromarray(env.render())
        
        images.append(img)
    
        
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(savefile, format="GIF", append_images=frames,
               save_all=True, duration=250, loop=1)
    for img in frames:
        img.close()

class RewardShaper(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


# Load data sets
seed = 903860493

if __name__ == '__main__':
    os.chdir(os.path.split(__file__)[0])
 
# Import bettermdptools
os.chdir('../bettermdptools')   
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.plots import Plots
from examples.grid_search import GridSearch
from examples.blackjack import Blackjack
from examples.test_env import TestEnv 

os.chdir('../Assignment4')

np.random.seed(seed)

n_iters = 10000

###############################################################################
f = gym.make('FrozenLake-v1',render_mode = 'rgb_array', desc=generate_random_map(size=20),is_slippery=True)
b = Blackjack(render_mode="rgb_array")
c = gym.make("MountainCar-v0",render_mode='rgb_array')

envs = [f,
        b,
        c,]

names = ['FrozenLake','Blackjack','MountainCar-v0']

## FrozenLake
env = envs[0]
name = names[0]

def reward_f(r):
    if r>0:
        return 2
    else:
        return r-0.01

#env = TransformReward(env, lambda r: reward_f(r))

###########################################################################
# Value & Policy Iteration
###########################################################################
# Value
vPrint('Value Iteration:')
start = time()
env.reset(seed=seed)
V, V_track, pi = Planner(env.P).value_iteration(gamma = 0.99)    
end = time()
printTime(start,end)
avg_r_v = [sum(V_track[i]).mean() for i in range(len(V_track))]
x_f_v = np.argmax(avg_r_v)
# Plots
Plots.grid_values_heat_map(V, "Value Iteration State Values")
max_value_per_iter_v = np.amax(V_track, axis=1)
Plots.v_iters_plot(max_value_per_iter_v, "Value Iteration Max State Values")
n_states = env.observation_space.n
new_pi_v = list(map(lambda x: pi[x], range(n_states)))
s = int(math.sqrt(n_states))
Plots.grid_world_policy_plot(np.array(new_pi_v), "Value Iteration Policy")

test_scores_v = TestEnv.test_env(env=env,
                                 seed=seed,
                                 render=False,
                                 pi=pi,
                                 user_input=False,
                                 n_iters = n_iters)
vPrint("Value Iterations Results:")
test_scores_v_c = Counter(test_scores_v)
wins = test_scores_v_c[1]
wins_p = round(wins / n_iters * 100,2)
losses = test_scores_v_c[0]
losses_p = round(losses / n_iters * 100,2)
vPrint(f'\tWins: {wins} ({wins_p}%); Losses: {losses} ({losses_p}%)\n')


# Policy
vPrint('Policy Iteration:')
start = time()
env.reset(seed=seed)
V, V_track, pi = Planner(env.P).policy_iteration(gamma = 0.99) 
end = time()
printTime(start,end)
avg_r_p = [sum(V_track[i]).mean() for i in range(len(V_track))]
x_f_p = np.argmax(avg_r_p)
Plots.grid_values_heat_map(V, "Policy Iteration State Values")

max_value_per_iter_p = np.amax(V_track, axis=1)
Plots.v_iters_plot(max_value_per_iter_p, "Policy Iteration Max State Values")

n_states = env.observation_space.n
new_pi_p = list(map(lambda x: pi[x], range(n_states)))
s = int(math.sqrt(n_states))
Plots.grid_world_policy_plot(np.array(new_pi_p), "Policy Iteration Policy")


test_scores_p = TestEnv.test_env(env=env,
                                 seed=seed,
                                 render=False,
                                 pi=pi,
                                 user_input=False,
                                 n_iters = n_iters)
vPrint("Value Iterations Results:")
test_scores_p_c = Counter(test_scores_p)
wins = test_scores_p_c[1]
wins_p = round(wins / n_iters * 100,2)
losses = test_scores_p_c[0]
losses_p = round(losses / n_iters * 100,2)
vPrint(f'\tWins: {wins} ({wins_p}%); Losses: {losses} ({losses_p}%)\n')

# Only plot up to where it converges
fig, ax = plt.subplots(figsize=(8,6),dpi = 200)
title_ = f'Policy_v_Value_Iteration_{name}'
ax.plot(avg_r_v[:x_f_v],label='Value Iteration')
ax.set_xlabel('Iteration')
ax.set_ylabel('Reward')
ax.plot(avg_r_p[:x_f_p],label='Policy Iteration')
plt.legend()
plt.title(title_)
fig.tight_layout()
ax.grid()
plt.savefig(f'Images/{title_}.png')
plt.show()


###############################################################################
# Q_Learning
###############################################################################

vPrint('Q-Learning:')

gamma = 1.0 #Discount factor
init_alpha = 1.0 #Learning rate
min_alpha = 0.1
alpha_decay_ratio = 0.4
init_epsilon = 1.0 #Initial epsilon value for epsilon greedy strategy
min_epsilon = 0.9
epsilon_decay_ratio=0.9999
n_episodes=1e6
# =============================================================================
# 
# epsilon_schedule = RL.decay_schedule(init_value = init_epsilon,
#                   min_value = min_epsilon, 
#                   decay_ratio = epsilon_decay_ratio, 
#                   max_steps = n_episodes, log_start=-2, log_base=10)
# 
# learning_schedule = RL.decay_schedule(init_value = init_alpha,
#                   min_value = min_alpha, 
#                   decay_ratio = alpha_decay_ratio, 
#                   max_steps = n_episodes, log_start=-2, log_base=10)
# =============================================================================

# =============================================================================
# fig, ax = plt.subplots(figsize=(8,6),dpi = 200)
# ax.plot(range(n_episodes),epsilon_schedule,label = 'Epsilon Schedule')
# ax.plot(range(n_episodes),learning_schedule,label = 'Learning Schedule')
# ax.set_xlabel('Episode')
# ax.set_ylabel('Value')
# ax.legend()
# plt.tight_layout()
# plt.savefig('Images/Decay_Schedules_Q-Learning.png')
# plt.show()
# =============================================================================


gammas = np.linspace(0.9,1.0,1) #Discount factor
init_alphas = np.linspace(1,1.0,1) #Learning rate
min_alphas = np.linspace(0.1,0.1,1)
alpha_decay_ratios = np.linspace(0.5,0.5,1)
init_epsilons = np.linspace(1.0,1.0,1) #Initial epsilon value for epsilon greedy strategy
min_epsilons = np.linspace(0.1,0.9,1)
epsilon_decay_ratios = np.linspace(0.99999999,0.99999999,1)
n_episodes_ = [5e6,]

tot = len(gammas) * len(init_alphas) * len(min_alphas) * len(alpha_decay_ratios) * len(init_epsilons) * len(min_epsilons) * len(epsilon_decay_ratios ) * len(n_episodes_)

rl_model = RL(env)

def custom_decay_schedule(r):
    
    def func(init_val,min_val,ratio,n_episodes):
        ratio = r
        return np.linspace(init_val,min_val,n_episodes)**ratio

    return func

rl_model.decay_schedule = custom_decay_schedule(0.5)


for g, gamma in enumerate(gammas):
    for ia, init_alpha in enumerate(init_alphas):
        for ma, min_alpha in enumerate(min_alphas):
            for adr, alpha_decay_ratio in enumerate(alpha_decay_ratios):
                for ie, init_epsilon in enumerate(init_epsilons):
                    for me, min_epsilon in enumerate(min_epsilons):
                        for edr, epsilon_decay_ratio in enumerate(epsilon_decay_ratios):
                            for n, n_episodes in enumerate(n_episodes_):
                                
                                env.reset(seed=seed)                               
                                print(f'{gamma}:{init_alpha}:{min_alpha}:{alpha_decay_ratio}:{init_epsilon}:{min_epsilon}:{epsilon_decay_ratio}:{n_episodes}')
                                
                                Q, V, pi, Q_track, pi_track = rl_model.q_learning(nS=env.observation_space.n,
                                                                                    nA=env.action_space.n,
                                                                                    gamma=gamma,
                                                                                    init_alpha=init_alpha,
                                                                                    min_alpha=min_alpha,
                                                                                    alpha_decay_ratio = alpha_decay_ratio,
                                                                                    init_epsilon=init_epsilon,
                                                                                    min_epsilon=min_epsilon,
                                                                                    epsilon_decay_ratio=epsilon_decay_ratio,
                                                                                    n_episodes=int(n_episodes))
                                
                                n_states = env.observation_space.n
                                new_pi = list(map(lambda x: pi[x], range(n_states)))
                                s = int(math.sqrt(n_states))
                                Plots.grid_world_policy_plot(np.array(new_pi), "Q-Learning Grid World Policy")
                                
                                max_q_value_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
                                Plots.v_iters_plot(max_q_value_per_iter, "Q-Learning Max Q-Values")
                                
                                Plots.grid_values_heat_map(V, "Q-Learning State Values")


epsilon_schedule = rl_model.decay_schedule(init_val = init_epsilon,
                  min_val = min_epsilon, 
                  ratio = epsilon_decay_ratio, 
                  n_episodes = int(n_episodes))

learning_schedule = rl_model.decay_schedule(init_val = init_alpha,
                  min_val = min_alpha, 
                  ratio = alpha_decay_ratio, 
                  n_episodes = int(n_episodes))

fig,ax = plt.subplots(figsize=(8,6),dpi=200)
ax.plot(range(int(n_episodes)),epsilon_schedule,label="Epsilon")
ax.plot(range(int(n_episodes)),learning_schedule,label="Alpha")
ax.set_xlabel('Episode')
ax.set_ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()


###############################################################################
# Blackjack
###############################################################################
n_iters = 10000
Q, V, pi, Q_track, pi_track = RL(b.env).q_learning(b.n_states, b.n_actions, b.convert_state_obs)
test_scores_q = TestEnv.test_env(env=b.env, seed=seed, render=False, pi=pi, user_input=False,
                               convert_state_obs=b.convert_state_obs, n_iters = n_iters)
vPrint("Q-Learning Results:")
test_scores_q_c = Counter(test_scores_q)
wins = test_scores_q_c[1]
wins_p = round(wins / n_iters * 100,2)
losses = test_scores_q_c[-1]
losses_p = round(losses / n_iters * 100,2)
ties = test_scores_q_c[0]
ties_p = round(ties / n_iters * 100,2)
vPrint(f'\tWins: {wins} ({wins_p}%); Losses: {losses} ({losses_p}%); Ties: {ties} ({ties_p}%)\n')

max_q_value_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
Plots.v_iters_plot(max_q_value_per_iter, "Max Q-Values")
    

V, V_track, pi = Planner(b.P).value_iteration()
max_value_per_iter = np.amax(V_track, axis=1)

test_scores_v = TestEnv.test_env(env=b.env, seed=seed, render=False, pi=pi, user_input=False,
                               convert_state_obs=b.convert_state_obs, n_iters = n_iters)

Plots.v_iters_plot(max_value_per_iter, "Value Iteration State Values")
vPrint("Value Iterations Results:")
test_scores_v_c = Counter(test_scores_v)
wins = test_scores_v_c[1]
wins_p = round(wins / n_iters * 100,2)
losses = test_scores_v_c[-1]
losses_p = round(losses / n_iters * 100,2)
ties = test_scores_v_c[0]
ties_p = round(ties / n_iters * 100,2)
vPrint(f'\tWins: {wins} ({wins_p}%); Losses: {losses} ({losses_p}%); Ties: {ties} ({ties_p}%)\n')


V, V_track, pi = Planner(b.P).policy_iteration()
max_value_per_iter = np.amax(V_track, axis=1)
Plots.v_iters_plot(max_value_per_iter, "Policy Iteration State Values")


test_scores_p = TestEnv.test_env(env=b.env, seed=seed, render=False, pi=pi, user_input=False,
                               convert_state_obs=b.convert_state_obs, n_iters = n_iters)
vPrint("Policy Iterations Results:")
test_scores_p_c = Counter(test_scores_p)
wins = test_scores_p_c[1]
wins_p = round(wins / n_iters * 100,2)
losses = test_scores_p_c[-1]
losses_p = round(losses / n_iters * 100,2)
ties = test_scores_p_c[0]
ties_p = round(ties / n_iters * 100,2)
vPrint(f'\tWins: {wins} ({wins_p}%); Losses: {losses} ({losses_p}%); Ties: {ties} ({ties_p}%)\n')


# =============================================================================
# 
# for v, V in enumerate(vs):
#     
#     Plots.grid_values_heat_map(V, f"Q_learning{v} State Values")
# 
# avg_r = [np.sum(q_tracks[2][i],axis=1).mean() for i in range(len(q_tracks[2]))]
# 
# fig, ax = plt.subplots(figsize=(8,6),dpi = 200)
# title_ = f'Q_learning_convergence_{name}'
# ax.plot(avg_r)
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Reward')
# plt.title(title_)
# fig.tight_layout()
# ax.grid()
# plt.savefig(f'Images/{title_}.png')
# plt.show()
# =============================================================================

# =============================================================================
# 
# epr = []
# eps = []
# ep_avg_r = []
# for ep in range(q_tracks[-1].shape[0]):
#     # For each episode, take the optimal actions
#     env.reset()
#     truncated = terminated = False
#     n_ep = 0
#     while not (truncated or terminated):
#         
#         action = np.argmax(q_tracks[-1][ep][env.get_wrapper_attr('s')])
#         
#         # Take action
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         
#         # Add the reward to this episode's list, increment n_ep
#         n_ep+=1
#     
#     epr.append(reward)
#     eps.append(n_ep)
#     ep_avg_r.append(reward / n_ep)
# =============================================================================

##############
# =============================================================================
# 
# # Mountain car problem
# # discretizze the state space
# c_env = c.env
# c_env.reset()
# 
# c_states = 200
# 
# min_p = c_env.observation_space.low[0]
# max_p = c_env.observation_space.high[0]
# min_v = c_env.observation_space.low[1]
# max_v = c_env.observation_space.high[1]
# 
# p_range = np.array(list(map(float,np.linspace(min_p,max_p,c_states))))
# v_range = np.array(list(map(float,np.linspace(min_v,max_v,c_states))))
# 
# state_space = []
# 
# for i_p, p in enumerate(p_range):
#     for i_v, v in enumerate(v_range):
#         state_space.append((p,v))
# 
# def get_p_v_state(p_range, v_range, pos,vel):
#     
#     x = np.argmin(abs(p_range - pos))
#     y = np.argmin(abs(v_range - vel))
#     
#     return p_range[x], v_range[y], x, y
#     
# def state_to_bucket(state):
#     # state is p, v
#     b_state = get_p_v_state(p_range = p_range, v_range = v_range, pos = state[0], vel = state[1])
#     
#     return b_state[-2], b_state[-1]
# 
# vels = []
# pos = []
# c_env.reset(seed=seed)
# for _ in range(1000):
#     
#     next_state, reward, terminated, truncated, _ = c_env.step(1)
#     pos.append(next_state[0])
#     vels.append(next_state[1])
#         
# =============================================================================

#############################################################################
#############################################################################
# =============================================================================
# 
# q_table = np.ndarray((c_states,c_states,c_env.action_space.n))
# 
# def select_action(state, explore_rate):
#     if random.random() < explore_rate:
#         action = c_env.action_space.sample()
#     else:
#         action = np.argmax(q_table[state[0]][state[1]])
#     return action
# 
# learning_rate = 0.5
# min_learning_rate = 0.05
# learning_decay = 0.5
# 
# explore_rate = 1
# min_explore_rate = 0.1
# decay_ratio = 0.5
# 
# discount_factor = 0.99
# num_streaks = 0
# n_steps = 10000
# 
# epsilon_schedule = RL.decay_schedule(init_value = explore_rate,
#                   min_value = min_explore_rate, 
#                   decay_ratio = decay_ratio, 
#                   max_steps = n_steps, log_start=-2, log_base=10)
# 
# learning_schedule = RL.decay_schedule(init_value = learning_rate,
#                   min_value = min_learning_rate, 
#                   decay_ratio = learning_decay, 
#                   max_steps = n_steps, log_start=-2, log_base=10)
#  
# 
# def get_explore_rate(episode):
#     
#     return epsilon_schedule[episode]
#     
# def get_learning_rate(episode):
#     
#     return learning_schedule[episode]
# 
# for episode in range(n_steps):
#     
#     observ, _ = c_env.reset()
#     
#     state_0 = state_to_bucket(observ)
#     
#     for t in range(250):
#         
#         #c_env.render()
#         
#         action = select_action(state_0, explore_rate)
#         
#         next_state, reward, terminated, truncated, _ = c_env.step(action)
#         
#         state = state_to_bucket(next_state)
#         
#         best_q = np.amax(q_table[state])
#         
#         q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor*(best_q) - q_table[state_0 + (action,)])
#         
#         
#         state_0 = state
#         
#         print("\nEpisode = %d" % episode)
#         print("t = %d" % t)
#         print("Action: %d" % action)
#         print("State: %s" %str(state))
#         print("Reward: %f" % reward)
#         print("Best Q: %f" % best_q)
#         print("Explore rate: %f" % explore_rate)
#         print("Learning rate: %f" % learning_rate)
#         print("Streaks: %d" %num_streaks)
#         
#         print("")
#         
#         if terminated or truncated:
#             print("Episode %d finished after %f time steps" % (episode, t))
#             
#             if (t >= 199):
#                 num_streaks += 1
#             else:
#                 num_streaks = 0
#             break
#         
#         if num_streaks > 120:
#             break
#         
#         explore_rate = get_explore_rate(episode)
#         learning_rate = get_learning_rate(episode)
# =============================================================================


