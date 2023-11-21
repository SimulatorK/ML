# -*- coding: utf-8 -*-

import gym
import pygame
import os
#from test_env import TestEnv
#from plots import Plots
from algorithms.rl import RL
import itertools

class GridSearch:
    @staticmethod
    def Q_learning_grid_search(env, epsilon_decay,
                                gammas, init_alphas,
                                min_alphas,
                                alpha_decay_ratios,
                                iters):
        qs = []
        vs = []
        pis = []
        q_tracks = []
        pi_tracks = []
        for epsilon_decay_ratio, gamma, init_alpha, min_alpha, alpha_decay_ratio, n_episodes in itertools.product(epsilon_decay,
                                                                                                                    gammas, init_alphas,
                                                                                                                    min_alphas,
                                                                                                                    alpha_decay_ratios,
                                                                                                                    iters):
            print(f"running -- with epsilon decay: {epsilon_decay}\tGamma: {gamma}\tinit_alpha: {init_alpha}\tmin_alpha: {min_alpha}")
            
            Q, V, pi, Q_track, pi_track = RL(env).q_learning(epsilon_decay_ratio=epsilon_decay_ratio,
                                                             gamma=gamma,
                                                             init_alpha=init_alpha,
                                                             min_alpha=min_alpha,
                                                             alpha_decay_ratio=alpha_decay_ratio,
                                                             n_episodes=n_episodes)
            qs.append(Q)
            vs.append(V)
            pis.append(pi)
            q_tracks.append(Q_track)
            pi_tracks.append(pi_tracks)
            
        return qs, vs, pis, q_tracks, pi_tracks

if __name__ == "__main__":
    frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)
    epsilon_decay = [.4, .7, .9]
    iters = [500, 5000, 50000]
    GridSearch.Q_learning_grid_search(frozen_lake.env, epsilon_decay, iters)
