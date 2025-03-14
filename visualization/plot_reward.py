import matplotlib.pyplot as plt
import numpy as np
import os
from .save_figure import save_fig
from scipy.ndimage.filters import uniform_filter1d


def plot_reward(rewards: np.array, world_name: str, question: str, task: str, show: bool = True, save: bool = True):
    plt.figure()
    x = np.arange(1, len(rewards)+1)
    y = rewards
    plt.plot(x, y)
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title(f"Episode Rewards for {world_name}")
    if save:
        save_fig(plt, question, world_name, task)
    if show:
        plt.show(show)
        
    return x, y

def plot_reward_comparison(rewards: np.array, rewards_2: np.array, world_name: str, question: str, task: str, show: bool = True, save: bool = True):
    plt.figure()
    x = np.arange(1, len(rewards)+1)
    y = rewards
    x2 = np.arange(1, len(rewards_2)+1)
    y2 = rewards_2
    plt.plot(x, y)
    plt.plot(x2, y2)
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title(f"Episode Rewards for {world_name}")
    if save:
        save_fig(plt, question, world_name, task)
    if show:
        plt.show(show)
        
    return x, y, x2, y2

def plot_accumulated_reward(sarsa_rewards, q_learning_rewards, episodes,question, world_name, task, save=True):
    plt.figure(figsize=(12, 6))

    
    plt.plot(np.arange(1, episodes + 1), uniform_filter1d(sarsa_rewards, size=50), label=f"SARSA ", linestyle="--")
    plt.plot(np.arange(1, episodes + 1), uniform_filter1d(q_learning_rewards, size=50), label=f"Q-Learning ")

    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward")
    plt.title("Comparison of Accumulated Reward for SARSA and Q-Learning")
    plt.legend()
    plt.grid()
    if save:
        save_fig(plt, question, world_name, task)
    plt.show()
    