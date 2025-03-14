import matplotlib.pyplot as plt
import numpy as np
import os
from .save_figure import save_fig

def plot_conv(iter_history: np.array, i: int, world_name: str, question: str, task: str, save: bool = True):
    plt.figure()
    if task == "Convergence PI":
        plt.plot(np.arange(1, len(iter_history)), [np.sum(iter_history[i] != iter_history[i-1]) for i in range(1, len(iter_history))])
        plt.ylabel("Number of policy changes")
    elif task == "Convergence VI":
        plt.plot(np.arange(1, len(iter_history)), [np.max(np.abs(iter_history[i] - iter_history[i-1])) for i in range(1, len(iter_history))])
        plt.ylabel("Manathan Distance of Value Function")
    
    plt.xlabel("Iteration")
    plt.title(f"{task} for {world_name}")
    if save:
        save_fig(plt, question, world_name, task)
    plt.show()
    

def plot_comparative_conv(iter_history_sync, iter_history_async, i, world_name: str, question: str, task: str, save: bool = True):
    plt.figure(figsize=(8, 6))
    
    sync_distances = [np.sum(np.abs(iter_history_sync[i] - iter_history_sync[i-1])) for i in range(1, len(iter_history_sync))]
    async_distances = [np.sum(np.abs(iter_history_async[i] - iter_history_async[i-1])) for i in range(1, len(iter_history_async))]

    plt.plot(np.arange(1, len(iter_history_sync)), sync_distances, label="Synchronous VI", linestyle="--", marker="o", color="r")
    plt.plot(np.arange(1, len(iter_history_async)), async_distances, label="Asynchronous VI", linestyle="-", marker="s", color="b")
    
    plt.xlabel("Iteration")
    plt.ylabel("Manhattan Distance of Value Function")
    plt.ylim([0,100])
    plt.title(f"Comparison of Sync vs Async VI for {world_name}")
    plt.legend()
    
    if save:
        save_fig(plt, question, world_name, task)
    plt.show() 
    
def plot_sarsa_convergence(iter_snap, world_name: str, question: str, task: str, save: bool = True):
    num_episodes = len(iter_snap)
    max_q_values = [np.max(Q) for Q in iter_snap]
    x = np.cumsum(iter_snap)
    y = np.arange(1, num_episodes+1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Episodes")
    plt.title("SARSA Convergence Over Episodes")
    plt.grid()
    if save:
        save_fig(plt, question, world_name, task)
    plt.show()
    
    return x, y

def plot_sarsa_comparison_convergence(iter_snap_sarsa, iter_snap_exp_sarsa, world_name: str, question: str, task: str, save: bool = True):

    num_episodes_sarsa = len(iter_snap_sarsa)
    num_episodes_exp_sarsa = len(iter_snap_exp_sarsa)

    x_sarsa = np.cumsum(iter_snap_sarsa)
    x_exp_sarsa = np.cumsum(iter_snap_exp_sarsa)

    y_sarsa = np.arange(1, num_episodes_sarsa + 1)
    y_exp_sarsa = np.arange(1, num_episodes_exp_sarsa + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x_sarsa, y_sarsa, label="SARSA", linestyle="dashed", color="blue")
    plt.plot(x_exp_sarsa, y_exp_sarsa, label="Q_Learning", linestyle="solid", color="red")

    plt.xlabel("Cumulative Iterations")
    plt.ylabel("Episodes")
    plt.title("SARSA vs Q_Learning Convergence")
    plt.legend()
    plt.grid()

    if save:
        save_fig(plt, question, world_name, task)

    plt.show()

    return x_sarsa, y_sarsa, x_exp_sarsa, y_exp_sarsa