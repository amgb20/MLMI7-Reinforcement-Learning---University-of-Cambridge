# import utils
from model import *
from visualization.plot_vp import *
from visualization.plot_convergence import *
from visualization.plot_reward import *
from policy_iteration import *
from value_iteration import *
from world_config import *
from sarsa import *

# import python libraries
import numpy as np
import os
import matplotlib.pyplot as plt

# Import required libraries
import wandb

# Constants
TASK_B2_BO = "SARSA_implementation_BO_Sweep"
QUESTION_B2 = "QUESTION B2"
show = False

# World model selection
world_model = [small_world]

# Define SARSA Bayesian Optimization Training Function
def train_BO_sweep_Sarsa():
    for i, model in enumerate(world_model):
        model = Model(model)
        
        # Initialize W&B Run
        wandb.init(
            project="MLMI7-RL",
            name=f"{model.world.world_name}_{QUESTION_B2}_{TASK_B2_BO}",
            notes="This run is used for BO for Epsilon and Alpha Hyperparameter sweep"
        )

        # Get hyperparameters from W&B
        alpha = wandb.config.alpha
        epsilon = wandb.config.epsilon

        # Train SARSA
        sarsa = Sarsa(model, model.world.world_name, maxit=100, episode=500, alpha=alpha, epsilon=epsilon)
        Q, iter_history_sarsa, V, pi, rewards, iter_episdode = sarsa.on_policy_sarsa()

        # Generate Plots and Extract Data
        reward_x, reward_y = plot_reward(rewards, model.world.world_name, QUESTION_B2, TASK_B2_BO, show=show, save=False)
        plot_vp(model, V, pi, model.world.world_name, QUESTION_B2, TASK_B2_BO, show=show, save=False)
        conv_x, conv_y = plot_sarsa_convergence(iter_episdode, model.world.world_name, QUESTION_B2, TASK_B2_BO, save=False)

        # Log Key Metrics to W&B
        wandb.log({
            "Q_max": np.max(Q),
            "reward_sum": np.sum(rewards),
            "reward_curve": wandb.Table(data=[[x, y] for x, y in zip(reward_x, reward_y)], columns=["Episode", "Reward"]),
            "convergence_curve": wandb.Table(data=[[x, y] for x, y in zip(conv_x, conv_y)], columns=["Iteration", "Episode"]),
        })

        wandb.finish()  # Finish W&B run 
        
wandb.agent(
            "ab3149-university-of-cambridge/MLMI7-RL/nllbsv7g", 
            function=train_BO_sweep_Sarsa
        )