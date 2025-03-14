from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions
import matplotlib.pyplot as plt
from world_config import cliff_world, small_world, grid_world
from visualization.plot_vp import plot_vp


"""
This is a Deterministic Greedy Policy derived from Model-Based Dynamic Programming. Specifically:

- Model-Based: The algorithm requires access to the environment's dynamics (transition probabilities and reward function).
- Deterministic: For each state, the policy chooses the action that maximizes expected returns without stochasticity.
- Greedy Policy Improvement: Policy improvement step uses argmax to select the best action.
- Value-Based: The policy is derived from the state-value function computed during evaluation.
"""
def policy_iteration(model: Model, world_name: str, maxit: int = 100):

    V = np.zeros((model.num_states,)) # initialize value function V to zeros for all states
    pi = np.zeros((model.num_states,)) # initialize policy pi to zeros for all states
    iter_snap = [np.copy(pi)] # to store the policy at each iteration
    
    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )
        
    """
    for each state s, compute the value of following the current policy 
    pi by summing expected rewards and future discounted values based on the chosen action a.
    """
    def policy_evaluation():
        for s in model.states:
            R = model.reward(s, pi[s])
            V[s] = compute_value(s, pi[s], lambda *_: R)
    
    """
    For each state, choose the action that maximizes the expected value of taking that action and update the policy accordingly
    """
    def policy_improvement():
        for s in model.states:
            action_index = np.argmax(
                [compute_value(s, a, model.reward) for a in Actions]
            )
            pi[s] = Actions(action_index)

    for i in tqdm(range(maxit)):
        for _ in range(5): # this is actually doing an approximate policy iteration evaluation
            policy_evaluation()
        pi_old = np.copy(pi)
        policy_improvement()
        iter_snap.append(np.copy(pi))
        # After policy improvement, compare the old and new policies. If they remain unchanged, the algorithm terminates
        if all(pi_old == pi): 
            print(f"✅ breaking for {world_name}")
            break
    
    # return the value function and policy
    return V, pi, iter_snap


if __name__ == "__main__":
    model = Model(cliff_world)
    V, pi,_ = policy_iteration(model, world_name=model.world.world_name)
    plot_vp(model, V, pi, world_name=model.world.world_name, QUESTION="QUESTION A1", task="PI comparison", show=True, save=True)
    plt.show()
