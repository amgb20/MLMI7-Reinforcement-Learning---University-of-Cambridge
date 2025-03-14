import numpy as np
from tqdm import tqdm
from model import Model, Actions
from typing import Callable

class ValueIteration:
    def __init__(self, model: Model, world_name: str, maxit: int = 100, tol: float = 1e-6):
        self.model = model
        self.world_name = world_name
        self.maxit = maxit
        self.tol = tol
        self.V = np.zeros(len(model.states))
        self.pi = np.zeros(len(model.states))
        self.iter_snap = [np.copy(self.V)]  # to store the value function at each iteration

    def update_value(self, s, V, a, reward: Callable):
        updated_value = np.sum([self.model.transition_probability(s, s_, a)
                                * (reward(s, a) + self.model.gamma * V[s_])
                                for s_ in self.model.states])
        return updated_value

    def sync_value_iteration(self):
        for i in tqdm(range(self.maxit)):
            V_new = np.zeros(len(self.model.states))
            for s in self.model.states:
                val = []
                for a in Actions:
                    R = self.model.reward(s, a)
                    val.append(self.update_value(s, self.V, a, lambda *_: R))
                V_new[s] = np.max(val)
            theta = np.max(np.abs(V_new - self.V))  # update the difference before
            self.iter_snap.append(np.copy(V_new))
            self.V = V_new.copy()
            # Stop when the value function stops changing
            if theta < self.tol:
                print(f"✅ Synchronous value iteration converged after {i+1} iterations for {self.world_name}.")
                break

        for s in self.model.states:
            action_index = np.argmax(
                [self.update_value(s, self.V, a, lambda *_: self.model.reward(s, a)) for a in Actions]
            )
            self.pi[s] = Actions(action_index)

        return self.V, self.pi, self.iter_snap

    def async_value_iteration(self):
        for i in tqdm(range(self.maxit)):
            theta = 1e-16
            for s in self.model.states:
                val = []
                for a in Actions:
                    R = self.model.reward(s, a)
                    val.append(self.update_value(s, self.V, a, lambda *_: R))
                V_new = np.max(val)
                theta = max(theta, np.abs(V_new - self.V[s]))
                self.V[s] = V_new
            self.iter_snap.append(np.copy(self.V))
            # Stop when the value function stops changing
            if theta < self.tol:
                print(f"✅ Asynchronous value iteration converged after {i+1} iterations for {self.world_name}.")
                break

        for s in self.model.states:
            action_index = np.argmax(
                [self.update_value(s, self.V, a, lambda *_: self.model.reward(s, a)) for a in Actions]
            )
            self.pi[s] = Actions(action_index)

        return self.V, self.pi, self.iter_snap
