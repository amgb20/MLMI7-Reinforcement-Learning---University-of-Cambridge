import numpy as np
from tqdm import tqdm
from model import Model, Actions
from typing import Callable
from tqdm import tqdm

class QL:
    def __init__(self, model: Model, world_name: str, maxit: int = 100, episode: int = 1000, alpha: float = 0.1, epsilon: float = 0.1):
        self.model = model
        self.world_name = world_name
        self.episode = episode
        self.alpha = alpha
        self.epsilon = epsilon
        self.maxit = maxit
        self.Q_learning = np.zeros((len(model.states), len(Actions)))
        self.iter_snap = []  # snapshots of Q after each episode
        self.rewards = np.zeros((episode,))
        self.iter_episode = np.zeros((episode,))
        
    def update_q(self, s, a, r, s_next):
        current_q = self.Q_learning[s, a]

        # If next state is terminal, no future Q
        if s_next == self.model.goal_state:
            target_q = r
        else:
            target_q = r + self.model.gamma * np.max(self.Q_learning[s_next, :])

        return current_q + self.alpha * (target_q - current_q)
    
    def epsilon_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q_learning.shape[1])
        else:
            return np.argmax(self.Q_learning[s, :])
        
    def sample_next_state(self, s, a):
        # we us the model's transition probabilities to sample the next state
        probs = [self.model.transition_probability(s, s_, a) for s_ in self.model.states]
        s_ = np.random.choice(self.model.states, p=probs)
        return s_
        
    def q_learning(self):
        for ep in tqdm(range(self.episode)):
            s = self.model.start_state
            total_reward = 0
            iter_count = 0
            for t in range(self.maxit):
                iter_count += 1
                a = self.epsilon_greedy(s)
                s_ = self.sample_next_state(s,a)
                r = self.model.reward(s, a)
                total_reward += r
                
                self.Q_learning[s, a] = self.update_q(s, a, r, s_)
                if s_ == self.model.goal_state:
                    break
                
                s = s_
            self.rewards[ep] = total_reward
            self.iter_snap.append(np.copy(self.Q_learning))
            self.iter_episode[ep] = iter_count
        pi = np.argmax(self.Q_learning, axis=1)
        V = np.max(self.Q_learning, axis=1)
        return self.Q_learning, self.iter_snap, V, pi, self.rewards, self.iter_episode