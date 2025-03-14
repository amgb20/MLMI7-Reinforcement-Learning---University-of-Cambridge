import numpy as np
from tqdm import tqdm
from model import Model, Actions
from typing import Callable

class Sarsa:
    def __init__(self, model: Model, world_name: str, maxit: int = 100, episode: int = 1000, alpha: float = 0.1, epsilon: float = 0.1):
        self.model = model
        self.world_name = world_name
        self.episode = episode
        self.alpha = alpha
        self.epsilon = epsilon
        self.maxit = maxit
        self.Q = np.zeros((len(model.states), len(Actions)))
        self.iter_snap = []  # snapshots of Q after each episode
        self.rewards = np.zeros((episode,))
        self.iter_episode = np.zeros((episode,))
        
    def update_q(self, s, a, r, s_, a_):
        current_q = self.Q[s, a]
        # if s_ is terminal, then we consider future reward to be zero
        target_q = r + (self.model.gamma * self.Q[s_, a_] if s_ != self.model.goal_state else r)
        return current_q + self.alpha * (target_q - current_q)

    def epsilon_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        else:
            return np.argmax(self.Q[s, :])
    
    def sample_next_state(self, s, a):
        # we us the model's transition probabilities to sample the next state
        probs = [self.model.transition_probability(s, s_, a) for s_ in self.model.states]
        s_ = np.random.choice(self.model.states, p=probs)
        return s_
    
    def on_policy_sarsa(self):
        for ep in tqdm(range(self.episode)):
            s = self.model.start_state
            a = self.epsilon_greedy(s)
            total_reward = 0
            iter_count = 0
            for t in range(self.maxit):
                iter_count += 1
                # sample next state 
                s_ = self.sample_next_state(s, a)
                # get reward after taking action a and reaching s_
                r = self.model.reward(s, a)  # or self.model.reward(s_, a) depending on your model
                total_reward += r
                # if s_ is terminal, update Q and break
                if s_ == self.model.goal_state:
                    self.Q[s, a] = self.update_q(s, a, r, s_, None)
                    break
                # choose next action a_ epsilon-greedily
                a_ = self.epsilon_greedy(s_)
                # update Q(s, a)
                self.Q[s, a] = self.update_q(s, a, r, s_, a_)
                s, a = s_, a_
            self.rewards[ep] = total_reward
            self.iter_snap.append(np.copy(self.Q))
            self.iter_episode[ep] = iter_count
        pi = np.argmax(self.Q, axis=1)
        V = np.max(self.Q, axis=1)
        return self.Q, self.iter_snap, V, pi, self.rewards, self.iter_episode
    
    

    
class Expected_SARSA:
    def __init__(self, model: Model, world_name: str, maxit: int = 100, episode: int = 1000, alpha: float = 0.1, epsilon: float = 0.1):
        self.model = model
        self.world_name = world_name
        self.episode = episode
        self.alpha = alpha
        self.epsilon = epsilon
        self.maxit = maxit
        self.Q_exp = np.zeros((len(model.states), len(Actions)))
        self.iter_snap = []  # snapshots of Q after each episode
        self.rewards = np.zeros((episode,))
        self.iter_episode = np.zeros((episode,))
        
    def expected_q(self, s_):
        best_action = np.argmax(self.Q_exp[s_, :])
        num_actions = self.Q_exp.shape[1]
        
        expected_value = 0
        for a_p in range(num_actions):
            if a_p == best_action:
                prob_a_p = (1 - self.epsilon) + (self.epsilon / num_actions)
            else:
                prob_a_p = self.epsilon / num_actions
            expected_value += prob_a_p * self.Q_exp[s_, a_p]
        
        return expected_value
        
    def epsilon_greedy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q_exp.shape[1])
        else:
            return np.argmax(self.Q_exp[s, :])
    
    def sample_next_state(self, s, a):
        # we us the model's transition probabilities to sample the next state
        probs = [self.model.transition_probability(s, s_, a) for s_ in self.model.states]
        s_ = np.random.choice(self.model.states, p=probs)
        return s_
        
    def expected_sarsa(self):
        for ep in tqdm(range(self.episode)):
            s = self.model.start_state
            total_reward = 0
            iter_count = 0
            
            for t in range(self.maxit):
                iter_count += 1
                a = self.epsilon_greedy(s)
                
                s_ = self.sample_next_state(s,a)
                r = self.model.reward(s,a)
                total_reward += r
                
                if s_ == self.model.goal_state:
                    self.Q_exp[s,a] += self.alpha * (r - self.Q_exp[s,a])
                    break
                    
                exp_q = self.expected_q(s_)
                
                current_q = self.Q_exp[s,a]
                target_q = r + self.model.gamma * exp_q
                self.Q_exp[s,a] = current_q + self.alpha * (target_q - current_q)
                
                s = s_
                
            self.rewards[ep] = total_reward
            self.iter_snap.append(np.copy(self.Q_exp))
            self.iter_episode[ep] = iter_count
        
        pi = np.argmax(self.Q_exp, axis=1)
        V = np.max(self.Q_exp, axis=1)
        return self.Q_exp, self.iter_snap, V, pi, self.rewards, self.iter_episode
    