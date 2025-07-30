import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """Initialize the Q-learning agent."""
        self.q_table = np.ones((n_states, n_actions), dtype=np.float32)
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        """Select an action using an epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.randrange(self.q_table.shape[1])
        else:
            # Exploit: choose the action with the highest Q-value
            return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Update the Q-table based on experience."""
        # If done, there is no future reward
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])

        # Target Q-value
        target_q = reward + self.discount_factor * max_future_q
        
        # Update the Q-value
        self.q_table[state, action] += self.learning_rate * (target_q - self.q_table[state, action])
        
        return next_state
