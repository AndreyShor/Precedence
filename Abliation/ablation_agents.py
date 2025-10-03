import numpy as np
import random

# Q-Learning Agent 0- Base Model (No Phi Penalty, No Threshold Penalty, No Rollback)
class QLearningAgent:
    def __init__(self, n_states, n_actions, q_table_init=0.0, alpha=0.1, gamma=0.99, epsilon=0.1):
        """Initialize Q-table and learning parameters."""
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.alpha = alpha                   # learning rate
        self.q_table_init = q_table_init
        self.gamma = gamma                   # discount factor
        self.epsilon = epsilon               # exploration rate (epsilon-greedy)
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.randrange(self.q_table.shape[1])
        else:
            # Exploit: choose action with highest Q-value, return q value
            return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using standard Q-learning update rule."""
        # Compute the TD target: r + γ * max Q(s', ·)
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])
        target_q = reward + self.gamma * max_future_q
        # Q-learning update: Q(s,a) ← Q(s,a) + α * (target_q - Q(s,a))
        self.q_table[state, action] += self.alpha * (target_q - self.q_table[state, action])
        # Return the next state (no special handling in standard Q-learning)
        return next_state
    
    def reset(self):
        """Reset the Q-table to zero."""
        self.q_table.fill(self.q_table_init)

# Ablation Agent 1: Phi Penalty Only (Phi Penalty, No Threshold Penalty, No Rollback)
class PrecedenceAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, 
                 K=10, alpha_phi=0.01, q_table_init=-1.0, lambda_precedence=1.0, phi_init=0.5):
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.phi = np.full((n_states, n_actions), phi_init, dtype=np.float32)
        # Fixed: remove trailing comma so these are scalars, not 1-tuples
        self.q_table_init = q_table_init
        self.phi_init = phi_init
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.K = K
        self.alpha_phi = alpha_phi
        self.lambda_precedence = lambda_precedence
        self.precedence_buffer = []
        self.time_step = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        self.time_step += 1
        
        # Update Phi table
        finished_records = []
        for rec in self.precedence_buffer:
            s0, a0, deadline = rec['s0'], rec['a0'], rec['deadline']
            if next_state == s0:
                y = 1
            elif self.time_step > deadline:
                y = 0
            else:
                continue
            self.phi[s0, a0] = (1 - self.alpha_phi) * self.phi[s0, a0] + self.alpha_phi * y
            finished_records.append(rec)
        for rec in finished_records:
            self.precedence_buffer.remove(rec)
        
        self.precedence_buffer.append({'s0': state, 'a0': action, 'deadline': self.time_step + self.K})
        
        # Apply precedence penalty to reward
        phi_val = self.phi[state, action]
        r_prime = reward - self.lambda_precedence * (1.0 - phi_val)
        
        # Standard Q-Learning update with penalized reward
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])

        target = r_prime + self.gamma * max_future_q
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
        
        return next_state, False  # No rollback
    
    def reset(self):
        # fill expects scalars — q_table_init / phi_init are now scalars
        self.q_table.fill(self.q_table_init)
        self.phi.fill(self.phi_init)
        self.precedence_buffer.clear()
        self.time_step = 0

# Ablation Agent 2: Threshold Penalty Only (No Phi, Threshold Penalty, no rollback)
class ThresholdPenaltyAgent:
    def __init__(self, n_states, n_actions, q_table_init=-1.0, alpha=0.1, gamma=0.99, epsilon=0.1, 
                 threshold=0.8, penalty=1.2):
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.q_table_init = q_table_init
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.threshold = threshold
        self.penalty = penalty
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        # Standard TD target
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])
        target = reward + self.gamma * max_future_q
        
        # Apply threshold penalty
        if target <= self.threshold * self.q_table[state, action]:
            beta = self.penalty
        else:
            beta = 1.0
            
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * beta * td_error
        
        return next_state, False  # No rollback
    
    def reset(self):
        self.q_table.fill(self.q_table_init)

# Ablation Agent 3: Rollback Only (No Phi, No Threshold Penalty, Rollback)
class RollBackAgent:
    def __init__(self, n_states, n_actions, q_table_init=-1.0, alpha=0.1, gamma=0.99, epsilon=0.1, 
                 threshold=0.8):
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.threshold = threshold
        self.q_table_init = q_table_init

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        # Standard TD target
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])
        target = reward + self.gamma * max_future_q
        
        # Apply threshold penalty and rollback
        if target <= self.threshold * self.q_table[state, action]:
            rollback_flag = True
        else:
            rollback_flag = False
            
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        
        # Rollback mechanism
        if rollback_flag and not done:
            return state, rollback_flag
        else:
            return next_state, rollback_flag

    def reset(self):
        self.q_table.fill(self.q_table_init)

# Ablation Agent 4: Threshold Penalty + Rollback (no Phi, Threshold Penalty + Rollback)
class RollbackAndThresholdPenaltyAgent:
    def __init__(self, n_states, n_actions, q_table_init=-1.0, alpha=0.1, gamma=0.99, epsilon=0.1, 
                 threshold=0.8, penalty=1.2):
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.q_table_init = q_table_init
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.threshold = threshold
        self.penalty = penalty
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        # Standard TD target
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])
        target = reward + self.gamma * max_future_q
        
        # Apply threshold penalty and rollback
        if target <= self.threshold * self.q_table[state, action]:
            beta = self.penalty
            rollback_flag = True
        else:
            beta = 1.0
            rollback_flag = False
            
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * beta * td_error
        
        # Rollback mechanism
        if rollback_flag and not done:
            return state, rollback_flag
        else:
            return next_state, rollback_flag
    
    def reset(self):
        self.q_table.fill(self.q_table_init)

# Ablation Agent 5: Precedence + Threshold Penalty (Phi Penalty + Threshold Penalty, no rollback
class PrecedenceThresholdPenaltyAgent:
    def __init__(self, n_states, n_actions, q_table_init=-1.0, alpha=0.1, gamma=0.99, epsilon=0.1,
                 K=10, alpha_phi=0.01, lambda_precedence=1.0, phi_init=0.5,
                 threshold=0.8, penalty=1.2):
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.phi = np.full((n_states, n_actions), phi_init, dtype=np.float32)
        self.q_table_init = q_table_init
        self.phi_init = phi_init
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.K = K
        self.alpha_phi = alpha_phi
        self.lambda_precedence = lambda_precedence
        self.threshold = threshold
        self.penalty = penalty
        self.phi_init = phi_init
        self.precedence_buffer = []
        self.time_step = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        self.time_step += 1
        
        # Update Phi table
        finished_records = []
        for rec in self.precedence_buffer:
            s0, a0, deadline = rec['s0'], rec['a0'], rec['deadline']
            if next_state == s0:
                y = 1
            elif self.time_step > deadline:
                y = 0
            else:
                continue
            self.phi[s0, a0] = (1 - self.alpha_phi) * self.phi[s0, a0] + self.alpha_phi * y
            finished_records.append(rec)
        for rec in finished_records:
            self.precedence_buffer.remove(rec)
        
        self.precedence_buffer.append({'s0': state, 'a0': action, 'deadline': self.time_step + self.K})
        
        # Apply precedence penalty
        phi_val = self.phi[state, action]
        r_prime = reward - self.lambda_precedence * (1.0 - phi_val)
        
        # Compute target with threshold penalty
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])
        target = r_prime + self.gamma * max_future_q
        
        # Apply threshold penalty (but no rollback)
        if target <= self.threshold * self.q_table[state, action]:
            beta = self.penalty
        else:
            beta = 1.0
            
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * beta * td_error
        
        return next_state, False  # No rollback
    
    def reset(self):
        self.q_table.fill(self.q_table_init)
        self.phi.fill(self.phi_init)
        self.precedence_buffer.clear()
        self.time_step = 0

# Ablation Agent 6: Precedence + Rollback (Phi Penalty, No Threshold Penalty, Rollback)
class PrecedenceRollbackAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 K=10, alpha_phi=0.01, q_table_init=-1.0, lambda_precedence=1.0, phi_init=0.5, threshold=0.8):
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.phi = np.full((n_states, n_actions), phi_init, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.K = K
        self.alpha_phi = alpha_phi
        self.lambda_precedence = lambda_precedence
        self.threshold = threshold
        self.phi_init = phi_init
        self.q_table_init = q_table_init
        self.precedence_buffer = []
        self.time_step = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        self.time_step += 1
        
        # Update Phi table
        finished_records = []
        for rec in self.precedence_buffer:
            s0, a0, deadline = rec['s0'], rec['a0'], rec['deadline']
            if next_state == s0:
                y = 1
            elif self.time_step > deadline:
                y = 0
            else:
                continue
            self.phi[s0, a0] = (1 - self.alpha_phi) * self.phi[s0, a0] + self.alpha_phi * y
            finished_records.append(rec)
        for rec in finished_records:
            self.precedence_buffer.remove(rec)
        
        self.precedence_buffer.append({'s0': state, 'a0': action, 'deadline': self.time_step + self.K})
        
        # Apply precedence penalty
        phi_val = self.phi[state, action]
        r_prime = reward - self.lambda_precedence * (1.0 - phi_val)
        
        # Standard Q-Learning update with penalized reward
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])
        target = r_prime + self.gamma * max_future_q
        
        # Rollback based on threshold (no penalty factor)
        rollback_flag = target <= self.threshold * self.q_table[state, action]
        
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error  # No penalty factor
        
        if rollback_flag and not done:
            return state, rollback_flag
        else:
            return next_state, rollback_flag
    
    def reset(self):
        self.q_table.fill(self.q_table_init)
        self.phi.fill(self.phi_init)
        self.precedence_buffer.clear()
        self.time_step = 0

# Ablation Agent 6: Precedence + Rollback + Threshold Penalty (Phi Penalty, No Threshold Penalty, Rollback)
class FullAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 K=10, alpha_phi=0.01, q_table_init=-1.0, lambda_precedence=1.0, phi_init=0.5, threshold=0.8, penalty=1.2):
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=np.float32)
        self.phi = np.full((n_states, n_actions), phi_init, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.K = K
        self.alpha_phi = alpha_phi
        self.lambda_precedence = lambda_precedence
        self.threshold = threshold
        self.phi_init = phi_init
        self.q_table_init = q_table_init
        self.precedence_buffer = []
        self.time_step = 0
        self.penalty = penalty


    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        self.time_step += 1
        
        # Update Phi table
        finished_records = []
        for rec in self.precedence_buffer:
            s0, a0, deadline = rec['s0'], rec['a0'], rec['deadline']
            if next_state == s0:
                y = 1
            elif self.time_step > deadline:
                y = 0
            else:
                continue
            self.phi[s0, a0] = (1 - self.alpha_phi) * self.phi[s0, a0] + self.alpha_phi * y
            finished_records.append(rec)
        for rec in finished_records:
            self.precedence_buffer.remove(rec)
        
        self.precedence_buffer.append({'s0': state, 'a0': action, 'deadline': self.time_step + self.K})
        
        # Apply precedence penalty
        phi_val = self.phi[state, action]
        r_prime = reward - self.lambda_precedence * (1.0 - phi_val)
        
        # Standard Q-Learning update with penalized reward
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])
        target = r_prime + self.gamma * max_future_q

        # Rollback based on threshold and penalty factor
        rollback_flag = False
        if target <= self.threshold * self.q_table[state, action]:
            beta = self.penalty
            rollback_flag = True
        else:
            beta = 1.0

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * beta * td_error
        
        if rollback_flag and not done:
            return state, rollback_flag
        else:
            return next_state, rollback_flag
    
    def reset(self):
        self.q_table.fill(self.q_table_init)
        self.phi.fill(self.phi_init)
        self.precedence_buffer.clear()
        self.time_step = 0

# Additional missing agents to complete ablation study
