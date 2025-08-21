import numpy as np # type: ignore
import random
from typing import Optional


# Q-Learning Agent (off-policy TD control)
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """Initialize Q-table and learning parameters."""
        self.q_table = np.ones((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha                   # learning rate
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
        self.q_table.fill(0.0)
    
class QLearningReversableAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, revers_penaltyLimit = 0.8):
        """Initialize the Q-learning agent."""
        self.q_table = np.ones((n_states, n_actions), dtype=np.float32)
        self.q_table.fill(-1.0)
        print(f"Q-table initialized with shape: {self.q_table.shape}")
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.epsilon = epsilon
        self.revers_penaltyLimit = revers_penaltyLimit

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

        # Penalty Factor
        penalty = 1
        if target_q < self.revers_penaltyLimit * self.q_table[state, action]:
            # print(f"Penalty applied: {target_q} < {self.revers_penaltyLimit} * {self.q_table[state, action]}")
            penalty = 1.2
            
        
        # Update the Q-value with penalty
        self.q_table[state, action] += self.learning_rate * penalty * (target_q - self.q_table[state, action])

        # If penalty is 0.8, rollback the state
        if penalty == 1.2:
            # Rollback the state
            return state, True
        else:
            return next_state, False
        
    def reset(self):
        """Reset the Q-table to zero."""
        self.q_table.fill(-1.0)
 
# SARSA Agent (on-policy TD control)
class SarsaAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """Initialize Q-table and learning parameters."""
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        self.q_table.fill(-1.0)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state):
        """Epsilon-greedy action selection (same as QLearningAgent)."""
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, next_action, done):
        """Update Q-table using standard SARSA update rule."""
        # SARSA uses the actual next action's Q-value (on-policy TD target)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, next_action]
        # SARSA update: Q(s,a) ← Q(s,a) + α * (target - Q(s,a))
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
        return next_state  # Next state (no rollback in standard SARSA)
    
    def reset(self):
        """Reset the Q-table to zero."""
        self.q_table.fill(0.0)

class ExpectedSarsaAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, init_value=-1.0):
        """Initialize Q-table and learning parameters."""
        self.q_table = np.full((n_states, n_actions), fill_value=init_value, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state):
        """Epsilon-greedy action selection (same as your SARSA/QLearning)."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def _expected_q(self, next_state):
        """
        Compute E_a'[Q(s',a')] under the ε-greedy policy:
          P(a*=argmax)=1-ε+ε/|A|,  P(other)=ε/|A|
        """
        q_next = self.q_table[next_state]  # shape: (n_actions,)
        best_a = int(np.argmax(q_next))

        # Start with uniform ε/|A| over all actions, then add (1-ε) to best action
        pi = np.full(self.n_actions, self.epsilon / self.n_actions, dtype=np.float32)
        pi[best_a] += (1.0 - self.epsilon)

        # Expected value under π
        return float(np.dot(pi, q_next))

    def update(self, state, action, reward, next_state, next_action_unused, done):
        """
        Expected SARSA update:
          target = r + γ * E_{a'~π}[ Q(s',a') ]  (if not done), else r
          Q(s,a) ← Q(s,a) + α [target − Q(s,a)]
        Note: next_action is unused (kept in signature for drop-in compatibility).
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * self._expected_q(next_state)

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return next_state  # keep the same return as your SARSA agent

    def reset(self):
        """Reset the Q-table to zero (matches your SARSA.reset behavior)."""
        self.q_table.fill(0.0)


class DoubleQLearningAgent:
    """
    Tabular Double Q-Learning (Hasselt, 2010).

    - Maintains two Q-tables: Q_A and Q_B.
    - On each step, updates exactly one table:
        * If updating Q_A: choose a* = argmax_a Q_A(s',a), then evaluate using Q_B(s', a*).
        * If updating Q_B: choose b* = argmax_a Q_B(s',a), then evaluate using Q_A(s', b*).
    - Control uses ε-greedy over the *combined* estimate: Q_A + Q_B.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        init_value: float = 0.0,
        update_mode: str = "random",   # "random" or "alternate"
        seed: Optional[int] = None,
    ) -> None:
        # Hyperparameters
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.init_value = float(init_value)

        # Update scheduling
        assert update_mode in {"random", "alternate"}
        self.update_mode = update_mode
        self._alternate_flag = False  # toggled if update_mode == "alternate"

        # PRNG (single source for reproducibility)
        self.rng = np.random.default_rng(seed)

        # Q-tables
        self.q_a = np.full((n_states, n_actions), self.init_value, dtype=np.float32)
        self.q_b = np.full((n_states, n_actions), self.init_value, dtype=np.float32)


    def select_action(self, state: int) -> int:
        """ε-greedy over (Q_A + Q_B)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q_sum = self.q_a[state] + self.q_b[state]
        return self._argmax_uniform(q_sum)


    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> int:
        """
        Perform one Double Q-learning update using transition (s, a, r, s', done).
        Returns next_state for drop-in loop compatibility.
        """
        which = self._choose_table_to_update()

        if which == "A":
            # Select via Q_A, evaluate via Q_B
            target = reward
            if not done:
                a_star = self._argmax_uniform(self.q_a[next_state])
                target += self.gamma * float(self.q_b[next_state, a_star])
            td = target - float(self.q_a[state, action])
            self.q_a[state, action] += self.alpha * td

        else:  # which == "B"
            # Select via Q_B, evaluate via Q_A
            target = reward
            if not done:
                b_star = self._argmax_uniform(self.q_b[next_state])
                target += self.gamma * float(self.q_a[next_state, b_star])
            td = target - float(self.q_b[state, action])
            self.q_b[state, action] += self.alpha * td

        return next_state
    
    def _argmax_uniform(self, values: np.ndarray) -> int:
        """Argmax with *uniform* tie-breaking."""
        max_val = values.max()
        ties = np.flatnonzero(values == max_val)
        return int(self.rng.choice(ties))

    def _choose_table_to_update(self) -> str:
        """Return 'A' or 'B' per update_mode."""
        if self.update_mode == "alternate":
            self._alternate_flag = not self._alternate_flag
            return "A" if self._alternate_flag else "B"
        return "A" if self.rng.random() < 0.5 else "B"

    def reset(self, to_value: Optional[float] = None) -> None:
        """
        Reset both Q-tables. If to_value is None, uses init_value.
        """
        val = self.init_value if to_value is None else float(to_value)
        self.q_a.fill(val)
        self.q_b.fill(val)


# Modified Q-Learning Agent with Rollback and Precedence
class ModifiedQLearningAgent:
    """
    Q-Learning agent with a rollback mechanism and precedence (Φ) estimates.
    - Maintains a reversibility table Φ[s,a] that estimates the probability of returning to state s after taking action a within K steps.
    - Uses a FIFO buffer of recent transitions (length K) to update Φ via an exponential moving average.
    - Computes a penalized reward r' = r – λ * (1 – Φ[s,a]) to downweight bad transitions.
    - Applies a penalty factor β (e.g. P=0.8 by default) if the TD target falls below a threshold T of the current Q-value.
    - If such a “heavy mistake” is detected, performs a **rollback**: the next state is reset to the current state (undo the transition).
    """
    def __init__(self, n_states, n_actions, q_table_init = 1, alpha=0.1, gamma=0.99, epsilon=0.1,
                 K=10, alpha_phi=0.01, lambda_precedence=1.0, phi_init=0.5,
                 threshold=0.8, penalty=1.2):
        # Q-value table and Φ table (initialize Φ values optimistically at phi_init)
        self.q_table_init = q_table_init
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=float)

        self.phi     = np.full((n_states, n_actions), phi_init, dtype=np.float32)
        self.phi_init = phi_init  # Initial value for Φ estimates
        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Precedence (reversibility) parameters
        self.K = K                          # horizon for tracking reversibility (FIFO length)
        self.alpha_phi = alpha_phi          # learning rate for Φ updates
        self.lambda_precedence = lambda_precedence  # λ coefficient for reward penalty
        # Rollback mechanism parameters
        self.threshold = threshold          # T: threshold fraction of Q to trigger penalty
        self.penalty = penalty              # P: penalty factor β for updates when triggered
        # FIFO buffer to track recent transitions for Φ updates
        # Each entry: {'s0': state, 'a0': action, 'deadline': time_step + K}
        self.precedence_buffer = []
        self.time_step = 0
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table and Φ-table for a single transition, possibly performing a rollback.
        Returns the effective next state (current state again if rolled back, else the true next_state).
        """
        self.time_step += 1  # advance global time-step counter
        
        # 1. Update Φ for any pending transitions that have finished their window
        finished_records = []
        for rec in self.precedence_buffer:
            s0, a0, deadline = rec['s0'], rec['a0'], rec['deadline']
            if next_state == s0:
                # Early return observed: came back to s0 within K steps
                y = 1
            elif self.time_step > deadline:
                # Timeout: K steps passed without returning to s0
                y = 0
            else:
                # Still within window and not returned; keep waiting
                continue
            # Update the reversibility estimate Φ[s0,a0] with an exponential moving average
            self.phi[s0, a0] = (1 - self.alpha_phi) * self.phi[s0, a0] + self.alpha_phi * y
            finished_records.append(rec)
        # Remove processed records from the FIFO buffer
        for rec in finished_records:
            self.precedence_buffer.remove(rec)
        
        # 2. Enqueue the current transition (state, action) with a deadline for return
        self.precedence_buffer.append({
            's0': state, 'a0': action, 'deadline': self.time_step + self.K
        })
        # (The FIFO naturally caps at K entries maximum, since each record expires after K steps.)
        
        # 3. Compute penalized reward r' integrating the precedence penalty
        # If Φ[s,a] is low (irreversible transition), (1 - Φ) is high, so r' is significantly lower than actual r.
        phi_val = self.phi[state, action]
        r_prime = reward - self.lambda_precedence * (1.0 - phi_val)
        
        # 4. Compute the standard Q-learning TD target using r'
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])

        target = r_prime + self.gamma * max_future_q
        td_error = target - self.q_table[state, action]
        
        # 5. Determine penalty factor β based on threshold condition 
        # If the new estimated return falls below T * current Q, we amplify correction by factor P (β = P).
        current_q = self.q_table[state, action]
        
        if target <= self.threshold * current_q:
            beta = self.penalty
            rollback_flag = True
        else:
            beta = 1.0
            rollback_flag = False
        
        # 6. Update Q-value with scaled learning rate β
        # Q(s,a) ← Q(s,a) + α * β * (td_error)
        self.q_table[state, action] += self.alpha * beta * td_error
        
        # 7. Rollback mechanism: if triggered, do *not* advance to next_state.
        if rollback_flag and not done:
            # Stay in the same state (agent rolls back the transition)
            return state, rollback_flag  
        else:
            return next_state, rollback_flag
        
    def reset(self):
        """Reset the Q-table, Φ-table, precedence buffer, and timestep."""
        # Zero out Q-values
        self.q_table.fill(self.q_table_init)
        # Re-initialize all Φ entries
        self.phi.fill(self.phi_init)
        # Clear any pending records
        self.precedence_buffer.clear()
        # Reset the global step counter
        self.time_step = 0


class ModifiedSARSAAgent:
    """
    SARSA agent with a rollback mechanism and precedence (Φ) estimates.
    - Maintains a reversibility table Φ[s,a] that estimates the probability of returning to state s after taking action a within K steps.
    - Uses a FIFO buffer of recent transitions (length K) to update Φ via an exponential moving average.
    - Computes a penalized reward r' = r – λ * (1 – Φ[s,a]) to downweight bad transitions.
    - Applies a penalty factor β if the TD target falls below a threshold T of the current Q-value.
    - If such a “heavy mistake” is detected, performs a rollback: the next state is reset to the current state (undo the transition).
    """
    def __init__(self, n_states, n_actions, q_table_init=1, alpha=0.1, gamma=0.99, epsilon=0.1,
                 K=10, alpha_phi=0.01, lambda_precedence=1.0, phi_init=0.5,
                 threshold=0.8, penalty=1.2):
        # Q-value table and Φ table (initialize Φ values optimistically at phi_init)
        self.q_table_init = q_table_init
        self.q_table = np.full((n_states, n_actions), q_table_init, dtype=float)

        self.phi      = np.full((n_states, n_actions), phi_init, dtype=np.float32)
        self.phi_init = phi_init

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Precedence (reversibility) parameters
        self.K = K                          # horizon for tracking reversibility (FIFO length)
        self.alpha_phi = alpha_phi          # learning rate for Φ updates
        self.lambda_precedence = lambda_precedence  # λ coefficient for reward penalty

        # Rollback mechanism parameters
        self.threshold = threshold          # T: thresh old fraction of Q to trigger penalty
        self.penalty = penalty              # β: penalty factor for updates when triggered

        # FIFO buffer to track recent transitions for Φ updates
        # Each entry: {'s0': state, 'a0': action, 'deadline': time_step + K}
        self.precedence_buffer = []
        self.time_step = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update for a single transition with Φ/penalty/rollback.
        Returns (effective_next_state, rollback_flag).
        NOTE: next_action should be the action actually chosen in next_state under the current policy.
        """
        self.time_step += 1

        # 1) Update Φ for any pending transitions whose window finished
        finished_records = []
        for rec in self.precedence_buffer:
            s0, a0, deadline = rec['s0'], rec['a0'], rec['deadline']
            if next_state == s0:
                y = 1  # returned to s0 within K steps
            elif self.time_step > deadline:
                y = 0  # timed out
            else:
                continue
            self.phi[s0, a0] = (1 - self.alpha_phi) * self.phi[s0, a0] + self.alpha_phi * y
            finished_records.append(rec)
        for rec in finished_records:
            self.precedence_buffer.remove(rec)

        # 2) Enqueue the current (s,a) with deadline
        self.precedence_buffer.append({
            's0': state, 'a0': action, 'deadline': self.time_step + self.K
        })

        # 3) Penalized reward r' using Φ
        phi_val = self.phi[state, action]
        r_prime = reward - self.lambda_precedence * (1.0 - phi_val)

        # 4) SARSA TD target using the actually chosen next_action
        if done:
            next_q = 0.0
        else:
            next_q = self.q_table[next_state, next_action]

        target = r_prime + self.gamma * next_q
        current_q = self.q_table[state, action]
        td_error = target - current_q

        # 5) Threshold penalty & rollback flag
        if target <= self.threshold * current_q:
            beta = self.penalty
            rollback_flag = True
        else:
            beta = 1.0
            rollback_flag = False

        # 6) Update Q(s,a)
        self.q_table[state, action] += self.alpha * beta * td_error

        # 7) Rollback: if triggered and episode not done, stay in the same state
        if rollback_flag and not done:
            return state, rollback_flag
        else:
            return next_state, rollback_flag

    def reset(self):
        """Reset Q-table, Φ-table, buffer, and timestep."""
        self.q_table.fill(self.q_table_init)
        self.phi.fill(self.phi_init)
        self.precedence_buffer.clear()
        self.time_step = 0
