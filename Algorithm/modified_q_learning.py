import numpy as np
import random

class ModifiedQLearningAgent:
    """
    Q-Learning agent enhanced with rollback and precedence (reversibility) mechanism.

    Key features:
    - Maintains a reversibility table Φ[s, a], estimating probability of returning to state s after taking action a within K steps.
    - Uses a FIFO buffer of pending transitions to update Φ via an exponential moving average (α_phi).
    - Penalizes low-reversibility transitions by shrinking the reward: r' = r - λ * (1 - Φ[s, a]).
    - Applies a penalty factor β when the TD target falls below a threshold T of the current Q-value.
    - Executes an explicit rollback (stays in the same state) on heavy mistakes (when penalty applied).
    """
    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        K=10,
        alpha_phi=0.01,
        lambda_precedence=1.0,
        phi_init=0.5,
        threshold=0.8,
        penalty=0.8
    ):
        # Q-table
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        # Reversibility table Φ initialized to phi_init
        self.phi = np.full((n_states, n_actions), phi_init, dtype=np.float32)

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Precedence (reversibility) parameters
        self.K = K  # horizon for tracking reversibility
        self.alpha_phi = alpha_phi  # learning rate for Φ
        self.lambda_precedence = lambda_precedence  # λ in penalized reward

        # Rollback & penalty parameters
        self.threshold = threshold  # T: fraction of Q below which to apply penalty
        self.penalty = penalty  # β: penalty factor when triggered

        # FIFO buffer for pending reversibility records
        # Each record: {'s0': state, 'a0': action, 'deadline': time_step + K}
        self.precedence_buffer = []
        self.time_step = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """
        Perform a single update of Q and Φ, and determine the next state (with rollback if needed).

        Returns:
            next_state_effective: the state to use for the next time-step (state if rolled back, else next_state).
        """
        # 1. Advance time
        self.time_step += 1

        # 2. Process precedence_buffer for early returns or timeouts
        finished = []
        for rec in self.precedence_buffer:
            s0, a0, deadline = rec['s0'], rec['a0'], rec['deadline']
            if next_state == s0:
                y = 1  # early return observed
            elif self.time_step > deadline:
                y = 0  # timeout without return
            else:
                continue  # still pending
            # Update Φ via exponential moving average (Eq.12)
            self.phi[s0, a0] = (1 - self.alpha_phi) * self.phi[s0, a0] + self.alpha_phi * y
            finished.append(rec)
        # Remove finished records
        for rec in finished:
            self.precedence_buffer.remove(rec)

        # 3. Enqueue new record for current transition
        self.precedence_buffer.append({
            's0': state,
            'a0': action,
            'deadline': self.time_step + self.K
        })

        # 4. Compute penalized reward r' = r - λ * (1 - Φ[state, action]) (Eq.15)
        phi_val = self.phi[state, action]
        r_prime = reward - self.lambda_precedence * (1.0 - phi_val)

        # 5. Compute max future Q-value
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])

        # 6. TD target & error
        target = r_prime + self.gamma * max_future_q  # Eq.16
        delta = target - self.q_table[state, action]

        # 7. Determine penalty factor β (Eq.17)
        if target <= self.threshold * self.q_table[state, action]:
            beta = self.penalty
            rollback = True
        else:
            beta = 1.0
            rollback = False

        # 8. Update Q-value (Eq.18)
        self.q_table[state, action] += self.alpha * beta * delta

        # 9. Rollback decision (Eq.19)
        if rollback:
            return state  # stay in the same state (rollback)
        return next_state  # proceed as normal
