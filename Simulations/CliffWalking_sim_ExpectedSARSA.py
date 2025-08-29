import re
from cv2 import log
import gymnasium as gym
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Algorithm.Brain import ExpectedSarsaAgent
from logger import loggerCSV

# Initialize the environment
env = gym.make("CliffWalking-v0")

# Initialize Q-learning agent

# Parameters for Q-learning
episodes = 1000  # Number of episodes
max_steps = 400  # Max steps per episode

seeds = list(range(episodes)) 

# Lists to collect statistics
episode_rewards = []  # Total reward for each episode
episode_lengths = []  # Length (number of steps) per episode
episode_falls = []  # Number of falls off the cliff per episode

agent = ExpectedSarsaAgent(n_actions=env.action_space.n, n_states=env.observation_space.n) # type: ignore
logger = loggerCSV("CliffWalking_sim_Expected_SARSA.csv", "cliff")
# Training loop
for episode in range(episodes):# Reset agent's Q-table for each episode
    agent.reset()  # Reset agent's Q-table for each episode
    state, info = env.reset(seed=seeds[episode])  # Reset environment for each episod
    done = False
    total_reward = 0.0
    steps = 0
    number_falls = 0  # Count of falls off the cliff

    # Choose initial action using the current policy (SARSA is on-policy)
    action = agent.select_action(state)

    total_reward = 0.0
    steps = 0
    number_falls = 0

    for _ in range(max_steps):
        next_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        if reward == -100:
            number_falls += 1

        # Choose the next action from next_state (on-policy)
        if not done:
            next_action = agent.select_action(next_state)
        else:
            next_action = None  # not used in update when done=True

        # Standard SARSA update uses the actual next_action’s Q-value
        agent.update(state, action, reward, next_state,
                     next_action if next_action is not None else 0,
                     done)

        total_reward += float(reward)
        steps += 1

        if done:
            break

        # Move forward: (s, a) <- (s', a')
        state = next_state
        action = next_action
    
    # Collect statistics after each episode
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    episode_falls.append(number_falls)
    
    # Optionally print stats after every 100 episodes for feedback
    logger.log_cliff(episode, total_reward, steps, number_falls)

    if episode % 10 == 0:
        print(f"Episode {episode} - Total Reward: {total_reward} - Steps: {steps} - Falls: {number_falls}")
# After training, calculate some statistics
average_reward = np.mean(episode_rewards)
average_length = np.mean(episode_lengths)
average_falls = np.mean(episode_falls)

logger.close()

print(f"Training complete after {episodes} episodes!")
print(f"Average Reward: {average_reward}")
print(f"Average Falls per Episode: {average_falls}")
print(f"Average Steps per Episode: {average_length}")
