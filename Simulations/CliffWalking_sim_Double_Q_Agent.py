import re
from cv2 import log
import gymnasium as gym
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Algorithm.Brain import DoubleQLearningAgent
from logger import loggerCSV

# Initialize the environment
env = gym.make("CliffWalking-v0")

# Initialize Q-learning agent

# Parameters for Q-learning
episodes = 100  # Number of episodes
max_steps = 800  # Max steps per episode

seeds = list(range(episodes)) 

# Lists to collect statistics
episode_rewards = []  # Total reward for each episode
episode_lengths = []  # Length (number of steps) per episode
episode_falls = []  # Number of falls off the cliff per episode

agent = DoubleQLearningAgent(n_actions=env.action_space.n, n_states=env.observation_space.n) # type: ignore
logger = loggerCSV("CliffWalking_sim_Double_Q_Agent.csv", "cliff")
# Training loop
for episode in range(episodes): # Reset agent's Q-table for each episode
    agent.reset()  # Reset agent's Q-table for each episode
    state, info = env.reset(seed=seeds[episode])  # Reset environment for each episod
    done = False
    total_reward = 0.0
    steps = 0
    number_falls = 0  # Count of falls off the cliff
    
    # Run the episode
    for _ in range(max_steps):
        # Choose action from the current state
        action = agent.select_action(state)
        
        # Take the action and observe the next state, reward, and termination
        next_state, reward, done, truncated, info = env.step(action)

        if reward == -100:  # If the agent falls off the cliff
            number_falls += 1
        # Update Q-table with the agent's experience (It will also return the next state), the same as from env

        next_state = agent.update(state, action, reward, next_state, done)
        
        # Accumulate total reward and step count
        total_reward += float(reward)
        steps += 1
        
        # Move to the next state
        state = next_state
        
        # If done (agent reaches the goal
        if done:
            break
    
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
