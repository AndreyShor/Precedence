import re
import gymnasium as gym
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Algorithm.Brain import SarsaAgent
from logger import loggerCSV
# Initialize the environment
env = gym.make("Taxi-v3")

# Initialize Q-learning agent

# Parameters for Q-learning
episodes = 1000  # Number of episodes
max_steps = 1000  # Max steps per episode

seeds = list(range(episodes)) 

# Lists to collect statistics
episode_rewards = []  # Total reward for each episode
episode_lengths = []  # Length (number of steps) per episode
episode_deliveries = []  # Number of successful passenger deliveries
episode_drops = []  # Number of drops or pickups of ghost passengers

agent = SarsaAgent(n_actions=env.action_space.n, n_states=env.observation_space.n) # type: ignore
logger = loggerCSV("taxi_sim_SARSA_Agent.csv", "taxi")
# Training loop
for episode in range(episodes):
    agent.reset()
    state, info = env.reset(seed=seeds[episode])  # Reset environment for each episod
    done = False
    total_reward = 0.0
    steps = 0
    drop_passsenger_pick_ghost = 0  # Count of falls off the cliff
    delivered_passenger = 0  # Count of successful passenger deliveries
    
    action = agent.select_action(state)

    # Run the episode
    for _ in range(max_steps):

        next_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        if reward == -10:  # If the agent falls off the cliff
            drop_passsenger_pick_ghost += 1

        if reward == 20:  
            delivered_passenger += 1

        if not done:
            next_action = agent.select_action(next_state)
        else:
            next_action = None  # not used in update when done=True
        
        agent.update(state, action, reward, next_state,
                     next_action if next_action is not None else 0,
                     done)
        
        # Accumulate total reward and step count
        total_reward += float(reward)
        steps += 1

        # If done (agent reaches the goal
        if done:
            break
        
        # Move to the next state
        state = next_state
        action = next_action
    
    # Collect statistics after each episode
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    episode_drops.append(drop_passsenger_pick_ghost)
    episode_deliveries.append(delivered_passenger)
    logger.log_taxi(episode, total_reward, steps, drop_passsenger_pick_ghost, delivered_passenger)
    # Optionally print stats after every 100 episodes for feedback
    if episode % 100 == 0:
        print(f"Episode {episode} - Total Reward: {total_reward} - Steps: {steps} - Falls: {drop_passsenger_pick_ghost}, Deliveries: {delivered_passenger}")

# After training, calculate some statistics
average_reward = np.mean(episode_rewards)
average_length = np.mean(episode_lengths)
average_deliveries = np.mean(episode_deliveries)
average_drops = np.mean(episode_drops)


print(f"Training complete after {episodes} episodes!")
print(f"Average Reward: {average_reward}")
print(f"Average Steps per Episode: {average_length}")
print(f"Average Deliveries of Passenger per Episode: {average_deliveries}")
print(f"Number of drops or pickups of ghost passengers {average_drops}")
