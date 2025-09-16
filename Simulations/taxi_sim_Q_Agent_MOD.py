import re
import gymnasium as gym
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Algorithm.Brain import FullAgent
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

episode_rollbacks = []  # Number of rollbacks

agent = FullAgent(n_actions=env.action_space.n, n_states=env.observation_space.n, q_table_init=-1.0, alpha_phi=0.01, lambda_precedence=1.0, phi_init=0.5, threshold=3, penalty=1.4, K=10) # type: ignore
logger = loggerCSV("taxi_sim_Q_Agent_MOD.csv", "taxi_mod")

# Training loop
for episode in range(episodes):
    agent.reset()  # Reset agent's Q-table for each episodesim
    state, info = env.reset(seed=seeds[episode])  # Reset environment for each episod
    done = False
    total_reward = 0.0
    steps = 0
    drop_passsenger_pick_ghost = 0  # Count of falls off the cliff
    delivered_passenger = 0  # Count of successful passenger deliveries
    number_rollbacks = 0  # Count of rollbacks
    
    # Run the episode
    for _ in range(max_steps):
        # Choose action from the current state
        action = agent.select_action(state)

        # Take the action and observe the next state, reward, and termination
        next_state, reward, done, truncated, info = env.step(action)
        
        # Take the action and observe the next state, reward, and termination
        next_state, rollback_flag =  agent.update(state, action, reward, next_state, done)

        if reward == -10 and rollback_flag == False:  # If the agent falls off the cliff
            drop_passsenger_pick_ghost += 1

        if reward == 20:  
            delivered_passenger += 1

        # Accumulate total reward and step count
        if rollback_flag:
            total_reward += 0 # No reward on rollback
            number_rollbacks += 1
        else:
            total_reward += float(reward)

        # Accumulate total reward and step count
        steps += 1
        
        # Move to the next state
        state = next_state
        
        # If done (agent reaches the goal
        if done:
            break
    
    # Collect statistics after each episode
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    episode_drops.append(drop_passsenger_pick_ghost)
    episode_deliveries.append(delivered_passenger)
    episode_rollbacks.append(number_rollbacks)
    
    logger.log_taxi_Mod(episode, total_reward, steps, drop_passsenger_pick_ghost, delivered_passenger, number_rollbacks)
    # Optionally print stats after every 100 episodes for feedback
    if episode % 100 == 0:
        print(f"Episode {episode} - Total Reward: {total_reward} - Steps: {steps} - Falls: {drop_passsenger_pick_ghost}, Deliveries: {delivered_passenger}")

# After training, calculate some statistics
average_reward = np.mean(episode_rewards)
average_length = np.mean(episode_lengths)
average_deliveries = np.mean(episode_deliveries)
average_drops = np.mean(episode_drops)

average_rollbacks = np.mean(episode_rollbacks)


print(f"Training complete after {episodes} episodes!")
print(f"Average Reward: {average_reward}")
print(f"Average Steps per Episode: {average_length}")
print(f"Average Deliveries of Passenger per Episode: {average_deliveries}")
print(f"Number of drops or pickups of ghost passengers {average_drops}")
print(f"Average Rollbacks per Episode: {average_rollbacks}")
