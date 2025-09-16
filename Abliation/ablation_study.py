#!/usr/bin/env python3
"""
Ablation Study for Reversibility in Learning
Exactly matches existing experimental setup and parameters
"""

import gymnasium as gym
import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Algorithm.Brain import QLearningAgent
from logger import loggerCSV

from ablation_agents import QLearningAgent, PrecedenceAgent, ThresholdPenaltyAgent, RollBackAgent, RollbackAndThresholdPenaltyAgent, PrecedenceThresholdPenaltyAgent, PrecedenceRollbackAgent, FullAgent

# CRITICAL: All ablation agents must match the exact initialization and behavior patterns
# of the original implementations

def run_exact_experiment(env_name, agent_class, agent_params, agent_name, episodes=1000):
    """
    Run experiment with EXACT same setup as existing simulation files
    """
    print(f"Running {agent_name} on {env_name}...")
    
    # Use EXACT same environment versions and parameters as existing sims
    env = gym.make(env_name)
    
    # Match existing parameter settings
    if env_name == "CliffWalking-v0":
        max_steps = 400
        failure_reward = -100  # Falls off cliff
        success_reward = None
    else:  # Taxi-v3
        max_steps = 1000
        failure_reward = -10   # Illegal pickup/dropoff
        success_reward = 20    # Successful delivery
    
    # Create agent with exact parameters
    agent = agent_class(
        n_actions=env.action_space.n,  # type: ignore
        n_states=env.observation_space.n, # type: ignore
        **agent_params
    )
    
    # Setup logging (exact match to existing pattern)
    log_filename = f"ablation_{env_name}_{agent_name}.csv"
    has_rollback = hasattr(agent, 'penalty') or 'Modified' in agent_class.__name__
    
    if has_rollback:
        logger_mode = "cliff_mod" if env_name == "CliffWalking-v0" else "taxi_mod"
    else:
        logger_mode = "cliff" if env_name == "CliffWalking-v0" else "taxi"
    
    logger = loggerCSV(log_filename, logger_mode)
    
    # Statistics tracking (exact match to existing sims)
    episode_rewards = []
    episode_lengths = []
    episode_failures = []  # Falls (cliff) or illegal actions (taxi)
    episode_successes = []  # Deliveries (taxi only)
    episode_rollbacks = []
    
    # Use same seeding pattern as existing sims
    seeds = list(range(episodes))
    
    # Training loop (EXACT match to existing simulation structure)
    for episode in range(episodes):
        agent.reset()  # Reset agent for each episode
        state, info = env.reset(seed=seeds[episode])
        done = False
        total_reward = 0.0
        steps = 0
        failures = 0
        successes = 0
        rollbacks = 0
        
        # Episode loop (match existing max_steps pattern)
        for _ in range(max_steps):
            # Choose action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update agent - handle different return signatures
            try:
                # Try rollback-capable agents (returns tuple)
                next_state, rollback_flag = agent.update(state, action, reward, next_state, done)
                
                # Handle reward accounting for rollbacks
                if rollback_flag:
                    rollbacks += 1
                    total_reward += 0  # No reward on rollback (match existing MOD pattern)
                else:
                    total_reward += float(reward)
                    
            except (TypeError, ValueError):
                # Standard agents (return single value)
                next_state = agent.update(state, action, reward, next_state, done)
                rollback_flag = False
                total_reward += float(reward)
            
            # Track environment-specific metrics (exact match to existing sims)
            if env_name == "CliffWalking-v0":
                if reward == failure_reward and not rollback_flag:
                    failures += 1  # Falls off cliff
            else:  # Taxi-v3
                if reward == failure_reward and not rollback_flag:
                    failures += 1  # Illegal pickup/dropoff
                if reward == success_reward:
                    successes += 1  # Successful delivery
            
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Collect statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_failures.append(failures)
        episode_successes.append(successes)
        episode_rollbacks.append(rollbacks)
        
        # Log with exact same pattern as existing sims
        if has_rollback:
            if env_name == "CliffWalking-v0":
                logger.log_cliff_Mod(episode, total_reward, steps, failures, rollbacks)
            else:
                logger.log_taxi_Mod(episode, total_reward, steps, failures, successes, rollbacks)
        else:
            if env_name == "CliffWalking-v0":
                logger.log_cliff(episode, total_reward, steps, failures)
            else:
                logger.log_taxi(episode, total_reward, steps, failures, successes)
        
        # Progress updates (match existing pattern)
        if episode % 100 == 0:
            print(f"  Episode {episode}: Reward={total_reward:.1f}, Steps={steps}, Failures={failures}, Rollbacks={rollbacks}")
    
    logger.close()
    
    # Calculate final statistics (match existing pattern)
    return {
        'agent': agent_name,
        'environment': env_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_steps': np.mean(episode_lengths),
        'std_steps': np.std(episode_lengths),
        'mean_failures': np.mean(episode_failures),
        'std_failures': np.std(episode_failures),
        'mean_successes': np.mean(episode_successes),
        'std_successes': np.std(episode_successes),
        'mean_rollbacks': np.mean(episode_rollbacks),
        'std_rollbacks': np.std(episode_rollbacks)
    }

def run_corrected_ablation_study(episodes=1000):
    """
    Run corrected ablation study with EXACT parameter matching
    """
    print("=" * 80)
    print("CORRECTED ABLATION STUDY - Exact Parameter Matching")
    print("=" * 80)
    
    # Define ablation configurations using EXACT parameters from existing sims
    ablation_configs = {
        # Environment: CliffWalking-v0
        'CliffWalking-v0': {
            'Baseline': {
                'class': QLearningAgent,
                'params': {}  # Uses default initialization (zeros after reset)
            },
            'RollbackOnly': {
                'class': RollBackAgent,
                'params': {'q_table_init': -1.0, 'threshold': 3}  # Match MOD params but no Phi
            },
            'PrecedenceOnly': {
                'class': PrecedenceAgent,
                'params': {'q_table_init': -1.0, 'K': 10, 'alpha_phi': 0.01, 
                          'lambda_precedence': 1.0, 'phi_init': 0.5}  # Phi only, no threshold
            },
            'ThresholdPeAgent': {
                'class': ThresholdPenaltyAgent,
                'params': {'q_table_init': -1.0, 'threshold': 3, 'penalty': 1.4}  # Threshold only
            },
            'Roll_Threshold': {
                'class': RollbackAndThresholdPenaltyAgent,
                'params': {'q_table_init': -1.0, 'threshold': 3, 'penalty': 1.4}  # Threshold only
            },

            'Precedence_Th': {
                'class': PrecedenceThresholdPenaltyAgent,
                'params': {'q_table_init': -1.0,'alpha_phi': 0.01, 'lambda_precedence': 1.0, 'phi_init': 0.5, 'threshold': 3, 'penalty': 1.4, 'K': 10}  # Threshold only
            },

            'Precedence_R': {
                'class': PrecedenceRollbackAgent,
                'params': {'q_table_init': -1.0, 'alpha_phi': 0.01, 'lambda_precedence': 1.0, 'phi_init': 0.5, 'threshold': 3, 'K': 10}  # Threshold only
            },

            'FullModel': {
                'class': FullAgent,
                'params': {'q_table_init': -1.0, 'alpha_phi': 0.01, 'lambda_precedence': 1.0, 'phi_init': 0.5, 'penalty': 1.4, 'K': 10}  # EXACT existing params
            }
        },
        
        # Environment: Taxi-v3  
        'Taxi-v3': {
            'Baseline': {
                'class': QLearningAgent,
                'params': {}  # Uses default initialization (zeros after reset)
            },
            'RollbackOnly': {
                'class': RollBackAgent,
                'params': {'q_table_init': -1.0, 'threshold': 3}  # Match MOD params but no Phi
            },
            'PrecedenceOnly': {
                'class': PrecedenceAgent,
                'params': {'q_table_init': -1.0, 'K': 10, 'alpha_phi': 0.01, 
                          'lambda_precedence': 1.0, 'phi_init': 0.5}  # Phi only, no threshold
            },
            'ThresholdPeAgent': {
                'class': ThresholdPenaltyAgent,
                'params': {'q_table_init': -1.0, 'threshold': 3, 'penalty': 1.4}  # Threshold only
            },
            'Roll_Threshold': {
                'class': RollbackAndThresholdPenaltyAgent,
                'params': {'q_table_init': -1.0, 'threshold': 3, 'penalty': 1.4}  # Threshold only
            },

            'Precedence_Th': {
                'class': PrecedenceThresholdPenaltyAgent,
                'params': {'q_table_init': -1.0,'alpha_phi': 0.01, 'lambda_precedence': 1.0, 'phi_init': 0.5, 'threshold': 3, 'penalty': 1.4, 'K': 10}  # Threshold only
            },

            'Precedence_R': {
                'class': PrecedenceRollbackAgent,
                'params': {'q_table_init': -1.0, 'alpha_phi': 0.01, 'lambda_precedence': 1.0, 'phi_init': 0.5, 'threshold': 3, 'K': 10}  # Threshold only
            },

            'FullModel': {
                'class': FullAgent,
                'params': {'q_table_init': -1.0, 'alpha_phi': 0.01, 'lambda_precedence': 1.0, 'phi_init': 0.5, 'penalty': 1.4, 'K': 10}  # EXACT existing params
            }
        }
    }
    
    all_results = []
    
    # Run experiments on both environments
    for env_name in ['CliffWalking-v0', 'Taxi-v3']:
        print(f"\n{env_name}:")
        print("=" * 50)
        
        env_configs = ablation_configs[env_name]
        
        for agent_name, config in env_configs.items():
            try:
                result = run_exact_experiment(
                    env_name, 
                    config['class'],
                    config['params'],
                    agent_name,
                    episodes
                )
                all_results.append(result)
                
                print(f"✓ {agent_name:15}: Reward={result['mean_reward']:7.1f}±{result['std_reward']:5.1f}, "
                      f"Failures={result['mean_failures']:5.3f}, Rollbacks={result['mean_rollbacks']:5.1f}")
                      
            except Exception as e:
                print(f"✗ {agent_name:15}: ERROR - {str(e)}")
    
    # Save comprehensive results
    save_corrected_results(all_results, 'ablation_results.csv')
    
    print(f"\n{'='*80}")
    print("CORRECTED ABLATION STUDY COMPLETED")
    print(f"Results saved to ablation_results.csv")
    print("="*80)
    
    return all_results

def save_corrected_results(results, filename):
    """Save results with comprehensive metrics"""
    with open(filename, 'w') as f:
        f.write("Agent,Environment,MeanReward,StdReward,MeanSteps,StdSteps,")
        f.write("MeanFailures,StdFailures,MeanSuccesses,StdSuccesses,MeanRollbacks,StdRollbacks\n")
        
        for r in results:
            f.write(f"{r['agent']},{r['environment']},")
            f.write(f"{r['mean_reward']:.4f},{r['std_reward']:.4f},")
            f.write(f"{r['mean_steps']:.4f},{r['std_steps']:.4f},")
            f.write(f"{r['mean_failures']:.4f},{r['std_failures']:.4f},")
            f.write(f"{r['mean_successes']:.4f},{r['std_successes']:.4f},")
            f.write(f"{r['mean_rollbacks']:.4f},{r['std_rollbacks']:.4f}\n")

def analyze_corrected_results(results):
    """Analyze corrected results"""
    print("\nCORRECTED ABLATION ANALYSIS:")
    print("="*60)
    
    for env in ['CliffWalking-v0', 'Taxi-v3']:
        env_results = [r for r in results if r['environment'] == env]
        baseline = next((r for r in env_results if r['agent'] == 'Baseline'), None)
        
        if not baseline:
            continue
            
        print(f"\n{env} Results:")
        print("-"*40)
        print(f"{'Agent':<15} {'Reward':<12} {'Δ Reward':<10} {'Δ%':<8} {'Failures':<8} {'Rollbacks':<10}")
        print("-"*70)
        
        sorted_results = sorted(env_results, key=lambda x: x['mean_reward'], reverse=True)
        
        for r in sorted_results:
            reward_str = f"{r['mean_reward']:.1f}±{r['std_reward']:.1f}"
            if r['agent'] == 'Baseline':
                delta_str = "baseline"
                pct_str = "0%"
            else:
                delta = r['mean_reward'] - baseline['mean_reward']  
                pct = (delta / abs(baseline['mean_reward'])) * 100
                delta_str = f"{delta:+.1f}"
                pct_str = f"{pct:+.1f}%"
            
            failures_str = f"{r['mean_failures']:.3f}"
            rollbacks_str = f"{r['mean_rollbacks']:.1f}" if r['mean_rollbacks'] > 0 else "n/a"
            
            print(f"{r['agent']:<15} {reward_str:<12} {delta_str:<10} {pct_str:<8} {failures_str:<8} {rollbacks_str:<10}")

if __name__ == "__main__":
    print("CORRECTED Ablation Study")
    print("This matches your existing experimental setup EXACTLY")
    print()
    
    episodes = int(input("Enter number of episodes (1000 recommended): ") or "1000")
    
    try:
        results = run_corrected_ablation_study(episodes)
        analyze_corrected_results(results)
        
        print(f"\nFiles generated:")
        print(f"- ablation_results.csv (summary)")
        print(f"- ablation_CliffWalking-v0_*.csv (detailed logs)")
        print(f"- ablation_Taxi-v3_*.csv (detailed logs)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()