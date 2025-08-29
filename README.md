# Precedence

A lightweight framework for experimenting with learning algorithms and environments.

- Algorithm implementations live in `Algorithm/Brain.py`.
- Each simulation in `Simulations/` is a single self-contained file that defines both the environment and how an algorithm is applied to it.
- Results and analysis live in `Data/`, including an R Markdown report `precedence_sim.Rmd`.
- `logger.py` contains logging utilities tailored for the Taxi and CliffWalking environments.
- `gym-sokoban-default/` contains a Sokoban implementation for Gym; you can try it in manual mode by running `env_human.py` (see notes below).

This layout makes it easy to:
- Reuse the same algorithm across different environments.
- Package a full experiment in a single, runnable Python file.

## Repository structure

```
.
├── Algorithm/
│   └── Brain.py                 # Algorithm implementations (agents, policies, etc.)
├── Data/
│   └── precedence_sim.Rmd       # Results and analysis of experiments
├── Simulations/
│   ├── <YourSimulationA>.py     # Environment + algorithm wiring
│   └── <YourSimulationB>.py     # Environment + algorithm wiring
├── gym-sokoban-default/
│   ├── env_human.py             # Run Sokoban in manual/play mode
│   └── ...                      # Sokoban Gym environment implementation
└── logger.py                    # Logger logic for Taxi and CliffWalking environments
```

## Quick start

1. Ensure you have Python 3 installed.
2. Pick any simulation in `Simulations/` and run it:
   ```
   python Simulations/<YourSimulation>.py
   ```
3. Tune your experiment by editing the simulation file:
   - Look for the environment configuration where the variables are defined:
     - `episodes` — number of training/evaluation episodes to run.
     - `max_steps` — maximum number of steps per episode.
   - These variables live in the “environment” portion of each simulation file and control the run length and horizon.

Random seeding:
- By default, each episode uses a new random seed: 1, 2, ..., N (where N is the number of episodes).

## Short-term vs. long-term learning

In each simulation’s training loop you’ll find a call like:

```python
for ep in range(episodes):
    agent.reset()  # controls memory reset between episodes
    # ... run episode up to `max_steps`
```

- Short-term learning (default): keep `agent.reset()`.
  - The agent’s memory is reset at the start of every episode.
  - Useful for episodic tasks where you want each episode to start fresh.

- Long-term convergence testing: remove `agent.reset()`.
  - The agent’s memory persists across episodes.
  - Useful to study convergence and long-horizon behavior.

Choose the mode that matches your experimental goal. Do not delete the method from the agent; only remove or keep the call in the training loop.

## Implemented and in-progress algorithms

Implemented and ready for testing:
- Q-Learning
- SARSA
- Expected SARSA
- Double Q-Learning
- Q-Learning modified with Rollback only
- Q-Learning modified with Rollback + Precedence

In progress:
- SARSA modified with Rollback only — nearly done, but I suspect there is a bug somewhere.
- SARSA modified with Rollback + Precedence — nearly there.

## Using and extending algorithms

- Existing algorithms are implemented in `Algorithm/Brain.py`.
- To use an algorithm in a simulation:
  - Import the desired class/function from `Algorithm.Brain`.
  - Instantiate it in your simulation and connect it to the environment loop.

- To add a new algorithm:
  1. Implement it in `Algorithm/Brain.py` (or add a new module if preferred).
  2. Import and wire it in a simulation under `Simulations/`.

## Creating a new simulation

1. Copy an existing file from `Simulations/` as a starting point.
2. Define your environment dynamics and reward/termination conditions.
3. Set `episodes` and `max_steps` in the environment config section.
4. Import an agent from `Algorithm/Brain.py` and connect it in the run loop.
5. Decide whether to keep or remove `agent.reset()` depending on short-term vs. long-term learning (see above).

## Data and analysis

- `Data/` contains experiment outputs and analyses.
- `Data/precedence_sim.Rmd` includes an R Markdown report with analysis of results.

## Logging

- `logger.py` implements logging utilities specifically for:
  - Taxi
  - CliffWalking

Use these helpers in your simulations to capture episode returns, step counts, and other metrics.

## Sokoban environment

- `gym-sokoban-default/` provides a Sokoban implementation compatible with Gym.
- You can try it in manual/play mode:
  ```
  python gym-sokoban-default/env_human.py
  ```
- Note: A Sokoban simulation using the tabular algorithm is not included due to high memory requirements, which are problematic for tabular methods.

## Tips

- Reproducibility: consider setting additional random seeds (e.g., NumPy, Python’s `random`, PyTorch) at the top of each simulation file if you need fully deterministic runs.
- Logging: print or log episode returns, success rates, or losses to monitor learning progress.
- Runtime: start with small `episodes` and `max_steps` to validate logic, then scale up.

## License


## Citation


## Questions or issues

If you find a bug or have a feature request, please open an issue in this repository with:
- The simulation you ran
- Your environment settings (`episodes`, `max_steps`)
- A short description of the observed behavior and expected outcome