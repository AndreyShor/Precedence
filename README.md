# Precedence

Experiments and ablation studies for reversible and precedence-aware reinforcement learning. This repository includes:
- Basic, runnable simulations for classic Gym environments.
- The full code used to generate all data for the associated paper’s experiments (ablation studies).
- Core algorithm implementations and utilities.

If you used this repo before: the project has been significantly refactored. Folder layout, entry points, and instructions have changed.

## Contents

- [Simulations](https://github.com/AndreyShor/Precedence/tree/main/Simulations)
  - Basic examples for classic control/toy-text environments, including Taxi-v3 and CliffWalking-v0.
- [Abliation](https://github.com/AndreyShor/Precedence/tree/main/Abliation)
  - All data used in the paper was generated from this folder’s code.
  - Experiment setups are defined in `ablation_study.py`.
- [Algorithm](https://github.com/AndreyShor/Precedence/tree/main/Algorithm)
  - Core RL algorithm implementations and training logic.
- [Data](https://github.com/AndreyShor/Precedence/tree/main/Data)
  - Outputs (metrics, logs, results) produced by simulations and ablation runs.
- [logger.py](https://github.com/AndreyShor/Precedence/blob/main/logger.py)
  - Lightweight logging utilities used across experiments.
- Paper and diagram:
  - [Reversibility in Learning (PDF)](https://github.com/AndreyShor/Precedence/blob/main/Reversibility_in_Learning_FULL.pdf)
  - [Algorithm diagram](https://github.com/AndreyShor/Precedence/blob/main/reversible_algorithm_diagram.png)

## Quick start

Prerequisites
- Python 3.x
- Typical dependencies include Gym/Gymnasium and NumPy (and optionally plotting libraries). Install packages you see imported in the scripts you plan to run.

Set up
- Clone the repo and create a virtual environment as preferred.
- Install dependencies accordingly (e.g., `pip install gymnasium numpy`), or follow imports in the scripts if your environment differs.

## Running basic simulations

You can find runnable examples in the [Simulations](https://github.com/AndreyShor/Precedence/tree/main/Simulations) folder for:
- Taxi-v3
- CliffWalking-v0

To run, execute the desired script directly, for example:
```
python Simulations/<your_simulation_script>.py
```
Open the script to tweak environment parameters (episodes, max steps, seeds) and algorithm settings.

## Reproducing paper data (Ablation studies)

All data used in the paper was generated from the code in the [Abliation](https://github.com/AndreyShor/Precedence/tree/main/Abliation) folder. Experiment setups and configurations live in:
- `Abliation/ablation_study.py`

To launch ablation runs:
```
python Abliation/ablation_study.py
```
Adjust configuration in `ablation_study.py` to select environments, algorithms, hyperparameters, and seeds. Outputs are written to the [Data](https://github.com/AndreyShor/Precedence/tree/main/Data) directory (see script and logger for exact paths and file names).

## Project structure

- Algorithm/ — RL algorithm implementations and training logic used by simulations and ablation runs.
- Simulations/ — Minimal examples to get started; see Taxi-v3 and CliffWalking-v0.
- Abliation/ — Scripts and experiment definitions used to generate all paper results; main entry point is `ablation_study.py`.
- Data/ — Collected results and logs from runs.
- logger.py — Shared logging helper.

## Notes and tips

- Start with the Simulations folder for quick sanity checks on your environment and setup.
- Use the Ablation entry point to run batched studies and reproduce paper-level figures/data.
- If you add new environments or algorithms, keep implementations under `Algorithm/` and create a corresponding script in `Simulations/` or a config in `Abliation/ablation_study.py`.

## Citation

If you use this repository or its results in your research, please cite the associated paper. A formal citation entry can be added here if needed.

## License

Copyright (c) 2025 Andrejs Sorstkins

All rights reserved.

Permission to use, copy, modify, or distribute this software (in whole or in part) 
is granted only with the express prior written consent of the copyright holder.

Unauthorized use is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.