# Natural Computing

Code for my Natural Computing wiki (University of Greenwich).

Install the requirements once:

    pip install numpy matplotlib

## Week 1: No Free Lunch Theorem

Tests the theorem on every possible problem in a small search space:

    python week-1/nfl_experiment.py

## Week 2: Sphere Function

Random solutions for the Sphere and Rastrigin functions, then a hill climber to improve them:

| Part | Task code | Figures |
|---|---|---|
| Part 1: one random solution | `week-2/sphere_part1.py` | `week-2/figures_part1.py` |
| Part 2: 50 random solutions | `week-2/sphere_part2.py` | `week-2/figures_part2.py` |
| Part 3: Rastrigin function | `week-2/sphere_part3.py` | `week-2/figures_part3.py` |
| Advanced: hill climber vs random search | `week-2/enhance.py` | (same file) |
| Mini-project encoding diagram | — | `week-2/figure_encodings.py` |

Run any file with `python <path>`, for example:

    python week-2/enhance.py
