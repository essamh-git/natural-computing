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
| Part 1: one random solution | `week-2/ToyProblemSphereFunction.py` | `week-2/Part1Figures.py` |
| Part 2: 50 random solutions | `week-2/ToyProblemSphereFunction.py` | `week-2/Part2Figures.py` |
| Part 3: Rastrigin function | `week-2/ToyProblemAnotherFunction.py` | `week-2/Part3Figures.py` |
| Advanced: hill climber vs random search | `week-2/EnhancedSolution.py` | (same file) |
| Mini-project encoding diagram | — | `week-2/FigureEncoding.py` |

Run any file with `python <path>`, for example:

    python week-2/EnhancedSolution.py
