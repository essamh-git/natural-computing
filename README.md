# Natural Computing

Code for my Natural Computing wiki (University of Greenwich).

Install the requirements once:

    pip install numpy matplotlib

## Week 1: No Free Lunch Theorem

Tests the theorem on every possible problem in a small search space:

    python Natural_Camp_1/nfl_experiment.py

## Week 2: Sphere Function

Random solutions for the Sphere and Rastrigin functions, then a hill climber to improve them:

| Part | Task code | Figures |
|---|---|---|
| Part 1: one random solution | `Natural_Camp_2/ToyProblemSphereFunction.py` | `Natural_Camp_2/Part1Figures.py` |
| Part 2: 50 random solutions | `Natural_Camp_2/ToyProblemSphereFunction.py` | `Natural_Camp_2/Part2Figures.py` |
| Part 3: Rastrigin function | `Natural_Camp_2/ToyProblemAnotherFunction.py` | `Natural_Camp_2/Part3Figures.py` |
| Advanced: hill climber vs random search | `Natural_Camp_2/EnhancedSolution.py` | (same file) |
| Mini-project encoding diagram | — | `Natural_Camp_2/FigureEncoding.py` |

Run any file with `python <path>`, for example:

    python Natural_Camp_2/EnhancedSolution.py

## Week 3: Dispersive Flies Optimisation

DFO on the Sphere and Rastrigin functions: one position update by hand, 30 runs of the algorithm, and a study of dimensionality and the disturbance threshold:

| Part | Task code | Figures |
|---|---|---|
| Part 1: position update by hand | `Natural_Camp_3/DFOHandExercise.py` | (same file) |
| Part 2: 30 runs on Sphere and Rastrigin | `Natural_Camp_3/DFOExperiment.py` | `Natural_Camp_3/DFOFigures.py` |
| Part 3: dimensionality and disturbance threshold | `Natural_Camp_3/DFOParameterStudy.py` | `Natural_Camp_3/DFOParameterFigures.py` |

For Parts 2 and 3, run the task code first, because it saves its results to a `.npz` file. Then run the figure script from the same folder:

    python Natural_Camp_3/DFOExperiment.py
    python Natural_Camp_3/DFOFigures.py

Part 2 takes about 3 minutes and Part 3 about 15 minutes.

## Week 4: Finding the Brightest Area with DFO

DFO adapted to search an image: each fly is a pixel position (x, y), and its fitness is the brightness around it. Covers finding the brightest area, other images, switching images half-way, a moving object, a fading and emerging object, and two bright spots:

| File | Purpose |
|---|---|
| `week-4/DFOImage.py` | DFO for images, plus helpers for making and loading test images (imported by the other two files) |
| `week-4/Week4Experiments.py` | Runs every task and saves the results to `week4_results.npz` |
| `week-4/Week4Figures.py` | Draws the figures and the moving-object animation from the saved results |

Keep all three files in the same folder, then run:

    python week-4/Week4Experiments.py
    python week-4/Week4Figures.py

The experiments take about 1–2 minutes.

## Credits

The DFO algorithm is based on the original Python implementation by Mohammad Majid al-Rifaie ([github.com/mohmaj/DFO](https://github.com/mohmaj/DFO)), licensed under the GNU GPL.


