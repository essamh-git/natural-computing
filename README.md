# Natural Computing

Code for my Natural Computing wiki (University of Greenwich).

Install the requirements once:

    pip install numpy scipy scikit-image matplotlib pillow pandas ortools
    pip install --use-pep517 ypstruct

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

## Week 5: Genetic Algorithm

The real-coded GA from the Yarpiz course compared with DFO, then GAs applied to the Travelling Salesman Problem (TSP) and a card-splitting puzzle:

| Part | Task code | Figures |
|---|---|---|
| Part 1: GA vs DFO (30 dimensions, Sphere and Rastrigin) | `Natural_Camp_5/GA_vs_DFO.py` | `Natural_Camp_5/Week5Figures.py` |
| Part 2: GA vs TSP (tutorial vs improvements) | `Natural_Camp_5/TSPExperiments.py` | `Natural_Camp_5/TSPFigures.py`, `Natural_Camp_5/FigureMutationDiagram.py` |
| Part 3: TSP challenge (8 UK cities) | `Natural_Camp_5/TSPChallenge.py` | `Natural_Camp_5/TSPChallengeFigures.py` |
| Part 4: card game (generational vs steady-state GA) | `Natural_Camp_5/CardsExperiments.py` | `Natural_Camp_5/CardsFigures.py` |

Some files are imported by others, so keep them all in the same folder:

- `ga.py` (the Yarpiz course code, unchanged) is used by Part 1
- `TSPTutorial.py` and `TSPImproved.py` are used by Part 2
- `CardsGA.py` is used by Part 4

As in Weeks 3 and 4, run the task code first, then the figure script from the same folder. Part 1 takes about 20 minutes, Part 2 about 8 minutes, Part 3 a few seconds and Part 4 about 30 seconds.

## Credits

- The DFO algorithm (Weeks 3–5) is based on the original Python implementation by Mohammad Majid al-Rifaie ([github.com/mohmaj/DFO](https://github.com/mohmaj/DFO)), licensed under the GNU GPL.
- `ga.py` (Week 5) is from *Practical Genetic Algorithms in Python and MATLAB* by Mostapha Kalami Heris ([Yarpiz](https://yarpiz.com/632/ypga191215-practical-genetic-algorithms-in-python-and-matlab)).
- `tsp_tutorial.py` (Week 5) is from Eric Stoltz's genetic algorithm tutorial ([github.com/ezstoltz/genetic-algorithm](https://github.com/ezstoltz/genetic-algorithm)).
