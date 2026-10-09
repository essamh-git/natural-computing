import numpy as np
import math as m
from random import random, seed

seed(42)   # FIXED SEED SO THE RESULTS CAN BE REPRODUCED

# FITNESS FUNCTION
def fitness_function( func_no, x ):
    global FEs

    # SPHERE FUNCTION
    if func_no == 1:
        sum = 0.0
        for i in range( len(x) ):
            sum = sum + m.pow(x[i],2)

    # RASTRIGIN FUNCTION
    elif func_no == 2:
        sum = 10.0 * len(x)
        for i in range( len(x) ):
            sum = sum + m.pow(x[i],2) - 10.0 * m.cos(2 * m.pi * x[i])

    FEs = FEs + 1
    return sum


FEs = 0    # COUNTING THE NUMBER OF TIMES FITNESS FUNCTION IS CALLED
D   = 30   # ASSUME THE DIMENSIONALITY OF THE PROBLEM TO BE 30


# CONSTRAINTS
lower_bound = -5.12
upper_bound =  5.12


# GENERATE RANDOM SOLUTION
x = [None] * D     # A SOLUTION

# INITIALISE THE VALUES FOR x
# EACH VALUE IS DRAWN UNIFORMLY FROM [lower_bound, upper_bound]
for i in range(D):
    x[i] = lower_bound + random() * (upper_bound - lower_bound)

# DISPLAY THE SAMPLE SOLUTION
print ( 'x: \n', np.round(x, 4).tolist(), '\n')


# EVALUATE AND DISPLAY THE SAMPLE SOLUTON
f_x = fitness_function(1, x)
print('Fitness of x: ', round(f_x, 4), '\n')


# GENERATE 50 SOLUTIONS AND FIND THE BEST
N = 50           # NUMBER OF SOLUTIONS
X = [[lower_bound + random() * (upper_bound - lower_bound) for j in range(D)]
     for i in range(N)]      # 2D LIST HOLDING ALL N SOLUTIONS (N x D)


# EVALUATE THE SAMPLE SOLUTON
fitness = [None]*N  # FITNESS VALUES

for i in range(N):
    fitness[i] = fitness_function(1, X[i])


# FIND THE INDEX OF BEST SOLUTION
best_index = int(np.argmin(fitness))   # SPHERE IS MINIMISED, SO LOWEST IS BEST


# DISPLAY ALL FITNESS VALUES, THE BEST SOLUTION, AND THE BEST FITNESS
print('Fitness values for all solutions: \n', np.round(fitness, 2).tolist(), '\n')
print('Best solution: \n', np.round(X[best_index], 4).tolist(), '\n')
print('Best fitness value: ', round(fitness[best_index], 4))
print('Worst fitness value:', round(max(fitness), 4))
print('Mean fitness value: ', round(float(np.mean(fitness)), 4))
print('Function evaluations used (FEs):', FEs)


# PART 3: EVALUATE THE SAME 50 SOLUTIONS WITH RASTRIGIN
fitness_r = [None]*N  # RASTRIGIN FITNESS VALUES

for i in range(N):
    fitness_r[i] = fitness_function(2, X[i])

best_index_r = int(np.argmin(fitness_r))

print('\n----- RASTRIGIN FUNCTION -----')
print('Fitness values for all solutions: \n', np.round(fitness_r, 2).tolist(), '\n')
print('Best solution: \n', np.round(X[best_index_r], 4).tolist(), '\n')
print('Best fitness value: ', round(fitness_r[best_index_r], 4))
print('Worst fitness value:', round(max(fitness_r), 4))
print('Mean fitness value: ', round(float(np.mean(fitness_r)), 4))
print('Best solution index (Sphere vs Rastrigin):', best_index + 1, 'vs', best_index_r + 1)
print('Function evaluations used (FEs):', FEs)