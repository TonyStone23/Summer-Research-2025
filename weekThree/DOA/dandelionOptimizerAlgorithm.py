import numpy as np
import random as r
import statistics as s
import matplotlib.pyplot as plt

def f(x):
    #return 1/2 * x**2 + 3
    return np.cos(x**2) - x + (x - 1)**2

def initialize(pop):
    return [r.uniform(-10, 10) for individual in range(pop)]
    
def repopulate(dandelions, pop):
    return [dandelions[0] for individual in range(pop)]

def sunnyDay(dandelions, t):
    return [D + (1/t) * (r.uniform(-10, 10) - D) for D in dandelions]

def rainyDay(dandelions, t, T):
    p = (t**2 - 2*t + 1)/(T**2 - 2*T + 1) + 1
    return [D * (1-r.random() * p) for D in dandelions]

def DOA(pop, T, seed = 42, plot = True):
    """
    pop: population size
    T: number of iterations
    seed: seed (42 by default)
    plot: visualize Dandilion Optimization
    """

    # Set up plot
    fig = plt.figure(figsize = (7, 5))
    x = np.linspace(-5, 5, 200)
    y = f(x)
    plt.plot(x, y, color = 'black')

    r.seed(seed)

    # Initialize the population
    dandelions = initialize(pop)
    # Evaluate Fitness
    fitness = [f(D) for D in dandelions]
    zipped = sorted(zip(fitness, dandelions), key=lambda x: x[0])
    fitness, dandelions = zip(*zipped)
    fitness, dandelions = list(fitness), list(dandelions)
    avD = s.mean(dandelions)
    avF = f(avD)
    avFs = []

    # Main loop
    t = 0
    while t <= T:
        # Output
        print(f"  Generation {t}")
        print(f"    D(best) : {dandelions[0]:.3f} | D(ave) : {avD:.3f} | F(best) : {fitness[0]:.3f} | F(ave) : {avF:.3f}")
        plt.scatter(dandelions[0], fitness[0], label = f"Best of Gen. {t}")

        # Randomize weather
        weather = r.random()

        # increment
        t += 1

        # Ascention
        if weather <.5:
            dandelionHeight = sunnyDay(dandelions, t)
        else:
            dandelionHeight = rainyDay(dandelions, t, T)
        
        # Descend
        for d in range(len(dandelions)):
            D = dandelions[d]
            a = dandelionHeight[d]
            b = r.normalvariate(0,1)
            mean = s.mean(dandelions)
            dandelions[d] = D - a * b * (mean - a * b * D)

        fitness = [f(D) for D in dandelions]
        avD = s.mean(dandelions)
        avF = f(avD)
        avFs.append(avF)
        zipped = sorted(zip(fitness, dandelions), key=lambda x: x[0])
        fitness, dandelions = zip(*zipped)
        fitness, dandelions = list(fitness), list(dandelions)
        dandelions = repopulate(dandelions, pop)

    if plot is True:
        plt.title('Best Dandelions')
        fig.text(.05, .9, f"number of Generations: {T}\nPopulation size: {pop}")
        plt.xlabel('x')
        plt.ylabel('Fitness')
        plt.show()

        plt.plot([t for t in range(T+1)], avFs, color = 'black')
        plt.title("Performance")
        plt.xlabel("Generations")
        plt.ylabel("Average Fitness")
        plt.show()

    return

DOA(pop = 20, T = 10)
