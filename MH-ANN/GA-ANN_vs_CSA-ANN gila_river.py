import numpy as np
import random as r
import pandas as pd
import pygad
import pygad.nn
import pygad.gann
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

seed = 42
r.seed(seed)
np.random.seed(seed)

# Sample Data
def getData(num):
    if num == 1:
       
        data = pd.read_csv("C:/Users/apsto/OneDrive - Southern Connecticut State University/2025 Summer Research/Datasets/gilaRiver.csv")
        y = np.asarray(data['y']).reshape(-1, 1)
        X = np.asarray(data.drop(columns = ['y']))

        activation = "relu"
        return X, y, activation

data_inputs, data_outputs, activation = getData(1)
scaler = StandardScaler()
data_inputs, test_inputs, data_outputs, test_outputs = tts(data_inputs, data_outputs, test_size = .2, random_state = seed)
num_inputs = data_inputs.shape[1]

data_inputs = scaler.fit_transform(data_inputs)
test_inputs = scaler.transform(test_inputs)

"""
x_scaler = StandardScaler()
y_scaler = StandardScaler()
data_inputs = x_scaler.fit_transform(data_inputs)
test_inputs = x_scaler.transform(test_inputs)
data_outputs = y_scaler.fit_transform(data_outputs.reshape(-1, 1)).flatten()
test_outputs = y_scaler.fit_transform(test_outputs.reshape(-1, 1)).flatten()
"""

def MSE(true, pred):
    return np.mean((pred - true) ** 2)

def MAE(true, pred):
    return np.mean(np.abs(pred - true))

def VAF(true, pred):
    return (1 - (np.var(pred - true) / np.var(true))) * 100

# Functions for Both
def update_network_weights(pop_vecs):
    """Decode vectors -> weight matrices and push into gann networks."""
    pop_mats = pygad.gann.population_as_matrices(
        population_networks=gann.population_networks, population_vectors=pop_vecs
    )
    gann.update_population_trained_weights(population_trained_weights=pop_mats)

def evaluate_population(pop_vecs):
    """Return fitness array for every solution in pop_vecs."""
    update_network_weights(pop_vecs)
    fitness = []
    for ind in range(len(pop_vecs)):
        preds = pygad.nn.predict(
            last_layer=gann.population_networks[ind],
            data_inputs=data_inputs,
            problem_type="regression",
        )
        fitness.append(1.0 / np.mean(np.abs(preds - data_outputs) ** 2))
    return np.asarray(fitness)

# Functions for GA
def single_point_crossover(parents, offspring_size):
    offspring = []
    while len(offspring) < offspring_size:
        p1, p2 = parents[np.random.choice(parents.shape[0], 2, replace=False)]
        point = np.random.randint(1, p1.size) if p1.size > 1 else 0
        c1 = np.concatenate((p1[:point], p2[point:]))
        c2 = np.concatenate((p2[:point], p1[point:]))
        offspring.extend([c1, c2])
    return np.asarray(offspring[:offspring_size])

def mutate(pop, percent, low, high):
    for sol in pop:
        n_mut = max(1, int(sol.size * percent / 100))
        idx = np.random.choice(sol.size, n_mut, replace=False)
        sol[idx] = np.random.uniform(low, high, n_mut)
    return pop

# Genetic Algorithm
def GA(population, num_generations, num_parents_mating, mutation_percent_genes, init_range_low, init_range_high, keep_parents):
    print('starting GA')
    fitness = evaluate_population(population)
    best_solution = population[fitness.argmax()].copy()
    best_fitness = -np.inf
    best_generation = -1
    fitness_history = []

    for gen in range(num_generations):
        if (gen % 10) == 0:
            print(f"gen: {gen}")
        fitness = evaluate_population(population)
        fitness_history.append(fitness.max())

        # Record best solution so far
        if fitness.max() > best_fitness:
            best_fitness = fitness.max()
            best_solution = population[fitness.argmax()].copy()
            best_generation = gen

        # Parent selection (select top‑k)
        parents_idx = fitness.argsort()[-num_parents_mating:]
        parents = population[parents_idx]

        # Crossover + mutation to form new offspring
        offspring_size = population.shape[0] - keep_parents
        offspring = single_point_crossover(parents, offspring_size)
        offspring = mutate(offspring, mutation_percent_genes, init_range_low, init_range_high)

        # Elitism + next generation
        if keep_parents > 0:
            elite_idx = fitness.argsort()[-keep_parents:]
            elites = population[elite_idx]
            population = np.vstack((elites, offspring))
        else:
            population = offspring

    update_network_weights(np.vstack((best_solution, population[1:])))
    best_network = gann.population_networks[0]

    return fitness_history, best_network, best_fitness, best_generation

# Crow Search Algorithm
def CSA(population, num_generations, flight_length, awareness_probability, init_range_low, init_range_high):
    print('starting CSA')
    dim = population.shape[1]
    memory = population.copy()  # each crow remembers its best so far

    fitness = evaluate_population(population)
    best_idx = fitness.argmax()
    best_network, best_fit, best_generation = population[best_idx].copy(), fitness[best_idx], 0

    history = [best_fit]

    for gen in range(num_generations):
        
        if (gen % 10) == 0:
            print(f"gen: {gen}")
        new_nests = population.copy()
        for i in range(num_solutions):
            j = np.random.randint(num_solutions)  # random crow to follow
            if np.random.rand() > awareness_probability:
                step = flight_length * np.random.rand(dim) * (memory[j] - population[i])
                new_pos = population[i] + step
            else:
                new_pos = np.random.uniform(init_range_low, init_range_high, dim)

            new_nests[i] = np.clip(new_pos, init_range_low, init_range_high)

        # evaluate new positions
        new_fit = evaluate_population(new_nests)

        # greedy selection & memory update
        memory_fitness = evaluate_population(memory)
        for i in range(num_solutions):
            if new_fit[i] > fitness[i]:
                population[i] = new_nests[i]
                fitness[i] = new_fit[i]
            if fitness[i] > memory_fitness[i]:
                memory[i] = population[i]

        # global best update
        best_idx = fitness.argmax()
        if fitness[best_idx] > best_fit:
            best_fit = fitness[best_idx]
            best_network = population[best_idx].copy()
            best_generation = gen

        history.append(best_fit)

    # finalise network with best weights
    population[0] = best_network

    update_network_weights(np.vstack((best_network, population[1:])))
    best_network = gann.population_networks[0]
    
    return history, best_network, best_fit, best_generation
    
num_generations = 500
num_neurons = 16
num_outputs = 1
num_solutions = 20
num_layers = 3

init_range_low, init_range_high = -2, 2

#==========================================================================
# Genetic Algorithm

# ANN Parameters (Individual Parameters)
gann = pygad.gann.GANN(
    num_solutions=num_solutions,
    num_neurons_input=num_inputs,
    num_neurons_hidden_layers = [num_neurons] * num_layers,
    num_neurons_output=num_outputs,
    hidden_activations=[activation] * num_layers,
    output_activation="None",
)

population_vectors = pygad.gann.population_as_vectors(
    population_networks=gann.population_networks
)
# Represent all the networks as vectors
population = np.array(population_vectors.copy())

GA_history, GA_best, GA_best_fitness, GA_best_generation = GA(population,
                                                    num_generations = num_generations,
                                                    num_parents_mating = 4, 
                                                    mutation_percent_genes = 5,  #%
                                                    init_range_low = init_range_low,
                                                    init_range_high = init_range_high,
                                                    keep_parents = 1 )
GA_predictions = pygad.nn.predict(
    last_layer = GA_best, data_inputs=data_inputs, problem_type="regression"
)
GA_testing = pygad.nn.predict(
    last_layer = GA_best, data_inputs=test_inputs, problem_type="regression"
)
#==========================================================================
# Crow Search Algorithm

# ANN Parameters (Individual Parameters)
gann = pygad.gann.GANN(
    num_solutions=num_solutions,
    num_neurons_input=num_inputs,
    num_neurons_hidden_layers=[num_neurons] * num_layers,
    num_neurons_output=num_outputs,
    hidden_activations=[activation] * num_layers,
    output_activation="None",
)

population_vectors = pygad.gann.population_as_vectors(
    population_networks=gann.population_networks
)
# Represent all the networks as vectors
population = np.array(population_vectors.copy())

CSA_history, CSA_best, CSA_best_fitness, CSA_best_generation = CSA(population,
                                                    num_generations = num_generations,
                                                    flight_length = 2.0,
                                                    awareness_probability = .25,
                                                    init_range_low = init_range_low,
                                                    init_range_high = init_range_high)

CSA_predictions = pygad.nn.predict(
    last_layer = CSA_best, data_inputs=data_inputs, problem_type="regression"
)
CSA_testing = pygad.nn.predict(
    last_layer = CSA_best, data_inputs=test_inputs, problem_type="regression"
)

#=====================================================================
#Plotting

print("GA PERFORMANCE")
print(f"   TRAINING:   MSE: {MSE(data_outputs, GA_predictions)} | MAE: {MAE(data_outputs, GA_predictions)} | VAF: {VAF(data_outputs, GA_predictions)}")
print(f"   TESTING:   MSE: {MSE(test_outputs, GA_testing)} | MAE: {MAE(test_outputs, GA_testing)} | VAF: {VAF(test_outputs, GA_testing)}")

print("CSA PERFORMANCE")
print(f"   TRAINING:   MSE: {MSE(data_outputs, CSA_predictions)} | MAE: {MAE(data_outputs, CSA_predictions)} | VAF: {VAF(data_outputs, CSA_predictions)}")
print(f"   TESTING:   MSE: {MSE(test_outputs, CSA_testing)} | MAE: {MAE(test_outputs, CSA_testing)} | VAF: {VAF(test_outputs, CSA_testing)}")


plt.figure(figsize=(6, 3))
plt.plot([x for x in range(num_generations)], GA_history, label = "Convergence of GA", color = 'orange')
plt.plot([x for x in range(num_generations+1)], CSA_history, label = "Convergence of CSA", color = 'green')
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Evolution of Fitness")
plt.legend()
plt.tight_layout()
plt.show()

true_train = data_outputs.flatten()
fig = plt.figure(figsize = (8, 5))
plt.title("Target vs. Predicted Training Case")
plt.plot(range(len(true_train)), true_train, label = "Target", color = 'blue')
plt.plot(range(len(GA_predictions)), GA_predictions, label = "GA Predicted", color = 'green')
plt.plot(range(len(CSA_predictions)), CSA_predictions, label = "CSA Predicted", color = 'orange')
plt.legend()
plt.show()

true_test = test_outputs.flatten()
fig = plt.figure(figsize = (8, 5))
plt.title("Target vs. Predicted Testing Case")
plt.plot(range(len(true_test)), true_test, label = "Target", color = 'blue')
plt.plot(range(len(GA_testing)), GA_testing, label = "GA Predicted", color = 'green')
plt.plot(range(len(CSA_testing)), CSA_testing, label = "CSA Predicted", color = 'orange')
plt.legend()
plt.show()

