import math
import random

def new_population(n, m, v_min, v_max):
    """
    """
    
    return [[int(random.uniform(v_min, v_max)) for j in range(m)] for i in range(n)]

def fitness(population, objective_func):
    """
    """
    
    objective_output = [objective_func(*p) for p in population]
    fitness_value = [1 / (1 + obj) for obj in objective_output]
    fitness_sum = sum(fitness_value)
    probability = [fit / fitness_sum for fit in fitness_value]
    
    return probability

def selection(population, probability):
    """
    """
    
    return random.choices(population, weights=probability, k=len(population))

def mating(population, crossover_rate):
    """
    """
    
    n = len(population)
    m = len(population[0])
    
    parents = []
    new_population = []
    for k in range(n):
        if random.random() < crossover_rate:
            parents.append(population[k])
        else:
            new_population.append(population[k])
    
    j = len(parents)
    cross_points = [random.randint(1, m-1) for x in range(j)]
    children = []
    for c in range(j):
        x = c
        y = (c+1) % j
        i = cross_points[c]
        children.append(parents[x][:i] + parents[y][i:])
    
    new_population.extend(children)
    
    return new_population

def mutation(population, mutation_rate, v_min, v_max):
    """
    """
    
    n = len(population)
    m = len(population[0])
    total_genes = n * m
    num_mutations = math.floor(mutation_rate * total_genes)
    target_mutations = [random.randint(0, total_genes-1) for x in range(num_mutations)]
    
    for t in target_mutations:
        i = t // m
        j = t % m
        
        val = population[i][j]
        val = random.randint(v_min, v_max) # MUDAR AQUI
        population[i][j] = val
    
    return population

def get_max(population, probability):
    """
    """
    
    return population[probability.index(max(probability))]

if __name__ == "__main__":
    v_min = 0
    v_max = 30
    objective_func = lambda a, b, c, d : abs((a + 2*b + 3*c + 4*d) - 30)
    crossover_rate = 0.25
    mutation_rate = 0.1
    max_iter = 100
    target = 0
    
    population = new_population(100, 4, v_min=v_min, v_max=v_max)
    
    for i in range(max_iter):
        probability = fitness(population, objective_func)
        
        best_solution = get_max(population, probability)
        best_output = objective_func(*best_solution)
        print(f"{i+1} : {best_solution} = {best_output}")
        
        if best_output == target:
            break
        
        population = selection(population, probability)
        population = mating(population, crossover_rate)
        population = mutation(population, mutation_rate, v_min, v_max)
    