import numpy as np
import matplotlib.pyplot as plt
    
class GeneticAlgorithm:
    
    def __init__(self):
        """
        """
        
        self.population = None
        self.v_min = None
        self.v_max = None
        self.n = None
        self.m = None
        self.k = None
        self.objective_output = None
        self.fitness_sum = None
        self.probability = None
        self.best_solution = None
        self.pool = None
        self.max_prob = []
        self.best_output = []
    
    def init_population(self, n, m, v_min, v_max):
        """
        """
        
        self.v_min = v_min
        self.v_max = v_max
        self.n = n
        self.m = m
        self.population = np.random.randint(v_min, v_max, (n, m))
    
    def fitness(self, obj_func):
        """
        """
        
        self.objective_output = np.apply_along_axis(obj_func, axis=1, arr=self.population)
        self.fitness_value = 1 / (1 + self.objective_output)
        self.fitness_sum = self.fitness_value.sum()
        self.probability = self.fitness_value / self.fitness_sum
        
        self.max_prob.append(np.argmax(self.probability))
        self.best_solution = self.population[self.max_prob[-1]]
        self.best_output.append(self.objective_output[self.max_prob[-1]])
    
    def selection(self, crossover_rate):
        """
        """
        
        size = int(crossover_rate * self.n)
        i = np.random.choice(self.n, size=size, replace=False, p=self.probability)
        self.pool = self.population[i]
        self.k = self.pool.shape[0]
    
    def mating(self):
        """
        """
        
        cross_points = np.random.randint(1, self.m, self.k)
        children = []
        for c in range(self.k):
            x = c
            y = (c+1) % self.k
            i = cross_points[c]
            children.append(np.concatenate((self.pool[x][:i], self.pool[y][i:])))
        
        self.population = np.concatenate(([self.best_solution], self.pool, np.array(children)))
        self.n = self.population.shape[0]
    
    def mutation(self, mutation_rate):
        """
        Ver depois:
        - mutação apenas no pool?
        """
        
        total_genes = self.n * self.m
        num_mutations = int(mutation_rate * total_genes)
        target_mutations = np.random.randint(0, total_genes, num_mutations)
        
        for t in target_mutations:
            i = t // self.m
            j = t % self.m
            
            self.population[i][j] = np.random.randint(self.v_min, self.v_max+1)
    
    def evolve(self, max_iter, target, n, m, v_min, v_max, obj_func, crossover_rate, mutation_rate):
        """
        """
        
        self.init_population(n, m, v_min, v_max)
        
        for i in range(max_iter):
            self.fitness(obj_func)
            
            print(f"{i+1} : {self.best_solution} = {self.best_output[-1]}")
            if self.best_output[-1] == target:
                break
            
            self.selection(crossover_rate)
            self.mating()
            self.mutation(mutation_rate)
    
    def plot_evolution(self):
        """
        """
        
        gens = list(range(len(self.best_output)))
        plt.figure(figsize=(12, 6))
        plt.plot(gens, self.best_output, marker="o", label="Função objetivo")
        plt.title("Evolução")
        plt.xlabel("Geração")
        plt.ylabel("Objetivo")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    v_min = 0
    v_max = 30
    crossover_rate = 0.5
    mutation_rate = 0.1
    max_iter = 100
    target = 0
    n = 100
    m = 4
    obj_func = lambda a : abs((a[0] + 2*a[1] + 3*a[2] + 4*a[3]) - 30)
    
    g = GeneticAlgorithm()
    g.evolve(max_iter, target, n, m, v_min, v_max, obj_func, crossover_rate, mutation_rate)
    g.plot_evolution()
    
