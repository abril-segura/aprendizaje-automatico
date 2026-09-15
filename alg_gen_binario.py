import random

# Parámetros clave 
POPULATION_SIZE = 10  
GENES_LENGTH = 8       
MUTATION_RATE = 0.1    
GENERATIONS = 20       

# Inicialización
def create_individual():
    return [random.randint(0, 1) for _ in range(GENES_LENGTH)]

# Función de Fitness
def fitness(individual):
    return sum(individual)

# Seleccion (Por Torneo)
def selection(population):
    tournament = random.sample(population, 3)
    return max(tournament, key=fitness)

# Cruce: (Crossover)
def crossover(parent1, parent2):
    point = random.randint(1, GENES_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutación
def mutate(individual):
    for i in range(GENES_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual

def main():
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)
        best = population[0]
        
        print(f"Generación {generation}: Mejor = {best} (Fitness: {fitness(best)})")

        if fitness(best) == GENES_LENGTH:
            print("¡Solución óptima encontrada!")
            break

        new_population = [best] 

        while len(new_population) < POPULATION_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            c1, c2 = crossover(p1, p2)
            new_population.extend([mutate(c1), mutate(c2)])

        population = new_population[:POPULATION_SIZE]

if __name__ == "__main__":
    main()