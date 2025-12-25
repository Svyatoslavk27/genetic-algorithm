import random
import copy
import numpy as np
# Запаси (постачальники)
supply = [20, 30, 25]

# Потреби (споживачі)
demand = [10, 25, 20, 20]

# Матриця вартостей
cost = np.array([
    [8, 6, 10, 9],
    [9, 12, 13, 7],
    [14, 9, 16, 5]
])

M = len(supply)
N = len(demand)
def heuristic_initial_solution():
    s = supply.copy()
    d = demand.copy()
    x = np.zeros((M, N))

    cells = [(i, j) for i in range(M) for j in range(N)]
    cells.sort(key=lambda c: cost[c[0]][c[1]])

    for i, j in cells:
        if s[i] > 0 and d[j] > 0:
            v = min(s[i], d[j])
            x[i][j] = v
            s[i] -= v
            d[j] -= v

    return x
def generate_population(size):
    population = []
    for _ in range(size):
        solution = heuristic_initial_solution()
        population.append(solution)
    return population
def fitness(solution):
    total_cost = np.sum(solution * cost)

    # штраф за перевищення концентрації на дорогих маршрутах
    penalty = 0
    avg_cost = np.mean(cost)

    for i in range(M):
        for j in range(N):
            if cost[i][j] > avg_cost:
                penalty += solution[i][j] * 0.5

    return total_cost + penalty
def selection(population, elite_size):
    population.sort(key=lambda x: fitness(x))
    return population[:elite_size]
def crossover(p1, p2):
    child = np.zeros((M, N))
    cut = random.randint(1, M - 1)

    child[:cut] = p1[:cut]
    child[cut:] = p2[cut:]

    return repair(child)
def mutate(solution, rate=0.2):
    sol = solution.copy()

    if random.random() < rate:
        i, j = divmod(np.argmax(cost), N)
        k = random.randint(0, N - 1)

        delta = min(sol[i][j], 3)
        sol[i][j] -= delta
        sol[i][k] += delta

    return repair(sol)
def repair(solution):
    sol = solution.copy()

    # корекція рядків
    for i in range(M):
        diff = supply[i] - np.sum(sol[i])
        if diff != 0:
            j = np.argmin(cost[i])
            sol[i][j] += diff

    # корекція стовпців
    for j in range(N):
        diff = demand[j] - np.sum(sol[:, j])
        if diff != 0:
            i = np.argmin(cost[:, j])
            sol[i][j] += diff

    return np.maximum(sol, 0)
def genetic_algorithm(
    generations=100,
    pop_size=20,
    elite_size=5,
    mutation_rate=0.3
):
    population = generate_population(pop_size)

    for g in range(generations):
        elites = selection(population, elite_size)
        new_population = elites.copy()

        while len(new_population) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    best = min(population, key=lambda x: fitness(x))
    return best
best_solution = genetic_algorithm()

print("Оптимальний план перевезень:")
print(best_solution)

print("\nЗагальна вартість:", fitness(best_solution))
