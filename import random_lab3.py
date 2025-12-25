import random
import numpy as np

# Дані задачі
supply = [20, 30, 25]
demand = [10, 25, 20, 20]

cost = np.array([
    [8, 6, 10, 9],
    [9, 12, 13, 7],
    [14, 9, 16, 5]
])

M, N = len(supply), len(demand)


# ----------------- Допоміжні функції -----------------

def heuristic_solution():
    s = supply.copy()
    d = demand.copy()
    x = np.zeros((M, N))

    cells = [(i, j) for i in range(M) for j in range(N)]
    random.shuffle(cells)

    for i, j in cells:
        if s[i] > 0 and d[j] > 0:
            v = min(s[i], d[j])
            x[i][j] = v
            s[i] -= v
            d[j] -= v

    return x


def repair(sol):
    sol = sol.copy()

    for i in range(M):
        diff = supply[i] - sol[i].sum()
        sol[i][random.randrange(N)] += diff

    for j in range(N):
        diff = demand[j] - sol[:, j].sum()
        sol[random.randrange(M)][j] += diff

    return np.maximum(sol, 0)


def fitness(sol):
    return np.sum(sol * cost)


# ----------------- Генетичні оператори -----------------

def mutate(sol):
    sol = sol.copy()
    i = random.randrange(M)
    j1, j2 = random.sample(range(N), 2)
    delta = random.randint(1, 5)

    sol[i][j1] = max(0, sol[i][j1] - delta)
    sol[i][j2] += delta

    return repair(sol)


def crossover(p1, p2):
    child = p1.copy()
    i = random.randrange(M)
    j = random.randrange(N)
    child[i][j] = p2[i][j]
    return repair(child)


# ----------------- Генетичний алгоритм -----------------

def genetic_algorithm(pop_size=20, generations=100):
    # 1. Початкове покоління
    population = []
    for _ in range(pop_size):
        sol = heuristic_solution()
        sol += np.random.randint(-3, 4, size=(M, N))
        population.append(repair(sol))

    # 2. Еволюція
    for _ in range(generations):
        new_population = population.copy()

        for i in range(pop_size):
            parent = population[i]
            p2 = random.choice(population)

            child = crossover(parent, p2)
            child = mutate(child)

            # 3. Заміна тільки якщо краще
            if fitness(child) < fitness(parent):
                new_population[i] = child

        population = new_population

    return min(population, key=fitness)


# ----------------- Запуск -----------------

best = genetic_algorithm()

print("Найкращий розв'язок:")
print(best)
print("Вартість:", fitness(best))
