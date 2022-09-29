import numpy as np
import fitness_functions as ff
from deap import base
from deap import creator
from deap import tools
from deap.tools import selNSGA2, ParetoFront


def init_random_pop(graph, _lambda):
    edges_count = graph.edges_list.num_edges
    pop = np.zeros(shape=[_lambda, edges_count])

    for i in range(len(pop)):
        ind = np.zeros(shape=[edges_count])
        for j in range(edges_count):
            adj_edges = graph.edges_list.adj_edges[j]
            r = np.random.randint(0, len(adj_edges))

            ind[j] = adj_edges[r]

        pop[i, :] = ind

    return pop


def crossover(ind1, ind2):
    cxpoint = np.random.randint(1, len(ind1) - 1)
    temp_ind = np.copy(ind1)

    ind1[cxpoint:] = ind2[cxpoint:]
    ind2[cxpoint:] = temp_ind[cxpoint:]

    return ind1, ind2


def mutation(graph, ind, indmutpb):
    for i in range(len(ind)):
        r1 = np.random.rand()
        if r1 < indmutpb:
            adj_edges = graph.edges_list.adj_edges[i]
            r2 = np.random.randint(0, len(adj_edges))
            ind[i] = adj_edges[r2]

    return ind


def crossover_and_mutation(graph, Cbest, mutpb, indmutpb):
    r1 = np.random.randint(0, len(Cbest))
    r2 = np.random.randint(0, len(Cbest))

    ind1 = Cbest[r1]
    ind2 = Cbest[r2]

    ind1, ind2 = crossover(ind1, ind2)

    mut_choice = np.random.rand()
    if mut_choice < mutpb:
        ind1 = mutation(graph, ind1, indmutpb)

    mut_choice = np.random.rand()
    if mut_choice < mutpb:
        ind2 = mutation(graph, ind2, indmutpb)
    
    return ind1, ind2


def fitness(graph, big_c):
    F = np.zeros(shape=[len(big_c), 2])
    for i in range(len(F)):
        comms = ff.get_communities(graph, big_c[i])
        F[i][0] = ff.local_clustering_coefficient(graph, comms)
        F[i][1] = ff.separability(graph, comms)

    return F


def get_individual(big_c, index):
    return big_c[index]


def nom_dominated_sort_and_sel_n_best_nsga2(big_c, big_f, _mu):
    toolbox = base.Toolbox()

    pop = []
    for i in range(len(big_c)):
        toolbox.register("get_individual", get_individual, big_c, i)
        pop.append(tools.initIterate(creator.Individual, toolbox.get_individual))
        toolbox.unregister("get_individual")

    
    for i in range(len(pop)):
        pop[i].fitness.values = (big_f[i][0], big_f[i][1])

    Cbest_deap = selNSGA2(pop, _mu)
    Cbest = np.zeros(shape=[_mu, len(big_c[0])])
    for i in range(len(Cbest)):
        Cbest[i, :] = Cbest_deap[i][:]

    pf = ParetoFront(similar=np.array_equal)
    pf.update(Cbest_deap)
    
    return Cbest, pf


def run(graph, _lambda, _mu, ngen, nconv, mutpb, indmutpb):
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    C = np.zeros(1)
    i = 0
    convergence = 0
    Cbest = np.zeros(1)
    pf = ParetoFront()

    while (i <= ngen) \
    and (convergence < nconv):
        print(i)
        if i == 0:
            C = init_random_pop(graph, _lambda)
        else:
            C[0:_mu] = Cbest[0:_mu]
            j = _mu
            while j < _lambda:
                ind1, ind2 = crossover_and_mutation(graph, Cbest, mutpb, indmutpb)
                C[j,:] = ind1
                j += 1

                C[j,:] = ind2
                j += 1
        
        F = fitness(graph, C)
        Cbest, pf = nom_dominated_sort_and_sel_n_best_nsga2(C, F, _mu)
        i += 1

    return pf