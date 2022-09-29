import numpy as np
import dataset_loader as dl
import moga_ocd

np.random.seed(seed=3)

graph = dl.test_function()

_lambda = 100
_mu = 10
ngen = 5
nconv = 10
mutpb = 0.1
indmutpb = 0.1

#print(ff.get_communities(graph, np.array([2, 4, 1, 0, 3, 6, 8, 5, 7])))

print(moga_ocd.run(graph, _lambda, _mu, ngen, nconv, mutpb, indmutpb))