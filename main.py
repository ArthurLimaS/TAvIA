import numpy as np
import dataset_loader as dl
import moga_ocd
import crossovers as c

graph = dl.karate_club_loader()

_lambda = 1000
_mu = 100
ngen = 50
nconv = 10
mutpb = 0.1
indmutpb = 0.1

moga_ocd.run(graph, _lambda, _mu, ngen, nconv, mutpb, indmutpb, c.cx_one_point)