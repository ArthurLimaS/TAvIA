import numpy as np


def cx_one_point(ind1, ind2):
    cxpoint = np.random.randint(1, len(ind1) - 1)
    temp_ind = np.copy(ind1)

    ind1[cxpoint:] = ind2[cxpoint:]
    ind2[cxpoint:] = temp_ind[cxpoint:]

    return ind1, ind2


def cx_two_point(ind1, ind2):
    cxpoint1 = np.random.randint(1, ind1)
    cxpoint2 = np.random.randint(1, ind1 - 1)
    temp_ind = np.copy(ind1)

    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        temp_cxpoint = cxpoint1
        cxpoint1 = cxpoint2
        cxpoint2 = temp_cxpoint

    ind1[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2]
    ind2[cxpoint1:cxpoint2] = temp_ind[cxpoint1:cxpoint2]

    return ind1, ind2


def cx_uniform(ind1, ind2, indpb=0.5):
    size = min(len(ind1), len(ind2))

    for i in range(size):
        if np.random.rand() < indpb:
            temp = ind1[i]
            ind1[i] = ind2[i]
            ind2[i] = temp

    return ind1, ind2