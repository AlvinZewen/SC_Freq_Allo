
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import csv
import copy

from check_solution import *
from yield_mc_simulation import *
from frequency_graph import FrequencyGraph

conn_map = [(0, 1), (1, 2), (3, 2), (4, 0), (4, 5), (4, 8), (5, 1), (5, 6), (5, 9), (6, 2), (6, 10), (7, 3), (7, 6), (7, 11), (8, 9), (10, 9), (11, 10), (12, 8), (12, 13), (13, 9), (13, 14), (14, 10), (15, 11), (15, 14)]

tolerance = 1e-4

if __name__ == '__main__':

    path = sys.argv[1]

    freqs = []
    with open(path + 'freqs.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            freqs.append(float(row[1]))
    freqs = np.array(freqs)

    a = []
    with open(path + 'anharms.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            a.append(float(row[1]))
    a = np.array(a)

    freqs_d = []
    with open(path + 'drive_freqs.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            freqs_d.append(float(row[-1]))
    freqs_d = np.array(freqs_d)

    # thresholds
    d = np.array([0.017, 0.03, 0.03, 0.017, 0.03, 0.002, 0.017, 0.025, 0.017])

    best_yield = 0

    np.random.seed(0)
    for iters in range(1000): 
        if iters>0:
            freqs   = best_freqs + tolerance*np.random.uniform(0,1,len(best_freqs))
            a       = best_a + tolerance*np.random.uniform(0,1,len(best_a))
            freqs_d = best_freqs_d + tolerance*np.random.uniform(0,1,len(best_freqs_d))

        yields = np.zeros(10)
        for reps in range(10):

            # graph definition
            G = FrequencyGraph(conn_map, freqs, a, f_drive=freqs_d, cz=False)

            # key definition
#            keys = ['A1', 'A2i', 'A2j', "E1", "E2", "E4", "F1", "F2", "M1"]
            d_dict = {k: dd for (k, dd) in zip(keys, d)}
#            cr_keys = ['A1', 'A2i', 'A2j', "E1", "E2", "E4", "C1", "C1b", "F1", "F2", "M1"]
            cr_keys = ['A1', 'A2i', 'A2j', "D1", "C1", "C1b", "S1", "S2", "T1"]
            cstr_key = cr_keys

            # plotting

            collisions, c, idx_len, _ = G.get_collision(
                d_dict, sigma=0.01, qutrit=False, cstr=cstr_key)

            yields[reps] = np.sum(collisions==0)/len(collisions)
        if np.mean(yields) >= best_yield:
            best_yield = np.mean(yields)
            best_freqs = copy.deepcopy(freqs)
            best_a     = copy.deepcopy(a)
            best_freqs_d = copy.deepcopy(freqs_d)
            print(best_yield)
        
        if best_yield >= 0.999:
            break

    print(best_yield)
    print(best_freqs)
    print(best_a)
    print(best_freqs_d)

