# Проверка правильности работы алгоритмов
import algo as alg
import algo_try as alg_try
import math
from networkx import nx
import numpy as np
from goto import with_goto
from Nezavisimoe_mnoj_verwin import colorize_numpy, colorize_true, colorize_true_1
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def run_test(n, m, seed):
    G = nx.gnm_random_graph(n, m, seed=seed)
    matrix = nx.to_numpy_matrix(G)
    GS = []
    for v in G.nodes:
        GS.append(set(G.adj[v].keys()))
    T = set(G.nodes())
    start_time = timer()
    ind_set, comb = colorize_numpy(GS, T)
    numpy_time = timer() - start_time
    start_time = timer()
    J = alg_try.Colorize(matrix)
    print(J)
    alg_time = timer() - start_time
    print("numpy time: %.4f,\t alg time: %.4f" % (numpy_time, alg_time))
    print("alr res: {} \nnumpy res: {}".format(len(J), sum(comb)))
    colors = [1,]*n
    for idx, ind in enumerate(ind_set):
        if comb[idx] == 1:
            print(ind_set[idx])
            for i in ind_set[idx]:
                colors[i] = idx
    nx.draw(G, node_color=colors, pos=nx.drawing.layout.kamada_kawai_layout(G), with_labels=True, node_size=1000, cmap='Paired')
    #nx.draw(G, node_color=colors, pos=nx.drawing.layout.spiral_layout(G), with_labels=True, node_size=1000, cmap='Paired')
    #nx.draw(G, node_color=colors, pos=nx.drawing.layout.spring_layout(G, seed=3), with_labels=True, node_size=1000, cmap='Paired')
    plt.show()

def run_tests(min_n, max_n, seed_range, output_file="result.txt"):
    checked = 0
    errors = 0

    with open(output_file, 'a') as f:
        for n in range(min_n, max_n):
            print("n={}\n".format(n))
            for m in range(n-1, int(n*(n-1)/2)+1):
                for sid in range(seed_range):
                    checked += 1
                    G = nx.gnm_random_graph(n, m, seed=sid)
                    matrix = nx.to_numpy_matrix(G)
                    GS = []
                    for v in G.nodes:
                        GS.append(set(G.adj[v].keys()))
                    T = set(G.nodes())

                    ind_set, comb = colorize_numpy(GS, T)
                    #J = alg_try.Colorize(matrix)
                    J = colorize_true(GS, T)

                    if len(J) != sum(comb):
                    #if len(J) != len(res):
                        errors += 1
                        f.write("incorrect results for n = {} and m = {} (seed = {})\n".format(n, m, sid))
                if (m+1) % 10 == 0:
                    print("m = {} from {}".format(m, int(n*(n-1)/2)))
                    pass
            if errors != 0:
                print("{} ERRORS".format(errors))
    print("From {} graphs {} errors".format(checked, errors))

def compare_time(min_n, max_n, seed_range, output_file="result.txt"):
    time_difference = 0
    alg_all_time = 0
    try_all_time = 0
    with open(output_file, 'a') as f:
        for n in range(min_n, max_n):
            print("n={}\n".format(n))
            for m in range(n-1, int(n*(n-1)/2)+1):
                for sid in range(seed_range):
                    G = nx.gnm_random_graph(n, m, seed=sid)
                    matrix = nx.to_numpy_matrix(G)
                    GS = []
                    for v in G.nodes:
                        GS.append(set(G.adj[v].keys()))
                    T = set(G.nodes())

                    start_time = timer()
                    alg_J = alg.Colorize(matrix)
                    alg_time = timer() - start_time
                    start_time = timer()
                    try_J = alg_try.Colorize(matrix)
                    try_time = timer() - start_time

                    alg_all_time += alg_time
                    try_all_time += try_time
                    #print("numpy time: %.4f,\t alg time: %.4f"%(numpy_time, alg_time))

                    if len(alg_J) != len(try_J):
                        print("ERROR {}-{}-{}".format(n, m, sid))
                        f.write("incorrect results for n = {} and m = {} (seed = {})\n".format(n, m, sid))
                if (m+1) % 10 == 0:
                    print("m = {} from {}".format(m, int(n*(n-1)/2)))
                    pass
    print("alg time: {}".format(alg_all_time))
    print("try time: {}".format(try_all_time))

if __name__ == '__main__':
    run_tests(5, 13, 50)