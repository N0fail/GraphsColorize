# Замеры быстродействия алгоритмов
from algo_try import Colorize
import math
from networkx import nx
import numpy as np
from goto import with_goto
from Nezavisimoe_mnoj_verwin import colorize_numpy, Bron, colorize_true_1, colorize_true
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import timeit
from multiprocessing import Process

def wrapper(func, *args, **kwargs):
     def wrapped():
        return func(*args, **kwargs)
     return wrapped

def generate_data(min_n, max_n, sections, seeds, alg = "Olemskoy"):
    out_file = "time_data/" + "{}_{}_".format(min_n, max_n) + alg
    full_data = np.empty((max_n-min_n, max_n*(max_n-1)//2 + 1, seeds))
    plot_data = np.empty((max_n-min_n, sections+1))
    full_data[:, :, :] = np.nan
    plot_data[:, :] = np.nan
    x_data = np.arange(min_n, max_n, dtype=int)
    y_data = np.arange(sections+1, dtype=int)
    for n in range(min_n, max_n):
        max_m = n*(n-1)//2
        prev_section = -1
        section_data = 1
        for m in range(max_m + 1):
            for seed in range(seeds):
                G = nx.gnm_random_graph(n, m, seed=seed)
                T = set(G.nodes())
                GS = []
                for v in G.nodes:
                    GS.append(set(G.adj[v].keys()))
                matrix = nx.to_numpy_matrix(G)
                #start = timer()
                if alg == "Olemskoy":
                    func = wrapper(Colorize, matrix)
                elif alg == "Novikov":
                    func = wrapper(colorize_true, GS, T)
                elif alg == "NovikovOpt":
                    func = wrapper(colorize_true_1, GS, T)
                elif alg == "NovikovMatr":
                    func = wrapper(colorize_numpy, GS, T)
                time = timeit.timeit(func, 'gc.enable()', number=1)
                #res = colorize_true_1(GS, T)
                #time = timer() - start
                full_data[n-min_n, m, seed] = time
            section = int(m / max_m * sections)
            if prev_section == section:
                section_data += 1
                plot_data[n - min_n, section] += np.mean(full_data[n - min_n, m])
            else:
                plot_data[n - min_n, section] = np.mean(full_data[n - min_n, m])
                plot_data[n - min_n, prev_section] /= section_data
                prev_section = section
                section_data = 1

        print(alg + "finished n = {}".format(n))
        np.save(out_file, full_data)
        np.save(out_file + "_plot", plot_data)
        np.save(out_file + "_x", x_data)
        np.save(out_file + "_y", y_data)

if __name__ == '__main__':
    procs = []
    algs = ["Olemskoy", "Novikov", "NovikovOpt", "NovikovMatr"]
    min_ns = [5, 5, 5, 5]
    max_ns = [24, 11, 21, 21]
    for min_n, max_n, alg in zip(min_ns, max_ns, algs):
        proc = Process(target=generate_data, args=(min_n, max_n, 100, 5, alg))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    #generate_data(min_n=5, max_n=13, sections=100, seeds=5, alg="Novikov")
    #generate_data(min_n=10, max_n=16, sections=100, seeds=5, alg="NovikovOpt")
    #generate_data(min_n=10, max_n=16, sections=100, seeds=5, alg="NovikovMatr")
    #generate_data_bron(min_n=5, max_n=10, sections=100, seeds=5, out_file="time_data/compare_full_vs_col_5_8_abs")
