# Построение графиков по файлам получаемым из get_data
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy import interpolate
import pandas as pd

def show_graphs(file_name="alg_time"):
    #plt.rcParams["figure.figsize"] = (16, 6)
    x = np.load(file_name+'_x.npy')
    y = np.load(file_name+'_y.npy')
    z = np.load(file_name+'_plot.npy')
    all_data = np.load(file_name+".npy")


    z = np.array(pd.DataFrame(z).interpolate(axis=1).to_numpy())
    #z = np.array(pd.DataFrame(z).interpolate(axis=0).to_numpy())
    pos = np.nanargmax(all_data)
    pos = np.unravel_index(pos, all_data.shape)
    # # print(pos)
    # # print(Z[pos])

    lx = 0
    rx = 3
    lrange = lx + 5
    rrange = rx + 5
    y = y/100
    x = x[lx:rx:]
    z = z[lx:rx, :]
    mindens = 0
    maxdens = 80
    y = y[mindens:maxdens]
    z = z[:, mindens:maxdens]
    #xnew, ynew = np.mgrid[y[0]:y[-1]-0.05:((y[-1] - y[0])/100), x[0]:x[-1]+1:1]
    x, y = np.meshgrid(y, x)
    #tck = interpolate.bisplrep(x, y, z, s=0.003)
    #znew = interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, Z[:, :, 0], cmap=cm.magma, antialiased=True, vmin=0, vmax=2.5)
    #ax.set_zlim(0, 30)
    ax.set_ylabel('число вершин')
    ax.set_xlabel('плотность')
    ax.set_zlabel('время работы, с.')
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title("Алгоритм ПНМ на [{};{}] вершинах при плотности<{}".format(lrange, rrange-1, maxdens/100))
    #surf = ax.plot_surface(xnew, ynew, znew, cmap=cm.magma, antialiased=True, vmin=0)  # cm.plasma
    surf = ax.plot_surface(x, y, z, cmap=cm.magma, antialiased=True, vmin=0)  # cm.plasma
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('alg.png')
    plt.yticks(list(range(lrange, rrange)))
    plt.show()

def show_comparison(x_list, names = ["Olemskoy", "PNMopt", "PNMmatr"], alg_list = ["5_24_Olemskoy", "5_21_NovikovOpt", "5_21_NovikovMatr"], file_name = "time_data/"):
    x = np.load(file_name+alg_list[0]+'_x.npy')
    y = np.load(file_name + alg_list[0] + '_y.npy')
    z = list()
    for alg_idx, alg in enumerate(alg_list):
        z.append(np.load(file_name+alg+'_plot.npy'))
        #z[alg_idx] = np.array(pd.DataFrame(z[alg_idx]).interpolate(axis=1).to_numpy())
    for nodes in x_list:
        for idx, name in enumerate(names):
            bol = np.argwhere(np.isnan(z[idx][nodes, :]) == False)
            bol = bol[:, 0]
            spl = interpolate.splrep(y[bol], z[idx][nodes, bol])
            #y_new = np.linspace(0, 100, num=1000)
            z_new = interpolate.splev(y, spl)
            #plt.plot(y/100, z[idx][nodes,:], label=name)
            plt.plot(y / 100, z_new, label=name)
        #fig = plt.figure()
        #ax = fig.gca()
        plt.title("Сравнение времени работы на графе с {} вершинами".format(nodes+5))
        plt.xlabel("плотность")
        plt.ylabel("время работы, с.")
        plt.legend(loc='upper right')
        plt.show()

if __name__ == '__main__':
    #show_graphs(file_name="time_data/5_11_Novikov")
    #show_comparison([0,3,6,9,12,15])
    show_comparison([6,7,8,9])