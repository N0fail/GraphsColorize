# Реализция алгоритма Олемского И.В.
import math
from networkx import nx
import numpy as np
from goto import with_goto
from Nezavisimoe_mnoj_verwin import colorize_numpy
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import bitarray as bit
import bitarray.util

@with_goto
def Colorize(matrix):
    h = list()  # Горизонтальные структурные множества
    for i in range(matrix.shape[0]):
        #h.append(set())
        h.append(bit.bitarray(matrix.shape[0]))
        h[i].setall(False)
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                #h[i].add(j)
                h[i][j] = True

    v = list()  # Вертикальные структурные множества
    for j in range(matrix.shape[1]):
        #v.append(set())
        v.append(bit.bitarray(matrix.shape[0]))
        v[j].setall(False)
        for i in range(matrix.shape[0]):
            if matrix[i, j] == 0:
                #v[j].add(i)
                v[j][i] = True

    #J0 = [set([i]) for i in range(matrix.shape[0])]   #Начальное разбиение на цвета
    J0 = []
    for i in range(matrix.shape[0]):
        J0.append(bit.bitarray(matrix.shape[0]))
        J0[i].setall(False)
        J0[i][i] = True

    v0 = len(J0)
    J = []
    nodes = set(range(matrix.shape[0])) # мн-во всех вершин

    d = list()
    for idx_i, i in enumerate(h):
        d.append(list())
        for idx_j, j in enumerate(v):
            #d[idx_i].append(i & j)
            d[idx_i].append(i & j)

    D = dict()
    for q in range(matrix.shape[0]):
        for r in range(q + 1, matrix.shape[0]):
            #if (set([q, r]) & d[q][r]) and (set([q, r]) & d[r][q]):
            if (d[q][r][q])and(d[q][r][r])and(d[r][q][q])and(d[r][q][r]):
            #if (d[q][r][q] or d[q][r][r]) and (d[r][q][q] or d[r][q][r]):
                D[(q, r)] = d[q][r] & d[r][q]

    #w = set(range(matrix.shape[0]))  # опорное мно-во
    w = bit.bitarray(matrix.shape[0])
    w.setall(True)
    w = [[w]]  # w-матрица опорных мн-в
    s = 0  # начальный уровень
    j = 0  # начальная ветка

    G = [[set()]]  # мн-во пар возможных продолжений
    Q = [[set()]]  # мн-во пар использованных элементов G
    #F = [[set()]]  # мн-во элементов, использованных на этом уровне
    F = [[bit.bitarray(matrix.shape[0])]]  # мн-во элементов, использованных на этом уровне
    F[0][0].setall(False)

    alpha = [[(1,2)]]  # узловой эл-т - пара
    #psi = [[set()]]  # множество исп. при прореживании
    psi = [[bit.bitarray(matrix.shape[0])]]
    psi[0][0].setall(False)
    Z = set()  # множество неперспективных элементов

    B = []

    while True:  # основной цикл раскраски, условие?
        label .p1
        #w[j][s] = nodes.copy()
        #w[j][s] = bit.bitarray(matrix.shape[0])
        w[j][s].setall(True)
        for i in range(j):
            #w[j][s] -= J[i]  # убираем из рассмотрения уже раскрашенные вершины (п.1)
            w[j][s] = w[j][s] & ~J[i]

        label .p2
        #if not w[j][s]:  # (п.2)
        if not w[j][s].any():
            if s == 0:
                goto .p8
                # переход на п.8
            else:
                goto .p5
                # переход на п.5

        label .p3

        G[j][s] = set()
        for key in D:  # формирование списка возможных продолжений(п.3)
            #if len(set(key) & w[j][s]) == 2:
            if w[j][s][key[0]] and w[j][s][key[1]]:
                G[j][s].add(key)

        label .p4
        temp = G[j][s]-Q[j][s]  # мн-во возможных продолжений с отсечением (п.4)
        if temp:
            #alpha[j][s] = max(temp, key=lambda x: len(D[x] & w[j][s]))  # (п.4, равенство 5)
            alpha[j][s] = max(temp, key=lambda x: bit.util.count_and(D[x], w[j][s]))
            #ro = len(D[alpha[j][s]] & w[j][s])
            ro = bit.util.count_and(D[alpha[j][s]], w[j][s])
            if s == 0:
                #if j + len(w[j][0]) / ro >= v0:  # Проверка типа А)
                if j + math.ceil(w[j][0].count() / ro) >= v0:
                    j -= 1
                    if j < 0:
                        return J0
                    #s = math.ceil(len(J[j])/2) - 1
                    s = math.ceil(J[j].count() / 2) - 1
                    goto.p4

            if j == 0:  # Проверка типа B)
                if s > 0:
                    # Если максимальное потенциальное множество мешьше среднего в лучшей раскраске, оно не подходит
                    if 2 * (s) + ro <= math.ceil(matrix.shape[0] / v0):
                        s -= 1
                        goto.p4

            if j == v0 - 2:  # Проверка типа C)
                # если текущий лучший вариант не покроет все оставшиеся вершины
                #if 2 * (s) + ro != len(w[j][0]):
                if 2 * (s) + ro != w[j][0].count():
                    j -= 1
                    if j < 0:
                        return J0
                    # if len(J[j]) % 2 == 0:
                    #    s -= 1
                    #s = math.ceil(len(J[j]) / 2) - 1
                    s = math.ceil(J[j].count() / 2) - 1
                    goto.p4

            #new_w = (w[j][s] & D[alpha[j][s]]) - set(alpha[j][s])  # (п.4, равенство 6)
            new_w = w[j][s] & D[alpha[j][s]]
            new_w[alpha[j][s][0]] = False
            new_w[alpha[j][s][1]] = False
            #for ss in range(s+1):  # !!!
            #    Q[j][ss].add(alpha[j][s])
            Q[j][s].add(alpha[j][s])
            F[j][s][alpha[j][s][0]] = True  # !!!
            F[j][s][alpha[j][s][1]] = True  # !!!
            s += 1
            if s >= len(w[j]):
                w[j].append(new_w.copy())
            else:
                w[j][s] = new_w.copy()
            if len(G[j]) <= s:
                G[j].append(set())
            #else:
            #    G[j][s] = set()
            if len(Q[j]) <= s:
                Q[j].append(set())
            else:
                Q[j][s] = set()
            if len(alpha[j]) <= s:
                alpha[j].append((1,1))
            if len(F[j]) <= s:
                #F[j].append(set())
                F[j].append(bit.bitarray(matrix.shape[0]))
                F[j][s].setall(False)
            else:
                #F[j][s] = set()
                #F[j][s] = bit.bitarray(matrix.shape[0])
                F[j][s].setall(False)
            if len(psi[j]) <= s:
                #psi[j].append(set())
                psi[j].append(bit.bitarray(matrix.shape[0]))
                psi[j][s].setall(False)
            goto .p2
            pass  # переход на п.2
        else:
            #temp = w[j][s] - F[j][s]
            temp = w[j][s] & ~F[j][s]
            #for el in Q[j][s]:  ## верно !!!
                #temp -= set(el)
            #    temp[el[0]] = False
            #    temp[el[1]] = False
            #if temp:
            if temp.any():
                #beta = temp.pop()
                beta = temp.index(True)
                temp[beta] = False
                #F[j][s].add(beta)
                F[j][s][beta] = True
                ro = 1
                if s == 0:
                    #if j + len(w[j][0]) / ro >= v0:  # Проверка типа А)
                    if j + math.ceil(w[j][0].count() / ro) >= v0:  # Проверка типа А)
                        j -= 1
                        if j < 0:
                            return J0
                        #s = math.ceil(len(J[j]) / 2) - 1
                        s = math.ceil(J[j].count() / 2) - 1
                        goto.p4

                    for jj in range(j):  # !!!
                        F[jj][0][beta] = True

                if j == 0:  # Проверка типа B)
                    if s > 0:
                        if 2 * (s) + ro < math.ceil(matrix.shape[0] / v0):
                            s -= 1
                            goto.p4

                if j == v0 - 2:  # Проверка типа C)
                    #if 2 * (s) + ro != len(w[j][0]):
                    if 2 * (s) + ro != w[j][0].count():
                        j -= 1
                        if j < 0:
                            return J0
                        #s = math.ceil(len(J[j]) / 2) - 1
                        s = math.ceil(J[j].count() / 2) - 1
                        goto.p4
            else:
                s -= 1
                if s < 0:
                    j -= 1
                    if j < 0:
                        return J0
                    #s = math.ceil(len(J[j]) / 2) - 1
                    s = math.ceil(J[j].count() / 2) - 1
                goto .p4
                # переход на п.4

        label .p5
        #if not w[j][s]:  # (п.5)
        if not w[j][s].any():
            #tmp = set()
            tmp = bit.bitarray(matrix.shape[0])
            tmp.setall(False)
            for i in range(s):
                #tmp |= set(alpha[j][i])
                tmp[alpha[j][i][0]] = True
                tmp[alpha[j][i][1]] = True
            #if tmp:
            if tmp.any():
                if len(J) <= j:
                    J.append(tmp)
                else:
                    J[j] = tmp
        else:
            if (not G[j][s]-Q[j][s]):
                #tmp = set()  # текущее одноцветное множество
                tmp = bit.bitarray(matrix.shape[0])
                tmp.setall(False)
                for i in range(s):
                    #tmp |= set(alpha[j][i])
                    tmp[alpha[j][i][0]] = True
                    tmp[alpha[j][i][1]] = True
                #tmp.add(beta)
                tmp[beta] = True
                #if tmp:
                if tmp.any():
                    if len(J) <= j:
                        J.append(tmp.copy())
                    else:
                        J[j] = tmp.copy()

        label .p6
        #var = len(J[j])//2
        var = J[j].count()//2
        for ss in range(var):  # (п.6)
            psi[j][ss] = J[j].copy()
            temp = G[j][ss]-Q[j][ss]
            for i in range(ss):  # ss+1 ???
                #psi[j][ss] -= set(alpha[j][i])
                psi[j][ss][alpha[j][i][0]] = False
                psi[j][ss][alpha[j][i][1]] = False

            Z = set()
            for a in temp:
                #if (a[0] in psi[j][ss])and(a[1] in psi[j][ss]):
                #if set(a) <= psi[j][ss]:  # !!!
                #if psi[j][ss][a[0]] and psi[j][ss][a[1]]:
                if psi[j][ss][a[0]] and psi[j][ss][a[1]]:
                    #if (D[a] & w[j][ss]) <= psi[j][ss]:  # если в лучшем случае сформируется такое же множество # !!! не нужно??
                        Z.add(a)
            Q[j][ss] |= Z
        for jj in range(j):  # !!!
           Q[j][0] |= Q[jj][0]

        isPsiHere = False
        isJHere = False
        if len(B) <= j:
            B.append([])
        #B = B[0:j+1]  # !!!
        for el in B[j]:
            if el == psi[j][0]:
                isPsiHere = True
                s = math.ceil(J[j].count() / 2) - 1
                goto .p3

            #if el >= J[j]:  # !!!
            #if el | J[j] == el:
            #    isJHere = True
            #if el == J[j]
        if not isPsiHere:
            B[j].append(psi[j][0].copy())
        #if isJHere:
            #s = math.ceil(len(J[j]) / 2) - 1
            #s = math.ceil(J[j].count()/2) - 1
            #goto .p3


        label .p7
        j += 1  # п.7
        s = 0
        if len(B) > j:
            B[j] = []
        if len(G) <= j:
            G.append([set()])
        else:
            G[j] = [set()]
        if len(Q) <= j:
            Q.append([set()])
        else:
            Q[j] = [set()]
        if len(F) <= j:
            #F.append([set()])
            F.append([bit.bitarray(matrix.shape[0])])
            F[j][s].setall(False)
        else:
            #F[j][s] = set()
            F[j][s].setall(False)
        if len(alpha) <= j:
            alpha.append([(1,1)])
        if len(psi) <= j:
            #psi.append([set()])
            psi.append([bit.bitarray(matrix.shape[0])])
            psi[j][s].setall(False)
        else:
            psi[j][s].setall(False)
        if len(w) <= j:
            #w.append([set()])
            w.append([bit.bitarray(matrix.shape[0])])
            #w[j][s].setall(False)
        else:
            w[j] = [bit.bitarray(matrix.shape[0])]
        goto .p1
        pass  # переход на п.1

        label .p8
        #if (not w[j][0]) and (sum(map(lambda x: len(x), J[0:j])) == matrix.shape[0]):  # (п.8)
        check = bit.bitarray(matrix.shape[0])
        check.setall(False)
        for jj in range(j):
            check = check | J[jj]
        if (not w[j][0].any()) and (check.all()):
            v = j
            J = J[0:j]
            if v < v0:
                v0 = v
                J0 = J[0:j]
            if G[0][0] - Q[0][0]:
                #ro = len(D[alpha[0][0]] & w[0][0])
                ro = bit.util.count_and(D[alpha[0][0]], w[0][0])
            else:
                ro = 1
            #if math.ceil(len(w[0][0])/ro) == v0:  # лучшая раскраска проверка на стр.6
            if math.ceil(w[0][0].count() / ro) == v0:  # лучшая раскраска проверка на стр.6
                break
            j = v - 2
            if j < 0:
                j = 0
            #s = math.ceil(len(J[j]) / 2) - 1
            s = math.ceil(J[j].count() / 2) - 1
            goto .p4

    return J0





