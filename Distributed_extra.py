# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:16:20 2021

@author: DUTTD
"""
import time
import networkx as nx
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import *
import Graph
import graph_100_nodes as gnodes
import Dist_dual_extra as dde

nodes = 100 
L = pd.read_csv('E:/My_Research/Programe/Distributed_optimization/graph_L.csv')
A = pd.read_csv('E:/My_Research/Programe/Distributed_optimization/graph_A.csv')
L = L.values
A = A.values
w, v = np.linalg.eig(L)
diag = np.diagonal(L)
M = np.zeros([nodes,nodes])
for i in range(nodes):
    for j in range(nodes):
        if i == j:
            M[i,j] = 0
        else:
            M[i,j] = A[i,j]/(2*diag[i])
Mc = np.eye(nodes)/2 + M
w_max = np.max(w)

W = np.eye(nodes) - L/w_max
W_tao = (np.eye(nodes) + W)/2

def g(a,b):
    value = a - b
    return value
# np.random.seed(123)
# a_int = np.random.uniform(0.3,0.8,nodes)*100
# b_int = np.random.uniform(1,10,nodes)*100
# a = np.zeros(nodes)
# b = np.zeros(nodes)
# for i in range(nodes):
#     a[i] = int(a_int[i])/100
#     b[i] = int(b_int[i])/100

np.random.seed(123)
a_int = np.random.uniform(0.3,0.8,nodes)*100
b_int = np.random.uniform(1,10,nodes)*100
x_lb_int = np.random.uniform(-5,5,nodes)*100
a = np.zeros(nodes)
b = np.zeros(nodes)
x_lb = np.zeros(nodes)
x_ub = np.zeros(nodes)
for i in range(nodes):
    a[i] = int(a_int[i])/100
    b[i] = int(b_int[i])/100
    x_lb[i] = int(x_lb_int[i])/100
    x_ub[i] = x_lb[i] + 5
# alpha = 0.4
# x_opt, total_iter = dde.D_extra(nodes,W, 100, alpha, a, b)
# print(f'total iteration = {total_iter}')     


# x_lb = np.array([-3,-2,1,1,2,0,0,2,1,1])
# x_ub = np.array([1,2,3,3,4,4,4,4,4,4])

dem = np.array([1]*nodes).transpose()

x_new = np.zeros(nodes)
x_new_1 = np.zeros(nodes)

y_0 = np.zeros(nodes)
y_new_first = np.zeros(nodes)
y_new_second = np.zeros(nodes)

y_inner_value = np.zeros(nodes)


# x_0 = x_lb.copy()
x_0 = dem.copy()
x_value = x_0.copy()


df_y = {}
x_value = x_0.copy()
y_value = y_0.copy()
alpha = 0.4
k = 0
d = dem
total_iter = 0
time_start = time.time()
# W_new = Mc.copy()
W_new = W.copy()
# W_tao_new = W_tao.copy()
# phi = 50 
 
while True:
    # l = np.dot(W,y_0)
    # W_new = W.copy()
    # for n in range(phi):
    #     W_new = np.dot(W_new,W)
    # W_new = np.dot(W_new,Mc)
    # W_new = np.dot(W_new,W)
    # W_tao_new = np.dot(W_tao_new,W_tao)
    # l = np.dot(W_new,y_0)
    l = y_0.copy()
    t = 0
    for i in range(nodes):
        model_x = gp.Model()
        # model_x.update()
        # x = model_x.addVar(lb = x_lb[i], ub = x_ub[i], vtype = GRB.CONTINUOUS, name = 'x')
        x = model_x.addVar(lb = -np.inf, vtype = GRB.CONTINUOUS, name = 'x')

        obj = a[i]*x**2 - b[i]*x + l[i]*(x - d[i])

        model_x.setObjective(obj, GRB.MINIMIZE)
        model_x.setParam('Outputflag',0)
        model_x.optimize()
        x_new[i] = x.x
        # x_new_1[i] = x.x
        # x_new[i] = x_0[i] + 0.8*(x_new_1[i] - x_0[i])
    # y_new_second,t = dde.dual_alg(nodes, l, 100, W, x_new, alpha, k, t)
    for i in range(nodes):
        y = sp.symbols('y')
        f_y = 0.5*(y - l[i])**2 - g(x_new[i],d[i])*y
        df_y[i] = sp.diff(f_y ,y)
   
    for i in range(nodes):    
        y_new_first[i] = np.dot(W[i,:],y_0) - alpha*float(df_y[i].subs('y',y_0[i]))
        # y_new_first[i] = np.dot(Mc[i,:],y_0) - alpha*float(df_y[i].subs('y',y_0[i]))
    while True:
        for i in range(nodes):
            # y_new_second[i] = y_new_first[i] + np.dot(W[i,:],y_new_first) - np.dot(W_tao[i,:],y_0) - alpha*(float(df_y[i].subs('y',y_new_first[i]))- float(df_y[i].subs('y',y_0[i])))
            y_new_second[i] = 2*np.dot(W_tao[i,:],y_new_first) - np.dot(W_tao[i,:],y_0) - alpha*(float(df_y[i].subs('y',y_new_first[i]))- float(df_y[i].subs('y',y_0[i])))
            # y_new_second[i] = y_new_first[i] + np.dot(Mc[i,:],y_new_first) - np.dot(Mc[i,:],y_0) - alpha*(float(df_y[i].subs('y',y_new_first[i]))- float(df_y[i].subs('y',y_0[i])))
            # y_new_second[i] = y_new_first[i] + np.dot(W_new[i,:],y_new_first) - np.dot(W_tao_new[i,:],y_0) - alpha*(float(df_y[i].subs('y',y_new_first[i]))- float(df_y[i].subs('y',y_0[i])))
        if np.linalg.norm(y_new_second - y_new_first) < 1e-5 :
            break
        else:
            y_inner_value = np.append(y_inner_value,y_new_first)
            y_0 = y_new_first.copy()
            y_new_first = y_new_second.copy()
            t = t + 1
            print(f'Out_iter:{k},Inner_iter:{t}')

    if np.linalg.norm(x_new - x_0) < 1e-6 and k > 1:
        break
    else:
        total_iter = total_iter + t
        k = k + 1
        print(f'Out_iter:{k}')
        x_0 = x_new.copy()
        x_value = np.append(x_value, x_0)
        # y_0 = y_new_first.copy()
        y_0 = y_new_second.copy()
        y_value = np.append(y_value,y_0)
time_end = time.time()
# np.save('y_value.npy',y_value)
x_value = x_value.reshape(k+1,nodes).transpose()
y_value = y_value.reshape(k+1,nodes).transpose()
it_num = int(np.size(y_inner_value)/nodes)
y_inner_value = y_inner_value.reshape(it_num,nodes).transpose()
print(f'Total iteration = {total_iter}')
print('Total time cost = ', time_end - time_start)