from madeuce import *
import numpy as np
from matplotlib import pyplot as plt

axs = []

# examples taken from 'SYSTEM RELIABILITY THEORY' by Marvin Rausand and Arnljot Hsyland
# example 4.1 eq. inbetween 4.6 & 4.7
struct1_2 = SeriesStructure([Component(.95,None),Component(.97,None),Component(.94,None),])
print(f'exaple 4.1 theory 0.866 actual {struct1_2.R(0):.3f}')

# example 4.1 eq 4.7
struct1_3 = SeriesStructure([Component(.995,None) for _ in range(10)])
print(f'exaple 4.1 eq 4.7 theory 0.951 actual {struct1_3.R(0):.3f}')

# example 4.2 eq inbetween 4.8 & 4.9
struct2_5 = ParallelStructure([Component(.95,None),Component(.97,None),Component(.94,None)])
print(f'exaple 4.2 theory 0.99991 actual {struct2_5.R(0):.5f}')

# example 4.2 eq 4.9
T = np.linspace(0,10,1000)
p = lambda t: np.exp(-t)
if not False:
    print('example 4.2 eq 4.9 see graph')
    fig,ax = plt.subplots()
    axs.append(ax)
    #ax.set_title('ex 4.2 eq 4.9')
    fig.canvas.manager.set_window_title('ex 4.2 eq 4.9')
    ax.set_ylabel('Rs(t)')
    for n in [1,2,4,8]:
        struct2_6 = ParallelStructure([Component(p,None) for _ in range(n)])
        Z = [struct2_6.R(t) for t in T]
        R_theory = [1-(1-p(t))**n for t in T]
        ax.plot(T,R_theory, label=f'1oo{n} theoretic', linewidth=5)
        ax.plot(T,Z, label=f'1oo{n}')

# eq 4.10
T = np.linspace(0,5,100)
p = lambda t: 0.5*np.exp(-t)
if not False:
    print('eq 4.10 see graph')
    fig,ax = plt.subplots()
    axs.append(ax)
    #ax.set_title('eq 4.10')
    fig.canvas.manager.set_window_title('eq 4.10')
    ax.set_ylabel('Rs(t)')
    for n in [10,15]:
        for k in [2,3,5]:
            struct3 = KNStructure(k,n,None,p)
            R_th_f = lambda t: sum([(comb(n,y) * p(t)**y * (1-p(t))**(n-y)) for y in range(k,n+1)])
            Z_th = [R_th_f(t) for t in T]
            Z = [struct3.R(t) for t in T]
            ax.plot(T,Z_th, linewidth=5,label=f'{k}oo{n} theoretic')
            ax.plot(T,Z, label=f'{k}oo{n}')

# example 4.3
struct3_3 = KNStructure(2,4,None,.97)
print(f'exaple 4.3 theory 0.99989 actual {struct3_3.R(0):.5f}')

# example 4.4
p = [0,.1,.1,.1,.2,.3,.4,.5,.6]
struct_4_1 = SeriesStructure([
    KNStructure(2,3,None,p[1]), 
    Component(p[4],None),
    Component(p[5],None),
    Component(p[6],None),
    ParallelStructure([
        Component(p[7],None),
        Component(p[8],None),
    ])
])
print(f'exaple 4.4 theory {(p[1]*p[2] + p[1]*p[3] + p[2]*p[3] - 2*p[1]*p[2]*p[3]) * p[4]*p[5]*p[6] * (p[7] + p[8] - p[7]*p[8])} actual {struct_4_1.R(0)}')

# example 4.7
T = np.linspace(0,7,100)
z1 = lambda t: 1*np.exp(-2*t)
z2 = lambda _: 0.1
z3 = lambda t: 0.001*np.exp(1*t)
struct1_1 = SeriesStructure([Component(None, z1), Component(None, z2), Component(None, z3)])
Z1_1 = [struct1_1.Z(t) for t in T]
z1_1 = [z1(t) for t in T]
z1_2 = [z2(t) for t in T]
z1_3 = [z3(t) for t in T]
if not False:
    print('example 4.7 see graph')
    fig,ax = plt.subplots()
    #ax.set_title('example 4.7')
    fig.canvas.manager.set_window_title('example 4.7')
    axs.append(ax)
    ax.plot(T,Z1_1, label='3oo3')
    ax.plot(T,z1_1, linestyle='--', label='3oo3 z1 = e^(-2t)')
    ax.plot(T,z1_2, linestyle='--', label='3oo3 z2 = 0.1')
    ax.plot(T,z1_3, linestyle='--', label='3oo3 z3 = 0.001e^t')

# example 4.8
T = np.linspace(0,7,100)
if not False:
    fig,ax = plt.subplots()
    axs.append(ax)
    print('example 4.8 see graph')
    #ax.set_title('example 4.8')
    fig.canvas.manager.set_window_title('example 4.8')
    ax.set_ylabel('Rs(t)')
    for n in [2,4,8]:
        for z in [.5, 2]:
            struct2_3 = ParallelStructure([Component(None,z) for _ in range(n)])
            R2_3_th = lambda t: sum([(comb(n,x) * (-1)**(x+1) * np.exp(-z*x*t)) for x in range(1,n+1)])
            R2_3 = [struct2_3.R(t) for t in T]
            R2_3_th = [R2_3_th(t) for t in T]
            ax.plot(T,R2_3_th, linewidth=5, label=f'1oo{n} z={z} theoretic')
            ax.plot(T,R2_3, label=f'1oo{n} z={z}')


# example 4.9
T = np.linspace(0,10,100)
struct2_1 = ParallelStructure([Component(None,.4), Component(None,.6)])
struct2_2 = ParallelStructure([Component(None,.2), Component(None,.8)])
Z2_1 = [struct2_1.Z(t) for t in T]
Z2_2 = [struct2_2.Z(t) for t in T]
if not False:
    print('example 4.9 see graph')
    fig,ax = plt.subplots()
    #ax.set_title('example 4.9')
    fig.canvas.manager.set_window_title('example 4.9')
    axs.append(ax)
    ax.plot(T,Z2_1, label='1oo2 z1 = 0.4, z2 = 0.6')
    ax.plot(T,Z2_2, label='1oo2 z1 = 0.2, z2 = 0.8')

# eq 4.36
if not False:
    T = np.linspace(0,3,100)
    fig,ax = plt.subplots()
    axs.append(ax)
    #ax.set_title('eq 4.36')
    fig.canvas.manager.set_window_title('eq 4.36')
    ax.set_ylabel('Rs(t)')
    print('eq 4.36 see graph')
    for z in [1,2,4]:
        struct = KNStructure(2,3,z,None)
        Z = [struct.R(t) for t in T]
        Z_th = [(3*np.exp(-2*z*t) - 2*np.exp(-3*z*t)) for t in T]
        ax.plot(T,Z_th, linewidth=5, label=f'2oo3 z={z} theoretic')
        ax.plot(T,Z, label=f'2oo3 z={z} actual')


# fig 4.5
if not False:
    T = np.linspace(0,3,100)
    fig,ax = plt.subplots()
    axs.append(ax)
    #ax.set_title('fig 4.5')
    fig.canvas.manager.set_window_title('fig 4.5')
    ax.set_ylabel('Zs(t)')
    print('fig 4.5 see graph')
    struct = KNStructure(2,3,1,None)
    Z = [struct.Z(t) for t in T]
    ax.plot(T,Z, label=f'2oo3')

# table 4.1
T = np.linspace(0,3,100)
z=1
if not False:
    fig,ax = plt.subplots()
    axs.append(ax)
    #ax.set_title('table 4.1')
    fig.canvas.manager.set_window_title('table 4.1')
    ax.set_ylabel('Rs(t)')
    for z in [1]:
        struct1 = Component(None, z)
        struct2 = ParallelStructure([Component(None,z), Component(None,z)])
        struct3 = KNStructure(2,3,z,None)
        R1_act = [struct1.R(t) for t in T]
        R2_act = [struct2.R(t) for t in T]
        R3_act = [struct3.R(t) for t in T]
        R1_th_f = lambda t: np.exp(-z*t)
        R2_th_f = lambda t: 2*np.exp(-z*t) - np.exp(-2*z*t)
        R3_th_f = lambda t: 3*np.exp(-2*z*t) - 2*np.exp(-3*z*t)
        R1_th = [R1_th_f(t) for t in T]
        R2_th = [R2_th_f(t) for t in T]
        R3_th = [R3_th_f(t) for t in T]
        ax.plot(T,R1_th, linewidth=5,label=f'1oo1 z={z} theory')
        ax.plot(T,R2_th, linewidth=5,label=f'1oo2 z={z} theory')
        ax.plot(T,R3_th, linewidth=5,label=f'2oo3 z={z} theory')
        ax.plot(T,R1_act,label=f'1oo1 z={z} actual')
        ax.plot(T,R2_act,label=f'1oo2 z={z} actual')
        ax.plot(T,R3_act,label=f'2oo3 z={z} actual')

# eq 4.39
T = np.linspace(0,5,100)
z = 1
if not False:
    fig,ax = plt.subplots()
    axs.append(ax)
    #ax.set_title('eq 4.39')
    fig.canvas.manager.set_window_title('eq 4.39')
    ax.set_ylabel('Rs(t)')
    for n in [10,15]:
        for k in [2,3,5]:
            struct3 = KNStructure(k,n,z,None)
            R_th_f = lambda t: sum([(comb(n,x) * np.exp(-z*t*x) * (1-np.exp(-z*t))**(n-x)) for x in range(k,n+1)])
            Z_th = [R_th_f(t) for t in T]
            Z = [struct3.R(t) for t in T]
            ax.plot(T,Z_th, linewidth=5,label=f'{k}oo{n} theoretic')
            ax.plot(T,Z, label=f'{k}oo{n} actual')
for ax in axs:
    ax.set_xlabel('t')
    if not ax.get_ylabel():
        ax.set_ylabel('Zs(t)')

    ax.grid()
    ax.legend()
    ax.ticklabel_format(useOffset=False)

plt.show()
