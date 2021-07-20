#####################################################
#  Mestrado de Engenharia Biomédica - UFSC
#  Código Adaptado por:
#     -> Jone Follmann e Vinícius Zanon
#  Data: 19/07/2021 
#####################################################

import pyeeg
import numpy as np
import pandas as pd
import scipy.linalg as scpl
import matplotlib.pyplot as plt
from time import sleep
import os
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from pylab import *
from scipy import signal

# Cria listas de arquivos contendo todos os sinais do diretório separados por pastas
dirA = "./Datasets/setA/"
tempA = []
for file in os.listdir(dirA):
    fl = dirA + file
    tempA.append(fl)
tempA = sorted(tempA)  # class: 1

dirB = "./Datasets/setB/"
tempB = []
for file in os.listdir(dirB):
    fl = dirB + file
    tempB.append(fl)
tempB = sorted(tempB)  # class:2

dirC = "./Datasets/setC/"
tempC = []
for file in os.listdir(dirC):
    fl = dirC + file
    tempC.append(fl)
tempC = sorted(tempC)  # class: 3

dirD = "./Datasets/setD/"
tempD = []
for file in os.listdir(dirD):
    fl = dirD + file
    tempD.append(fl)
tempD = sorted(tempD)  # class: 4

dirE = "./Datasets/setE/"
tempE = []
for file in os.listdir(dirE):
    fl = dirE + file
    tempE.append(fl)
tempE = sorted(tempE)  # class: 5

#################################################################

ta = []  # Agrupa todos os sinais listados na lista ta
st = 'A'
for i in range(len(tempA)):
    x = pd.read_table(tempA[i], header=None)
    x.columns = [st + str(i)]
    ta.append(x)

tb = []  # Agrupa todos os sinais listados na lista tb
st = 'B'
for i in range(len(tempB)):
    x = pd.read_table(tempB[i], header=None)
    x.columns = [st + str(i)]
    tb.append(x)

tc = []  # Agrupa todos os sinais listados na lista tc
st = 'C'
for i in range(len(tempC)):
    x = pd.read_table(tempC[i], header=None)
    x.columns = [st + str(i)]
    tc.append(x)

td = []  # Agrupa todos os sinais listados na lista td
st = 'D'
for i in range(len(tempD)):
    x = pd.read_table(tempD[i], header=None)
    x.columns = [st + str(i)]
    td.append(x)

te = []  # Agrupa todos os sinais listados na lista te
st = 'E'
for i in range(len(tempE)):
    x = pd.read_table(tempE[i], header=None)
    x.columns = [st + str(i)]
    te.append(x)


# Define função para concatenar colunas em uma grande tabela
def table(table):
    big_table = None
    for ta in table:
        big_table = pd.concat([big_table, ta], axis=1)
    return big_table


# Preenche a tabela
bigA = table(ta)
bigB = table(tb)
bigC = table(tc)
bigD = table(td)
bigE = table(te)

headA = list(bigA.columns.values)
headB = list(bigB.columns.values)
headC = list(bigC.columns.values)
headD = list(bigD.columns.values)
headE = list(bigE.columns.values)

# Função que cria tabela com os dados e cabeçalho
def creat_mat(mat):
    head = list(mat.columns.values)
    matx = np.zeros((len(mat), (len(head))))
    for i in range(len(head)):
        matx[:, i] = mat[head[i]]
    #        sleep(0.01)
    return matx


matA = creat_mat(bigA)  # : refers to healthy data Eyes Open
matB = creat_mat(bigB)  # : refers to healthy data Eyes Closed
matC = creat_mat(bigC)  # : refers to over interictal  periods
matD = creat_mat(bigD)  # : refers to Interictal (transition between healthy to seizure)
matE = creat_mat(bigE)  # : of ictal or seizures

matA = np.nan_to_num(matA)  # matB[:,0] --- > channel 0, matB[:,1] --- > channel 1 like that
matB = np.nan_to_num(matB)
matC = np.nan_to_num(matC)
matD = np.nan_to_num(matD)
matE = np.nan_to_num(matE)

# Informações do Artigo (ver DOI: https://doi.org/10.1155/2011/406391)
# -> 4097 data point per channel
# -> 173.61 Hz sample rate and there are 4097 data point for each channel
# -> total 100 channel are their
# -> 4097/173.61 = 23.59 sec
# -> the raw data from one of the channels for the 23.59 sec

# Define número de features a serem calculadas e cria uma lista de nomes para colunas f1, f2, ..., f(n-1)
feature_size = 4 # Atenção! Para n features, define-se (n-1) feature_size (index_init is zero)
columns_name = list()
for i in range(feature_size+1):
    columns_name = columns_name + ['f' + str(i + 1)]
# print(columns_name)

# Função para a extração de features.
# Descomentar as features desejadas e ajustar o número de features de acordo na célula anterior

def features(mat):
    # Parâmetros definidos pelo Artigo
    Kmax = 5
    Tau = 4
    DE = 10

    lis = list()

    # To speed up, it is recommended to compute W before calling this function
    # (svd_entropy or fisher_info) because W may also be used by other functions
    # where as computing it here again will slow down.

    M = pyeeg.embed_seq(mat, Tau, DE)
    W = scpl.svd(M, compute_uv=0)
    W /= sum(W)

    # Extração de Features
    lis = lis + [pyeeg.dfa(mat)]                      # DFA
    lis = lis + [pyeeg.hfd(mat, Kmax)]                # HFD
    lis = lis + [pyeeg.svd_entropy(mat, Tau, DE, W)]  # SVD_Entropy
    lis = lis + [pyeeg.fisher_info(mat, Tau, DE, W)]  # Fisher_Information
    lis = lis + [pyeeg.pfd(mat)]                      # PFD

    sleep(0.01)

    return lis

# Cria matriz contendo os Features para cada Dataset
MftA = np.zeros((100, feature_size + 1))
for i in range(100):
    MftA[i, :] = features(matA[:, i])

MftB = np.zeros((100, feature_size + 1))
for i in range(100):
    MftB[i, :] = features(matB[:, i])

MftC = np.zeros((100, feature_size + 1))
for i in range(100):
    MftC[i, :] = features(matC[:, i])

MftD = np.zeros((100, feature_size + 1))
for i in range(100):
    MftD[i, :] = features(matD[:, i])

MftE = np.zeros((100, feature_size + 1))
for i in range(100):
    MftE[i, :] = features(matE[:, i])

FCM_A = pd.DataFrame(MftA, columns=columns_name)
FCM_B = pd.DataFrame(MftB, columns=columns_name)
FCM_C = pd.DataFrame(MftC, columns=columns_name)
FCM_D = pd.DataFrame(MftD, columns=columns_name)
FCM_E = pd.DataFrame(MftE, columns=columns_name)

TotalDataset = pd.concat([FCM_A, FCM_B, FCM_C, FCM_D, FCM_E], ignore_index=True)

# Salva atributos extraídos
TotalDataset.to_csv('features.csv')
# print(TotalDataset.to_csv(index=False))

# Extrai alguns parâmetros para o Plot (Média, Desvio Padrão, Minimo e Máximo)
FCM_A = np.mean(FCM_A); std_FCM_A = np.std(FCM_A); max_FCM_A = max(FCM_A); min_FCM_A = min(FCM_A);
FCM_B = np.mean(FCM_B); std_FCM_B = np.std(FCM_B); max_FCM_B = max(FCM_B); min_FCM_B = min(FCM_B);
FCM_C = np.mean(FCM_C); std_FCM_C = np.std(FCM_C); max_FCM_C = max(FCM_C); min_FCM_C = min(FCM_C);
FCM_D = np.mean(FCM_D); std_FCM_D = np.std(FCM_D); max_FCM_D = max(FCM_D); min_FCM_D = min(FCM_D);
FCM_E = np.mean(FCM_E); std_FCM_E = np.std(FCM_E); max_FCM_E = max(FCM_E); min_FCM_E = min(FCM_E);

# Extrai de cada Dataset a Média do Feature de Interesse para o Plot (conforme sugerido o Artigo)
datasets = ['A', 'B', 'C', 'D', 'E']
DFA_F = [FCM_A[0], FCM_B[0], FCM_C[0], FCM_D[0], FCM_E[0]]
HFD_F = [FCM_A[1], FCM_B[1], FCM_C[1], FCM_D[1], FCM_E[1]]
SVD_F = [FCM_A[2], FCM_B[2], FCM_C[2], FCM_D[2], FCM_E[2]]
FI_F = [FCM_A[3], FCM_B[3], FCM_C[3], FCM_D[3], FCM_E[3]]
PFD_F = [FCM_A[4], FCM_B[4], FCM_C[4], FCM_D[4], FCM_E[4]]

# Plot das Médias dos Features
figure(1, (12, 7))
plt.subplot(2, 3, 1)
plt.plot(datasets, DFA_F, '--*', label='DFA')
plt.title("Detrend Fluctuation Analysis")
plt.grid(color='black', linestyle='--', linewidth=0.2)
plt.legend()
plt.subplot(2, 3, 2)
plt.plot(datasets, HFD_F, '--*', label='HFD')
plt.title("Higuchi Fractal Dimension")
plt.grid(color='black', linestyle='--', linewidth=0.2)
plt.legend()
plt.subplot(2, 3, 3)
plt.plot(datasets, SVD_F, '--*', label='SVD')
plt.title("Singular Values Decomposition Entropy")
plt.grid(color='black', linestyle='--', linewidth=0.2)
plt.legend()
plt.subplot(2, 3, 4)
plt.plot(datasets, FI_F, '--*', label='FI')
plt.title("Fisher Information")
plt.grid(color='black', linestyle='--', linewidth=0.2)
plt.legend()
plt.subplot(2, 3, 5)
plt.plot(datasets, PFD_F, '--*', label='PFD')
plt.title("Petrosian Fractal Dimension")
plt.grid(color='black', linestyle='--', linewidth=0.2)
plt.legend()
plt.suptitle('Features Extraction on Datasets EEG (Averages)', fontsize=15)
plt.show()

# # Plot dos Features (Errorbar)
# figure(2, (12, 7))
# plt.subplot(2, 3, 1)
# plt.errorbar(datasets, DFA_F, yerr=(1-std_FCM_A)/max_FCM_A)
# plt.title("Detrend Fluctuation Analysis")
# plt.grid(color='black', linestyle='--', linewidth=0.2)
# plt.subplot(2, 3, 2)
# plt.errorbar(datasets, HFD_F, yerr=(1-std_FCM_B)/max_FCM_B)
# plt.title("Higuchi Fractal Dimension")
# plt.grid(color='black', linestyle='--', linewidth=0.2)
# plt.subplot(2, 3, 3)
# plt.errorbar(datasets, SVD_F, yerr=(1-std_FCM_C)/max_FCM_C)
# plt.title("Singular Values Decomposition Entropy")
# plt.grid(color='black', linestyle='--', linewidth=0.2)
# plt.subplot(2, 3, 4)
# plt.errorbar(datasets, FI_F, yerr=(1-std_FCM_D)/max_FCM_D)
# plt.title("Fisher Information")
# plt.grid(color='black', linestyle='--', linewidth=0.2)
# plt.subplot(2, 3, 5)
# plt.errorbar(datasets, PFD_F, yerr=(1-std_FCM_E)/max_FCM_E)
# plt.title("Petrosian Fractal Dimension")
# plt.grid(color='black', linestyle='--', linewidth=0.2)
# plt.suptitle('Features Extraction on Datasets EEG (Errorbars)', fontsize=15)
# plt.show()
