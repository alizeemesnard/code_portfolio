import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random

#Etats possibles: 0 (neuf), 1 (bon etat), 2 (mauvais état), 3 (en panne), 4 (définitivemet cassé).
#Actions possibles: 0 (entretenir), 1 (nerienfaire), 2 (réparer).

#on va coder une méthode congruentielle (pour loi uniforme) et une méthode d'inversion (pour loi non uniforme).
#la méthode congruentielle est déjà codée en Python.

#matrice des revenus: r[i][j] = revenu de l'action j dans l'état i.
r = [[2500, 3000, 0], [500, 1500, -500], [-1000, 500, -2500], [0, 0, -3000], [0, 0, 0]]
#matrice des probabilités de transition vers les états par actions et par états.
#Psa[i][j][k] = proba de transition de l'état i à l'état k par l'action j.
Psa = [[[3/5, 1/5, 0, 0, 1/5],
        [0, 4/6, 0, 1/6, 1/6],
        [0, 0, 0, 0, 0]],

       [[0, 4/8, 2/8, 1/8, 1/8],
        [0, 0, 4/6, 1/6, 1/6],
        [1, 0, 0, 0, 0]],

       [[0, 0, 3/5, 1/5, 1/5],
        [0, 0, 1/3, 1/3, 1/3],
        [1, 0, 0, 0, 0]],

       [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]],

       [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]]]
#matrice des politiques par défaut: pi[i][j] = proba de choisir l'action j dans l'état i.
pi = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]

random.seed(10)

def tirage(p):
    """
    Arguments:
    - p: liste ou numpy array représentant un vecteur stochastique (somme des éléments = 1)

    Retourne:
    - x: un entier aléatoire sélectionné avec la probabilité p[x]
    """
    pcumul = [sum(p[k] for k in range(0, j+1)) for j in range(0, len(p))]
    y = 0
    x = random.random()
    while (x >= pcumul[y]):
        y += 1
    return y
        
#Test de la procédure
estimProbs = [0, 0, 0, 0, 0]
n = 1
while (n<=1000):
    t = tirage([3.0/5, 1.0/5, 0, 0, 1.0/5]); #vecteur des proba d'actions dans etat 0
    n += 1.0
    estimProbs[t] += 1
print([3.0/5, 1.0/5, 0, 0, 1.0/5])
print ([estimProbs[0]/n, estimProbs[1]/n, estimProbs[2]/n, estimProbs[3]/n, estimProbs[4]/n])

def simuleUneTrajectoire(pi, Psa):
    trajectoire = []
    s = 0
    while (s != 4):
        action = tirage(pi[s])
        trajectoire.append([s, action])
        s = tirage(Psa[s][action])
    return trajectoire

#Simulation pour 100 000 trajectoires
nbTraj = 100000
i = 0
while (i < nbTraj):
    trajectoire = simuleUneTrajectoire(pi,Psa)
    i += 1
    for x in range(0, len(trajectoire)):
        print(trajectoire[x][0], trajectoire[x][1], " ",)
        print()

#FONCTION DE VALEUR
# r: matrice des revenus par états et par action (définie au début)
# R: liste des R[s]: revenu total futur suivant la premiere occurence de s
# nbEtatsVisites: nombre d'états visités
# V: liste des revenus moyens V[s]=R[s]*1.0/nbEtatsVisites[s]
# La fonction

def majRevenuMoyen(trajectoire, r, R, nbEtatsVisites,V):
    premiereVisite = [0, 0, 0, 0, 0]
    for x in range(0, len(trajectoire)):
        if premiereVisite[trajectoire[x][0]] == 0:
            premiereVisite[trajectoire[x][0]] = 1
            revenuCumule = sum(r[trajectoire[s][0]][trajectoire[s][1]] for s in range(x, len(trajectoire)-x))
            R[trajectoire[x][0]] += revenuCumule
            nbEtatsVisites[trajectoire[x][0]] += 1
            V[trajectoire[x][0]] = R[trajectoire[x][0]]*1.0/nbEtatsVisites[trajectoire[x][0]]
    return V

#matrice des fonctions de valeur par etat:
#V(i) designe la fonction de valeur pour l'etat i
V = np.zeros(shape=(4))
R = np.zeros(shape=(4))
nbEtatsVisites = np.zeros(shape=(4))
i=1
while(i <= 10):
    trajectoire = simuleUneTrajectoire(pi, Psa)
    majRevenuMoyen(trajectoire, r, R, nbEtatsVisites, V)
    print(V, trajectoire)
    i += 1

print("V=(",V[0],V[1],V[2],V[3],")")

#Affichage de la convergence de la fonction de valeur
print("Affichage de la convergence de la fonction de valeur:")
#matrice des fonctions de valeur par etat:
#V(i) designe la fonction de valeur pour l'etat i)
V = np.zeros(shape=(4))

#La matrice suivante sert à garder une trace des moyennes calculées à chaque etat de la simulation.
#Elle a pour taille le nombre de trajectoires simulees par 4 etats possibles.
courbeVMoyens = np.zeros(shape=(nbTraj, 4))

R = np.zeros(shape=(4))
nbEtatsVisites = np.zeros(shape=(4))

#On fige la graine du generateur aleatoire pour avoir les memes resultats a chaque rejeu:
random.seed(10)
i = 0
while (i < nbTraj):
    trajectoire = simuleUneTrajectoire(pi, Psa)
    majRevenuMoyen(trajectoire, r, R, nbEtatsVisites, V)
    courbeVMoyens[i] = V
    i += 1

print("V=(",V[0],V[1],V[2],V[3],")")
print("La politique appliquee est:")
print(pi[0][0],pi[0][1],pi[0][2])
print(pi[1][0],pi[1][1],pi[1][2])
print(pi[2][0],pi[2][1],pi[2][2])
print(pi[3][0],pi[3][1],pi[3][2])

print ("-----------------------------------------------------")

plt.grid(True, linestyle = "-.")
plt.plot([l for l in range (0, courbeVMoyens.shape[0])], courbeVMoyens[:,0].tolist(), "-", label="Etat 0")
plt.plot([l for l in range (0, courbeVMoyens.shape[0])], courbeVMoyens[:,1].tolist(), "-", label="Etat 1")
plt.plot([l for l in range (0, courbeVMoyens.shape[0])], courbeVMoyens[:,2].tolist(), "-", label="Etat 2")
plt.plot([l for l in range (0, courbeVMoyens.shape[0])], courbeVMoyens[:,3].tolist(), "-", label="Etat 3")
plt.legend()
plt.show()
