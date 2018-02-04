import numpy as np

__author__ = "Justine Pepin, d'apres le code fourni du TP1 listing 1"


# les arrays sont batis avec les dimensions suivantes :
# pluie, arroseur, watson, holmes
# et chaque dimension : faux, vrai

prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print("Pr(Pluie) = {}\n".format(np.squeeze(prob_pluie)))
prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print("Pr(Arroseur) = {}\n".format(np.squeeze(prob_arroseur)))
watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print("Pr(Watson | Pluie) = {}\n".format(np.squeeze(watson)))

holmes = np.array([[[1, 0], [0.1, 0.9]], [[0, 1], [0, 1]]]).reshape(2, 2, 1, 2)
print("Pr( Holmes | Pluie , arroseur) = {}\n".format(np.squeeze(holmes)))

# prob gazon watson mouille si pluie
prob_conj_p_w = (watson * prob_pluie)  # prob conjointes de p, w
watson_mouille = prob_conj_p_w.sum(0).squeeze()[1]  # prob gazon watson mouille
print("Pr(Watson = 1) = {}\n".format(watson_mouille))

# prob gazon holmes mouille si arroseur − pluie
prob_conj_p_a_h = (holmes * prob_pluie * prob_arroseur)  # prob conjointes de p, a, h
holmes_mouille = prob_conj_p_a_h.sum(1).sum(0).squeeze()[1]  # prob gazon holmes mouille
print("a) Pr(Holmes = 1) = {}\n".format(holmes_mouille))  # reponse au numero 1. a)

prob_conj_a_h = prob_conj_p_a_h.sum(0)
prob_arros_si_holmes = prob_conj_a_h.squeeze()[1][1] / holmes_mouille  # prob arroseur soit en marche si h = 1
print("b) Pr(Arroseur = 1 | Holmes = 1) = {}\n".format(prob_arros_si_holmes))  # reponse au numero 1. b)
"""b) Comment est-ce que cette probabilite differe de Pr(A = 1)?

Cette probabilite differe de Pr(A = 1) car A est le parent de H et donc influence la probabilite de H. 
En utilisant la regle de bayes : Pr(A = 1 | H = 1) = Pr(H = 1 | A = 1) * Pr(A = 1) / Pr(H = 1),
il y a un facteur introduit devant Pr(A = 1), et ce facteur n'est pas egal à un (il le serait si et seulement 
 si Pr(H = 1 | A = 1) = Pr(H = 1), autrement dit si A et H etaient conditionnellement independantes.
"""

prob_conj_p_a_w_h = prob_conj_p_a_h * prob_conj_p_w / prob_pluie  # prob conjointes de p, a, w, h
prob_conj_a_w_h = prob_conj_p_a_w_h.sum(0).squeeze()[1][1][1]  # prob conjointes de a = 1, w = 1, h = 1
prob_conj_w_h = prob_conj_p_a_w_h.sum(1).sum(0).squeeze()[1][1]  # prob conjointes de w = 1, h = 1
prob_arros_si_h_w = prob_conj_a_w_h / prob_conj_w_h  # prob arroseur soit en marche si les gazons w, h mouilles
print("c) Pr(Arroseur = 1 | Holmes = 1, Watson = 1) = {}\n".format(prob_arros_si_h_w))  # reponse au numero 1. c)
"""c) Comment est-ce que cette probabilite differe de Pr(A = 1 | H = 1)?

Cette probabilite differe de Pr(A = 1 | H = 1) car etant donne sa couverture de Markov, un noeud est conditionnel-
lement indépendant de tous les autres noeuds du reseau. Comme la couverture comprend les parents du noeud concerne,
ses enfants mais aussi les parents de ses enfants, il faudrait que la valeur de P soit aussi une evidence pour ne pas 
considerer W (ce qui aurait fait que Pr(A = 1 | H = 1) = Pr(A = 1 | H = 1, W = 1)). 
Consequemment, il y a un facteur correspondant à Pr(W = 1 | H, A) / Pr(W | H) avec lequel on retrouve 
Pr(A = 1 | H = 1, W = 1) en multipliant Pr(A = 1 | H = 1).
"""
