# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import numpy as np
from numpy.linalg import multi_dot
    
    
def loss(X, U, H, V, D_u, D_v, W_u, W_v, lamda_u, lamda_v):
    Part1 = X-multi_dot([U,H,V.T])
    Part3 = multi_dot([U.T,(D_u-W_u),U])
    Part5 = multi_dot([V.T,(D_v-W_v),V])
    sta1 = np.linalg.norm(Part1,ord=2,keepdims=False)
    sta3 = lamda_u * np.trace(Part3)
    sta5 = lamda_v * np.trace(Part5)
    
    print(sta1,sta3,sta5)


def NMF(X, U, H, V, D_u, W_u, D_v, W_v, steps=100, lamda_u = 0.1, lamda_v = 0.1):  

    for step in range(steps):  
        
        # Update matrix H
        H = H * ((multi_dot([U.T, X, V]) / multi_dot([U.T, U, H, V.T, V]))**(0.5))

        # Update matrix U
        U = U * (((multi_dot([X, V, H.T]) + lamda_u * np.dot(W_u, U) ) /
                 (multi_dot([U, H, V.T, V, H.T]) + lamda_u * np.dot(D_u, U)))**(0.5))
        
        # Update matrix V
        V = V * (((multi_dot([X.T, U, H]) + lamda_v * np.dot(W_v, V)) /
                ( multi_dot([V, H.T, U.T, U, H]) + lamda_v * np.dot(D_v, V)))**(0.5))

        
        # loss
        loss(X, U, H, V, D_u, D_v, W_u, W_v, lamda_u, lamda_v)
        
                
#        if step % 100 == 1:
#            print(step)
#            print('U_final: ')
#            print(U)
#            print('H_final: ')
#            print(H)
#            print('V_final: ')
#            print(V)

    return U, H, V