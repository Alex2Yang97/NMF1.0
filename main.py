# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import numpy as np
import nmf

# Initialize the number of cluster
k = 3

# 加载TFIDF矩阵
X = np.load("X.npy")

(a,b) = X.shape

# Initialize the constraint matrix for comments
# 299 300 299
#W_u = np.load('D_u.npy')
W_u = np.eye(898)

# Initialize the constraint matrix for spots
W_v = np.load('D_v.npy')
#W_v = np.eye(6700)

n = len(X)  
m = len(X[0])  

#X = np.array(X)  
W_u = np.array(W_u)  
W_v = np.array(W_v)  

# Initialize the W_u & W_v
D_u = np.zeros(shape=(n,n))
D_v = np.zeros(shape=(m,m))

sum_W_u = np.sum(W_u,axis=1)
sum_W_v = np.sum(W_v,axis=1)

for i in range(n):
    D_u[i][i] = sum_W_u[i]
    
for h in range(m):
    D_v[h][h] = sum_W_v[h]

   
U = np.random.rand(n,k)  
H = np.random.rand(k,k)  
V = np.random.rand(m,k)  
   
U_final, H_final, V_final = nmf.NMF(X, U, H, V, D_u, W_u, D_v, W_v)  

print("\nU_final:\n\n", U_final)
print("\nH_final:\n\n", H_final)
print("\nV_final:\n\n", V_final)