# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:19:31 2019

@author: yzr
"""

import numpy as np

# 299 300 299
D_u1 = np.ones((299,299))
a = np.zeros((299,599))
a1 = np.hstack([D_u1,a])

a = np.zeros((300,299))
D_u2 = np.ones((300,300))
b = np.zeros((300,299))
a2 = np.hstack([a,D_u2,b])

a = np.zeros((299,599))
D_u3 = np.ones((299,299))
a3 = np.hstack([a,D_u3])

D_u = np.vstack([a1,a2,a3])

np.save('D_u.npy',D_u)




