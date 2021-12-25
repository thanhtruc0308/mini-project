from numpy import *
import pandas as pd
from scipy.linalg import svd
import sys
import csv
import numpy as np
maxInt = sys.maxsize

while True:


    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
df = pd.read_csv('./LSA.csv', engine='python', nrows=1000)
B= array(df)
index = list()
for i in range(0, 1000, 1):
    index.append(B[i][0]) 


df.drop(df.columns[[0]], axis=1, inplace=True)
A = array(df)
print(A)

# SVD
U, s, VT = svd(A)
Sigma = zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
print(Sigma)
df2 = pd.DataFrame(Sigma, index=index)
df2.to_csv("SVD.csv")