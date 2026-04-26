import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pd.read_csv("shiftwing-mach-0.85_analysis_results.csv")


df_rel = df

print(df_rel.head())
print(len(df_rel))

print(np.min(df_rel.iloc[:, :9], axis=0))
print(np.max(df_rel.iloc[:, :9], axis=0))