import pandas as pd
import arff

df = pd.read_csv('dn2a.csv')
df['x'].plot()
