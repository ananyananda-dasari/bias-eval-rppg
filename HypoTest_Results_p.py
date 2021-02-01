import numpy as np 
import pandas as pd 
from scipy import stats

df = pd.read_csv("Results_Final_Main.csv")
df_v = df.values
df_v = np.asarray(df_v).astype(float)
# print(df.shape)
### Prints p-values for Masimo (Ground truth contact-based sensor) and rPPG methods in the order : 1) Masimo, 2) BKF, 3) Spherical Mean, 4) DeepPhys, 5) CHROM, 6) POS
### 5 p-values are printed for each method in the hypothesis test order: 
### 1) (India Male, India Female) 2) (Sierra Leone Male, Sierra Leone Female) 3) (India Female, Sierra Leone Female) 4) (India Male, Sierra Leone Male) 5) (India, Sierra Leone)

a = []
p_value = []
for i in [4,8,12,16,20]:
    sample_1 = df_v[:, i]
    sample_2 = df_v[:, i+1]
    sample_3 = df_v[:, i+2]
    sample_4 = df_v[:, i+3]
    sample_5 = df_v[:, i+1]
    sample_6 = df_v[:, i+3]
    sample_7 = df_v[:, i]
    sample_8 = df_v[:, i+2]
    sample_9 = np.concatenate((sample_1, sample_2))
    sample_10 = np.concatenate((sample_3, sample_4))
    t1 = stats.ks_2samp(sample_1, sample_2)[1]
    t2 = stats.ks_2samp(sample_3, sample_4)[1]
    t3 = stats.ks_2samp(sample_5, sample_6)[1]
    t4 = stats.ks_2samp(sample_7, sample_8)[1]
    t5 = stats.ks_2samp(sample_9, sample_10)[1]
    p_value.append([t1, t2, t3, t4, t5])

np.savetxt("p_value_.csv", p_value, fmt='%.3f')

