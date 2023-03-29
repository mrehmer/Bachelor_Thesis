import pandas as pd
import matplotlib.pyplot as plt

# directories
accuracies_dir_ldm = '/results/ldm_accuracies.csv'
accuracies_dir_vqgan = '/results/vqgan_accuracies.csv'

superclass_vqgan_dir = '/results/vqgan_accuracies_superclass.csv'
superclass_ldm_dir = '/results/ldm_accuracies_superclass.csv'

# load accuracies
ldm_acc = pd.read_csv(accuracies_dir_ldm)
vqgan_acc = pd.read_csv(accuracies_dir_vqgan)

# load superclass accuracy
ldm_sc = pd.read_csv(superclass_ldm_dir)
vqgan_sc = pd.read_csv(superclass_vqgan_dir)

df_ldm = pd.concat([ldm_acc.describe().loc['mean'],
                    ldm_sc.describe().loc['mean']],
                    keys=['LDM', 'LDM-SC'],
                    axis=1)
df_vqgan = pd.concat([vqgan_acc.describe().loc['mean'],
                    vqgan_sc.describe().loc['mean']],
                    keys=['VQGAN', 'VQGAN-SC'],
                    axis=1)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
df_vqgan.plot(kind='density', ax=ax1, color=['tab:orange', 'firebrick'])
ax1.set_xlim(left=0, right=1)
ax1.set_xlabel('accuracy')
ax1.set_title('VQGAN vs. VQGAN-SC')

df_ldm.plot(kind='density', ax=ax2, color=['tab:blue', 'lightsteelblue'])
ax2.set_xlim(left=0, right=1)
ax2.set_xlabel('accuracy')
ax2.set_title('LDM vs. LDM-SC')

plt.show()

