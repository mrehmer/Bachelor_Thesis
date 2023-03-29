import pandas as pd
import matplotlib.pyplot as plt

ldm_probs_dir = '/results/ldm_probs.csv'
vqgan_probs_dir = '/results/vqgan_probs.csv'
val_probs_dir = '/results/val_probs.csv'
accuracies_dir_ldm = '/results/ldm_accuracies.csv'
accuracies_dir_vqgan = '/results/vqgan_accuracies.csv'
val_dir = '/results/val_accuracies.csv'
ldm_class_count_dir = '/results/ldm_class_count.csv'
vqgan_class_count_dir = '/results/vqgan_class_count.csv'
val_class_count_dir = '/results/val_class_count.csv'

# read probabilities from csv file
ldm_probs = pd.read_csv(ldm_probs_dir)
vqgan_probs = pd.read_csv(vqgan_probs_dir)
val_probs = pd.read_csv(val_probs_dir)

# only needed because the probs were saved as tensors
def get_values(df):
    df = df.apply(lambda x: x.str.split('[', expand=True)[1].str.split(']', expand=True)[0].astype(float))
    return df
ldm_probs = get_values(ldm_probs)
vqgan_probs = get_values(vqgan_probs)
val_probs = get_values(val_probs)

# read accuracies from csv file
df_ldm = pd.read_csv(accuracies_dir_ldm)
df_ldm_describe = df_ldm.describe()
mean_ldm = df_ldm_describe.loc['mean']

df_vqgan = pd.read_csv(accuracies_dir_vqgan)
df_vqgan_describe = df_vqgan.describe()
mean_vqgan = df_vqgan_describe.loc['mean']

val_df = pd.read_csv(val_dir)
df_val_describe= val_df.describe()
mean_val = df_val_describe.loc['mean']

# read class count from csv file
ldm_cc = pd.read_csv(ldm_class_count_dir, usecols=[1], names=['LDM'], header=0)
vqgan_cc = pd.read_csv(vqgan_class_count_dir, usecols=[1], names=['VQGAN'], header=0)
val_cc = pd.read_csv(val_class_count_dir, usecols=[1], header=0, names=['VAL'])

# concat datasets 
df_cc = pd.concat([ldm_cc, vqgan_cc, val_cc], axis=1)

df_probs = pd.concat([ldm_probs.describe().loc['mean'], 
                vqgan_probs.describe().loc['mean'], 
                val_probs.describe().loc['mean']], 
                keys=['LDM', 'VQGAN', 'VAL'], axis=1)

df_acc = pd.concat([mean_ldm, mean_vqgan, mean_val], keys=['LDM', 'VQGAN', 'VAL'], axis=1)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

colors=['tab:blue', 'tab:orange', 'tab:green']
parts = ax1.violinplot(dataset=df_cc.values, widths=0.9, showmeans=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_color(colors[i])
    pc.set_facecolor(colors[i])
    pc.set_edgecolor(colors[i])
    pc.set_alpha(0.7)

ax1.set_xticks(range(1, len(df_cc.columns)+1))
ax1.set_xticklabels(df_cc.columns)
ax1.set_ylabel('count')
ax1.set_title('Distributions of class counts')


df_probs.plot(kind='density', ax=ax2)
ax2.set_xlim(left=0, right=1)
ax2.set_xlabel('probability')
ax2.set_title('Distribution of mean probability')


df_acc.plot(kind='density', ax=ax3)
ax3.set_xlim(left=0, right=1)
ax3.set_xlabel('accuracy')
ax3.set_title('Distribution of mean accuracy')

plt.show()