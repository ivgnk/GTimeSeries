import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("mpg_ggplot2.csv")
df_select = df.loc[df.cyl.isin([4,8]), :]

# Plot
sns.set_style("white")
# gridobj = sns.lmplot(x="displ", y="hwy", hue="cyl", data=df_select,
#                      height=7, aspect=1.5, robust=True, palette='tab10',
#                      scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))

x1 = np.array([1,2,2,4,5,6]); y1 = np.array([1,2,2,4,5,6]);

gridobj = sns.lmplot(x="x1", y="y1", data=df_select,
                     height=7, aspect=1.5, robust=True, palette='tab10',
                     scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))


# Decorations
gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))
plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=10)
plt.show()