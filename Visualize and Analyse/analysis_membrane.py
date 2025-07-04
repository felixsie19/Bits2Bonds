import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creating DataFrame
df = pd.read_csv("output_membrane.csv")

# Grouping by molecule
grouped = df.groupby('molecule')

# Calculating mean and standard deviation of work for each group
summary_df = grouped['work'].agg(['mean', 'std']).reset_index()

# Renaming columns for clarity
summary_df.columns = ['molecule', 'mean_work', 'std_work']

# Sorting by mean_work in descending order
#summary_df = summary_df.sort_values(by='mean_work', ascending=False).reset_index(drop=True)
print(summary_df)
# Plotting the data
plt.figure(figsize=(10, 6))
plt.errorbar(summary_df.index, summary_df['mean_work'], yerr=summary_df['std_work'], fmt='o', capsize=5, label='Mean Work with Std Dev')

# Fitting a trend line
z = np.polyfit(summary_df.index, summary_df['mean_work'], 1)
p = np.poly1d(z)
plt.plot(summary_df.index, p(summary_df.index), "r--", label='Trend Line')

plt.xlabel('Index')
plt.ylabel('Mean Work')
plt.title('Mean Work with Standard Deviation and Trend Line')
plt.legend()
plt.grid(True)
plt.show()
 
