import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Creating DataFrame
df = pd.read_csv("lead_output.csv")
print(df)

# Grouping by bead_hydro and bead_lipo
grouped = df.groupby(['bead_hydro', 'bead_lipo'])

# Calculating mean and standard deviation of lead_score for each group
summary_df = grouped['lead_score'].agg(['mean', 'std']).reset_index()

# Renaming columns for clarity
summary_df.columns = ['bead_hydro', 'bead_lipo', 'mean_lead_score', 'std_lead_score']
# Sorting by mean_lead_score in descending order
summary_df = summary_df.sort_values(by='mean_lead_score', ascending=False).reset_index(drop=True)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.errorbar(summary_df.index, summary_df['mean_lead_score'], yerr=summary_df['std_lead_score'], fmt='o', capsize=5, label='Mean Lead Score with Std Dev')

# Fitting a trend line
z = np.polyfit(summary_df.index, summary_df['mean_lead_score'], 1)
p = np.poly1d(z)
plt.plot(summary_df.index, p(summary_df.index), "r--", label='Trend Line')

plt.xlabel('Index')
plt.ylabel('Mean Lead Score')
plt.title('Mean Lead Score with Standard Deviation and Trend Line')
plt.legend()
plt.grid(True)
plt.show()