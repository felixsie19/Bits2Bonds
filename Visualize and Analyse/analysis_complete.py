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
 
 
# Creating DataFrame
df_sirna = pd.read_csv("output_sirna.csv")

# Grouping by molecule
grouped_sirna = df_sirna.groupby('molecule')

# Calculating mean and standard deviation of work for each group
summary_df_sirna = grouped_sirna['work'].agg(['mean', 'std']).reset_index()

# Renaming columns for clarity
summary_df_sirna.columns = ['molecule', 'mean_work', 'std_work']

# Sorting by mean_work in descending order
#summary_df = summary_df.sort_values(by='mean_work', ascending=False).reset_index(drop=True)
print(summary_df_sirna)
# Plotting the data
plt.figure(figsize=(10, 6))
plt.errorbar(summary_df_sirna.index, summary_df_sirna['mean_work'], yerr=summary_df_sirna['std_work'], fmt='o', capsize=5, label='Mean Work with Std Dev')

# Fitting a trend line
z = np.polyfit(summary_df_sirna.index, summary_df_sirna['mean_work'], 1)
p = np.poly1d(z)
plt.plot(summary_df_sirna.index, p(summary_df_sirna.index), "r--", label='Trend Line')

plt.xlabel('Index')
plt.ylabel('Mean Work')
plt.title('Mean Work with Standard Deviation and Trend Line')
plt.legend()
plt.grid(True)
plt.show()
 
# Creating DataFrame
df_sirna_disso = pd.read_csv("output_sirna_disso.csv")

# Grouping by molecule
grouped_sirna_disso = df_sirna_disso.groupby('molecule')

# Calculating mean and standard deviation of work for each group
summary_df_sirna_disso = grouped_sirna_disso['work'].agg(['mean', 'std']).reset_index()

# Renaming columns for clarity
summary_df_sirna_disso.columns = ['molecule', 'mean_work', 'std_work']

# Sorting by mean_work in descending order
#summary_df = summary_df.sort_values(by='mean_work', ascending=False).reset_index(drop=True)
print(summary_df_sirna_disso)
# Plotting the data
plt.figure(figsize=(10, 6))
plt.errorbar(summary_df_sirna_disso.index, summary_df_sirna_disso['mean_work'], yerr=summary_df_sirna_disso['std_work'], fmt='o', capsize=5, label='Mean Work with Std Dev')

# Fitting a trend line
z = np.polyfit(summary_df_sirna_disso.index, summary_df_sirna_disso['mean_work'], 1)
p = np.poly1d(z)
plt.plot(summary_df_sirna_disso.index, p(summary_df_sirna_disso.index), "r--", label='Trend Line')

plt.xlabel('Index')
plt.ylabel('Mean Work')
plt.title('Mean Work with Standard Deviation and Trend Line')
plt.legend()
plt.grid(True)
plt.show()


df_merged = pd.merge( summary_df_sirna,summary_df_sirna_disso, on='molecule', suffixes=( '_1','_2'))
print(df_merged)
# Subtract mean_work values
df_merged['mean_diff'] = df_merged['mean_work_1'] + df_merged['mean_work_2']

# Propagate the errors
df_merged['std_diff'] = np.sqrt(df_merged['std_work_1']**2 + df_merged['std_work_2']**2)
df_merged = df_merged.sort_values(by='mean_diff', ascending=False)
# Display the resulting dataframe
print(df_merged)

# Plotting the data
plt.figure(figsize=(12, 8))  # Increase figure size to provide more space for x-axis labels
plt.errorbar(df_merged['molecule'], df_merged['mean_diff'], yerr=df_merged['std_diff'], fmt='o', capsize=5, label='Mean Difference with Std Dev')

# Fitting a trend line
z = np.polyfit(range(len(df_merged)), df_merged['mean_diff'], 1)
p = np.poly1d(z)
plt.plot(range(len(df_merged)), p(range(len(df_merged))), "r--", label='Trend Line')

plt.xlabel('Molecule')
plt.ylabel('Mean Difference')
plt.title('Mean Difference with Standard Deviation and Trend Line')
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.legend()
plt.grid(True)
plt.show()