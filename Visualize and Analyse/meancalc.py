import pandas as pd
import matplotlib.pyplot as plt

# Assuming your data is in a CSV file named 'molecule_data.csv'
# If your data is in another format, you'll need to adjust the loading method accordingly
data = pd.read_csv('mean_diff_data_siRNA_disso.csv', header=None, names=['Molecule Number', 'Molecule Name', 'Score'])

# Group the data by molecule name and calculate the mean and standard deviation of the score
grouped_data = data.groupby('Molecule Number')['Score'].agg(['mean', 'std']).reset_index()
print(grouped_data)
# Create a bar plot of the mean scores with error bars representing the standard deviation
plt.figure(figsize=(10, 6))
plt.bar(grouped_data['Molecule Number'], grouped_data['mean'], yerr=grouped_data['std'], capsize=5)
plt.xlabel('Molecule Name')
plt.ylabel('Mean Score')
plt.title('Mean and Standard Deviation of Molecule Scores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
 
