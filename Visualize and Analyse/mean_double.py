import pandas as pd
import matplotlib.pyplot as plt

# **1. Load the CSV data**
df = pd.read_csv("double_membrane_mean.csv")

# **2. Extract the columns you want to plot**
work_column = df["index"]  # Assuming the "Work" column exists
columns_to_plot = df["work"].tolist()  
print(columns_to_plot) 

# **3. Create the plot**
plt.figure(figsize=(10, 6))  # Adjust figure size as needed

for column in columns_to_plot:
    plt.bar(work_column, df[column], label=column)

# **4. Customize the plot**
plt.xlabel("Molecule Index")
plt.ylabel("kJ/mol")  # Assuming your values are in kJ/mol
plt.title("Evolution of siRNA Interaction Energy")
plt.legend()
plt.grid(True)

# **5. Show the plot**
plt.show()
 
