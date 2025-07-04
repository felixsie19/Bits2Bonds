import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame.
data = pd.read_csv('lead_output.csv')  # Data is already loaded here

# Create a dictionary to map unique combinations of columns 1 and 2 to colors.
color_map = {}
colors = plt.cm.tab20.colors  # Use a colormap with enough distinct colors.
for i, (col1, col2) in enumerate(zip(data['1'].astype(str), data['2'].astype(str))):  # Use 'data' here
    key = (col1, col2)
    if key not in color_map:
        color_map[key] = colors[i % len(colors)]

# Create the scatter plot.
plt.figure(figsize=(10, 10))
for i in range(len(data)):  # Use 'data' here
    col1 = data['1'][i]  # Use 'data' here
    col2 = data['2'][i]  # Use 'data' here
    color = color_map[(str(col1), str(col2))]
    plt.scatter(data['generation'][i],data['lead_score'][i], color=color, label=(col1, col2),s=52)  # Use 'data' here

# Add labels and title.
plt.xlabel('Generation')
plt.ylabel('Lead Score')
plt.title('Lead Score vs. Generation Scatter Plot')

# Add green shaded area between y=-30 and y=30
plt.axhspan(-1, 1, facecolor='green', alpha=0.3)

# Add red shaded area above 30 and below -30

plt.xlim(0,70)
plt.axhspan(1, plt.gca().get_ylim()[1], facecolor='red', alpha=0.3)
plt.axhspan(plt.gca().get_ylim()[0], -1, facecolor='red', alpha=0.3)
# Create a legend.
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.ylim(10000,-10000)
#plt.yscale("symlog")
#plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5,  
# 1.20), ncol=2)  # Adjust legend position
plt.show()