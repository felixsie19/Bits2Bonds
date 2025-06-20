import matplotlib.pyplot as plt
import numpy as np
# Sample data (you'll need your actual data here)
series1_y = [ -15.57, -12.078, -42.155, -54.539, -75.245, -1.076, -2.221, -88.413, -23.732, -486.312, -3.639, -7.571, -515.712, -91.79, -465.47, -160.426, -153.378, -292.43, -489.891, -356.195, -49.955, -94.36, -239.476, -693.474, -315.294] 
series2_y = [489.291, 525.216, 271.702, 844.436, 775.968, 5.628, 331.178, 827.179, 550.168, 668.777,0, 542.591, 830.311, 958.853, 790.944, 620.09, 565.781, 755.361, 827.547, 920.417, 0.0, 0.0, 819.52, 766.727, 605.431 ]
differences = [a + b for a, b in zip(series1_y, series2_y)]

# Calculate absolute values (equivalent to 'Betrag' in German)
abs_differences = [abs(x) for x in differences]
x_labels = [
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
]

z = np.polyfit(x_labels, abs_differences, 1)  # Fit a linear trendline (degree 1)
p = np.poly1d(z)

plt.figure(figsize=(4, 8))
# Plot the data
plt.bar(x_labels, series1_y, label="siRNA Challenge")
plt.scatter(x_labels,abs_differences, label="Absolute Differences",color="green")
plt.plot(x_labels, p(x_labels), "r--", label="Trendline") 

# Customize if needed
plt.xlabel("Molecule Number") 
plt.ylabel("Work [kJ/mol]")
plt.legend()
plt.subplots_adjust(bottom=0.1,top=1)

# Rotate x-axis labels for readability
plt.xticks(rotation=30, ha='right') 
plt.savefig("test2.png")
plt.show()
