import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("lead_output.csv")
print(df.columns)

# Calculate the trendline (linear regression)
z = np.polyfit(df["generation"], df["lead_score"], 1)
p = np.poly1d(z)

plt.scatter(df["generation"], df["lead_score"], label="Score of lead molecule")
plt.plot(df["generation"],p(df["generation"]), "r--", label="Trendline")  # Add the trendline
plt.ylabel("Performance score")
#plt.ylim(1500,2500)
plt.xlabel("Generation")
plt.legend()
plt.show()
