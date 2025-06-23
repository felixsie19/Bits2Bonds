import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Read the data from the file
def read_data(filename):
    # Skip the first line with the "#! FIELDS" and then read the file
    df = pd.read_csv(filename, delim_whitespace=True, comment='#', names=['time','z-distance', 'res.work'])
    return df

def fit_polynomial(df, degree=3):
    # Fit a polynomial of the specified degree to the data
    p = Polynomial.fit(df['z-distance'], df['res.work'], deg=degree)
    return p

def calculate_derivative(p, df):
    # Calculate the derivative of the polynomial
    dp = p.deriv()
    
    # Apply the derivative function to the 'z-distance' column
    df['derivative'] = dp(df['z-distance'])
    return df

def plot_data(dfs, polynomials, filenames):
    # Plot the original data and the fitted polynomial for multiple files
    plt.figure(figsize=(10, 6))
    
    # Loop over each DataFrame, polynomial, and filename to plot them
    for df, p, filename in zip(dfs, polynomials, filenames):
        # Original data
        plt.plot(df['z-distance'], df['res.work'], label=f'Res Work ({filename})', linestyle='-', marker='x')
        
        # Fitted polynomial
        #plt.plot(df['z-distance'], p(df['z-distance']), label=f'Polynomial Fit ({filename})', linestyle='--')
        
        # Plot derivative as a second plot (optional)
        #plt.plot(df['z-distance'], df['derivative'], label=f'Derivative ({filename})', linestyle=':')
    
    # Labels and grid
    plt.xlabel('z-distance')
    plt.ylabel('Res Work / Derivative')
    plt.title('z-distance vs Res Work with Polynomial Fit and Derivative')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example of usage for multiple files
filenames = ['SQ2p_l+TP1_SQ2p_h+TC6/COLVAR_double_Membrane', 'SQ2p_l+SC1_SC1_SC1_SC1_SC1_SC1_SQ2p_h+TC6/COLVAR_double_Membrane', 'SQ2p_l+TQ4p_N6d+N5a_N5a_N5a_N5a/COLVAR_double_Membrane','SQ2p_l+TQ4p_SQ2p_h+TP1/COLVAR_double_Membrane']  # Add your file paths here

# Prepare lists to store dataframes and polynomials for each file
dfs = []
polynomials = []

# Loop through each file, process and store the data and polynomial
for filename in filenames:
    df = read_data(filename)
    polynomial_fit = fit_polynomial(df, degree=3)  # Adjust degree as needed
    df = calculate_derivative(polynomial_fit, df)
    
    # Append to lists
    dfs.append(df)
    polynomials.append(polynomial_fit)

# Plot all the results on the same figure
plot_data(dfs, polynomials, filenames)
