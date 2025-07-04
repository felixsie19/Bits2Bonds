 
import pandas as pd
import os 
import subprocess
import numpy as np
import matplotlib.pyplot as plt

##################Requirements############################

# 1. File "backbone.gro"
# 2. File "backbone.itp"
# 3. Working version of plumed (compiled with prefix gmx_plu)

################## Creation of .gro file ##################

def insert_beads_before_last_line(file_path, new_bead_names=['INS'], initial_distance=0.5):
    new_lines, beads_to_add, bead_info = [], [], []
    current_distance = initial_distance

    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_atom_count = int(lines[1].strip())  # Get the initial atom count from the second line

    for line in lines[1:-1]:  # Process all lines except the last
        if len(line.strip()) == 0:
            continue

        if 'SN4' in line:
            split_line, (x, y, z) = line.split(), map(float, line.split()[3:6])
            res_name, atom_type = "0UNK", int(split_line[2])

            for new_bead_name in new_bead_names:
                updated_atom_count += 1
                bead_info.append((new_bead_name, updated_atom_count))
                new_x, new_y, new_z = x, y + current_distance, z + current_distance
                current_distance += initial_distance
                bead_line = f"{res_name:>8}{new_bead_name:>7}{updated_atom_count:>5}{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}\n"
                beads_to_add.append(bead_line)
                
        
        new_lines.append(line)
    new_lines.extend(beads_to_add)  # Append new beads before the last line
    new_lines.append(lines[-1])     # Append the last line (box dimensions)
    new_lines[0] = f"This file was automatically generated\n{updated_atom_count}\n"  # Update atom count

    new_file_path = file_path.replace('.gro', f'_modified_{bead_names_str}.gro')
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(new_lines)

    return new_file_path, bead_info

# Example usage for .gro file creation
file_path = 'backbone.gro'
new_bead_names = ["SN4","SC1","TN6d","SC1","SC1"]
bead_names_str = ''.join(new_bead_names)
modified_file_path,bead_info = insert_beads_before_last_line(file_path, new_bead_names, initial_distance=0.5)


################## Adaption of .itp file [atoms] section ##################

def add_new_beads_to_itp(itp_file_path, new_beads, output_file_path):
    with open(itp_file_path, 'r') as file:
        lines = file.readlines()

    atoms_start, atoms_end = None, None
    for i, line in enumerate(lines):
        if '[ atoms ]' in line or '[atoms]' in line:
            atoms_start = i + 1
        elif atoms_start and line.startswith('['):
            atoms_end = i - 1
            break

    for number, name, charge in new_beads:
        if name=='TN6d':
            charge=1
        else:
            charge=0
        
        new_line = f"{number:>6}{name:>8}      1{name:>7}{name:>6}{number:>6}{charge:>8}\n"
        lines.insert(atoms_end, new_line)
        atoms_end += 1

    with open(output_file_path, 'w') as file:
        file.writelines(lines)

# Example usage for .itp file adaption
itp_file_path = "backbone.itp"

new_beads = [(number, name, 0) for name, number in bead_info]
output_file_path = 'backbone_modified_atoms.itp'
add_new_beads_to_itp(itp_file_path, new_beads, output_file_path)

################## Adaption of .itp file [bonds] section ##################

def update_itp_with_bond_info(csv_file_path, itp_file_path, output_file_path, bead_info):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    # Convert DataFrame to a dictionary for easier lookup
    bond_dict = {(row['atom1'], row['atom2']): (row['bond_length'], row['force_constant']) for index, row in df.iterrows()}

    # Convert bead_info to a dictionary for name lookup
    bead_dict = {number: name for name, number in bead_info}

    with open(itp_file_path, 'r') as file:
        lines = file.readlines()

    bonds_start, bonds_end = None, None
    for i, line in enumerate(lines):
        if '[ bonds ]' in line or '[bonds]' in line:
            bonds_start = i + 1
        elif bonds_start and line.startswith('['):
            bonds_end = i
            break

    if bonds_start is None:
        bonds_start = bonds_end = len(lines)
        lines.append("[bonds]\n")
    else:
        # Ensure no extra newline before appending new bonds
        if lines[bonds_end - 1].strip() == "":
            bonds_end -= 1

###Insertion of hardcoded bond###########
        
    # Find the lowest bead number from bead_info
    lowest_bead_number = min(bead_info, key=lambda x: x[1])[1]

    # Hardcoded bond details
    hardcoded_bond_length = 0.356
    hardcoded_force_constant = 1000

    # Add the hardcoded bond line
    hardcoded_bond_line = f"  14{lowest_bead_number:>6}     1  {hardcoded_bond_length:>4.3f}  {hardcoded_force_constant:>7}\n"
    lines.insert(bonds_end, hardcoded_bond_line)
    bonds_end += 1  # Update the bonds_end index

#########################################
    
###Dictionary lookup ####################    
    # Add new bonds for beads in bead_info
    for i in range(len(bead_info) - 1):
        current_bead_number = bead_info[i][1]
        next_bead_number = bead_info[i + 1][1]

        current_bead_name = bead_dict.get(current_bead_number)
        next_bead_name = bead_dict.get(next_bead_number)


        # Look for bond information between current bead name and next bead name
        if current_bead_name and next_bead_name:
            bond_key = (current_bead_name, next_bead_name)


            if bond_key in bond_dict:
                length, constant = bond_dict[bond_key]
                new_bond_line = f"{current_bead_number:>4}{next_bead_number:>6}     1  {length:>4.3f}  {constant:>7}\n"
                lines.insert(bonds_end, new_bond_line)
##########################################
    # Write to a new file
    with open(output_file_path, 'w') as file:
        file.writelines(lines)

# Example usage
csv_file_path = "bonds_bead-matrix.csv"
itp_file_path = "backbone_modified_atoms.itp"
output_file_path = f'backbone_modified_{bead_names_str}.itp' # Example bead_info
update_itp_with_bond_info(csv_file_path, itp_file_path, output_file_path, bead_info)

########################Creation of  minimization Topology file ######################
# Define the content of the topology file
topology_content = f"""
#include "martini_v3.0.0.itp"
#include "martini_v3.0.0_phospholipids_v1.itp"
#include "martini_v3.0.0_solvents_v1.itp"
#include "martini_v3.0.0_ions_v1.itp"
#include "{output_file_path}"

[ system ]
DPPC BILAYER SELF-ASSEMBLY in water

[ molecules ]
BCK 1
"""

# Path for the topology file
topology_file_path = f"top_{bead_names_str}.top"

# Writing the topology file
with open(topology_file_path, "w") as file:
    file.write(topology_content)

########################Setup Minimization#############################

# Bash script content
bash_script_content = f"""#!/bin/bash

# GROMACS simulation command
# Replace this with the actual command you need to run
gmx grompp -f minimization.mdp -c {modified_file_path} -p top_{bead_names_str}.top -o minim_{bead_names_str}.tpr 
gmx mdrun -v -deffnm minim_{bead_names_str}
gmx solvate -cp minim_{bead_names_str}.gro -cs water.gro -radius 0.21  -o solvated_{bead_names_str}.gro -p top_{bead_names_str}.top
gmx grompp -p top_{bead_names_str}.top -c solvated_{bead_names_str}.gro -f minimization.mdp -o minimization_{bead_names_str}.tpr
gmx mdrun -deffnm minimization_{bead_names_str} -v
"""

# Path for the bash script
bash_script_path = "minimization.sh"

# Writing the bash script
with open(bash_script_path, "w") as file:
    file.write(bash_script_content)

# Make the script executable
os.chmod(bash_script_path, 0o755)

##############################Running minim##############################

# Path to the Bash script
bash_script_path = 'minimization.sh'

# Path to the log file
log_file_path = f'{bead_names_str}.log'

# Open the log file in write mode
with open(log_file_path, 'w') as log_file:
    # Running the Bash script and redirecting stdout and stderr to the log file
    process = subprocess.Popen(['bash', bash_script_path], stdout=log_file, stderr=subprocess.STDOUT, text=True)

    # Wait for the process to complete
    process.wait()

print(f"Script output has been saved to {log_file_path}")


########################Creation of Double membrane Topology file ######################
# Define the content of the topology file
topology_content = f"""
#include "martini_v3.0.0.itp"
#include "martini_v3.0.0_phospholipids_v1.itp"
#include "martini_v3.0.0_solvents_v1.itp"
#include "martini_v3.0.0_ions_v1.itp"
#include "{output_file_path}"

[ system ]
DPPC BILAYER SELF-ASSEMBLY in water

[ molecules ]
DPPC   300
BCK 1
"""

# Path for the topology file
topology_file_path = f"top_{bead_names_str}_double_membrane.top"

# Writing the topology file
with open(topology_file_path, "w") as file:
    file.write(topology_content)

############################Importing the polymer into the double membrane setup#################################

# Bash script content
bash_script_content = f"""#!/bin/bash

# GROMACS simulation command
gmx insert-molecules -f DPPC_10x10.gro -ci minim_{bead_names_str}.gro -o dppc_{bead_names_str}.gro -nmol 1 -try 5000 -selrpos atom
gmx grompp -p top_{bead_names_str}_double_membrane.top -f minimization.mdp -c dppc_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr
gmx mdrun -deffnm minimization-vac_{bead_names_str} -v
gmx grompp -p top_{bead_names_str}_double_membrane.top -f minimization.mdp -c minimization-vac_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr

"""

# Path for the bash script
bash_script_path = "setup_double_membrane.sh"

# Writing the bash script
with open(bash_script_path, "w") as file:
    file.write(bash_script_content)

# Make the script executable
os.chmod(bash_script_path, 0o755)

##############################Running insertion into double membrane setup ##############################

# Path to the Bash script
bash_script_path = 'setup_double_membrane.sh'

# Path to the log file
log_file_path = f'{bead_names_str}_setup_double_membrane.log'

# Open the log file in write mode
with open(log_file_path, 'w') as log_file:
    # Running the Bash script and redirecting stdout and stderr to the log file
    process = subprocess.Popen(['bash', bash_script_path], stdout=log_file, stderr=subprocess.STDOUT, text=True)

    # Wait for the process to complete
    process.wait()

print(f"Script output has been saved to {log_file_path}")

log_file_path = f'{bead_names_str}_setup_double_membrane.log'  # Replace with your actual log file path


def extract_charge_from_log(file_path):
    charge = None
    with open(file_path, 'r') as file:
        for line in file:
            if "System has non-zero total charge:" in line:
                # Extract the number following the expression
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        charge = float(parts[1].strip())
                        break
                    except ValueError:
                        print("Error: Unable to convert the charge to a number.")
                        break
    return charge

# Extract the charge
total_charge = extract_charge_from_log(log_file_path)
total_charge = int(abs(total_charge))
if total_charge is not None:
    print(f"Total charge extracted from the log file: {total_charge}")
else:
    print("Total charge not found in the log file.")

############################Charge neutralisation and solvations#################################

# Bash script content
bash_script_content = f"""#!/bin/bash

# GROMACS simulation command
gmx insert-molecules -f DPPC_10x10.gro -ci minim_{bead_names_str}.gro -o dppc_{bead_names_str}.gro -nmol 1 -try 5000 -selrpos atom
gmx grompp -p top_{bead_names_str}_double_membrane.top -f minimization.mdp -c dppc_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr
gmx mdrun -deffnm minimization-vac_{bead_names_str} -v
gmx insert-molecules -f minimization-vac_{bead_names_str}.gro -ci CL.pdb -o {bead_names_str}_CL.gro -nmol {total_charge}
sed -i '$s/.*/Your new line here/' yourfile.gro
echo "CL   {total_charge}" >> top_{bead_names_str}_double_membrane.top
gmx grompp -p top_{bead_names_str}_double_membrane.top -f minimization.mdp -c {bead_names_str}_CL.gro -o minimization-vac2_{bead_names_str}.tpr
gmx mdrun -deffnm minimization-vac2_{bead_names_str} -v
gmx solvate -cp minimization-vac2_{bead_names_str}.gro -cs water.gro -radius 0.21  -o solvated_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top
gmx grompp -p top_{bead_names_str}_double_membrane.top -c solvated_{bead_names_str}.gro -f minimization.mdp -o minimization_{bead_names_str}.tpr
gmx mdrun -deffnm minimization_{bead_names_str} -v
gmx grompp -f martini_md.mdp -c minimization_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top -o dppc_md_{bead_names_str}.tpr -maxwarn 2
"""

# Path for the bash script
bash_script_path = "charge_neutralisation+solvation.sh"

# Writing the bash script
with open(bash_script_path, "w") as file:
    file.write(bash_script_content)

# Make the script executable
os.chmod(bash_script_path, 0o755)

##############################Running charge setup##############################

# Path to the Bash script
bash_script_path = "charge_neutralisation+solvation.sh"

# Path to the log file
log_file_path = f'{bead_names_str}_charge_setup.log'

# Open the log file in write mode
with open(log_file_path, 'w') as log_file:
    # Running the Bash script and redirecting stdout and stderr to the log file
    process = subprocess.Popen(['bash', bash_script_path], stdout=log_file, stderr=subprocess.STDOUT, text=True)

    # Wait for the process to complete
    process.wait()

############################Preparing and running plumed setup#################################

# Define the content of the topology file
end_mol=3601+(len(bead_info)+28)
plumed_content = f"""
# a point which is on the line connecting atoms 1 and 10, so that its distance
# from 10 is twice its distance from 1:
c1: CENTER ATOMS=1-3600
c2: CENTER ATOMS=3601-{end_mol} MASS

d2c: DISTANCE ATOMS=c1,c2 COMPONENTS
 

MOVINGRESTRAINT ...
# also notice that a LABEL keyword can be used and is equivalent
# to adding the name at the beginning of the line with colon, as we did so far
  LABEL=res
  ARG=d2c.z
  STEP0=0 AT0=10 KAPPA0=200
  STEP1=10000 AT1=9 KAPPA1=200
  STEP2=100000 AT2=8 KAPPA2=200
  STEP3=200000 AT3=7 KAPPA3=200
  STEP4=300000 AT4=6 KAPPA4=200
  STEP5=400000 AT5=5 KAPPA5=200
  STEP6=500000 AT6=4 KAPPA6=200
  STEP7=600000 AT7=3 KAPPA7=200
  STEP8=700000 AT8=2 KAPPA8=200
  STEP9=800000 AT9=1 KAPPA9=200
  STEP10=900000 AT10=0 KAPPA10=200
  STEP11=1000000 AT11=-1 KAPPA11=200
  STEP12=1100000 AT12=-2 KAPPA12=200
  STEP13=1200000 AT13=-3 KAPPA13=200
  STEP14=1300000 AT14=-4 KAPPA14=200
  STEP15=1400000 AT15=-5 KAPPA15=200
  STEP16=1500000 AT16=-6 KAPPA16=200
  STEP17=1600000 AT17=-7 KAPPA17=200
  STEP18=1700000 AT18=-8 KAPPA18=200
  STEP19=1800000 AT19=-9 KAPPA19=200
  STEP20=1900000 AT20=-10.5 KAPPA20=200
...
PRINT ARG=d2c.z,res.work FILE=COLVAR_{bead_names_str} STRIDE=1000

PRINT ARG=res.work FILE=WORK_{bead_names_str} STRIDE=1000

COMMITTOR ...
  ARG=d2c.z
  STRIDE=10
  BASIN_LL1=-10.5
  BASIN_UL1=-10.1
...
"""

# Path for the topology file
plumed_file_path = f"plumed_{bead_names_str}.dat"

# Writing the topology file
with open(plumed_file_path, "w") as file:
    file.write(plumed_content)    

# Bash script content
bash_script_content = f"""#!/bin/bash

# GROMACS simulation command
rm *step*
rm *#*
gmx_plu grompp -f martini_md.mdp -c minimization_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top -o plumed_{bead_names_str}.tpr -maxwarn 2
gmx_plu mdrun -deffnm plumed_{bead_names_str}  -plumed plumed_{bead_names_str}.dat -nt 15
"""
# Path for the bash script
bash_script_path = "plumed_run.sh"

# Writing the bash script
with open(bash_script_path, "w") as file:
    file.write(bash_script_content)

# Make the script executable
os.chmod(bash_script_path, 0o755)

##############################Running plumed run setup##############################

# Path to the Bash script
bash_script_path = "plumed_run.sh"

# Path to the log file
log_file_path = f'{bead_names_str}_plumed_run.log'

# Open the log file in write mode
with open(log_file_path, 'w') as log_file:
    # Running the Bash script and redirecting stdout and stderr to the log file
    process = subprocess.Popen(['bash', bash_script_path], stdout=log_file, stderr=subprocess.STDOUT, text=True)

    # Wait for the process to complete
    process.wait()


################################Analysis of run##################################
    
def calculate_means(data):
    mean_first_50000 = np.mean(data[:50, 1])
    mean_last_50000 = np.mean(data[-50:, 1])
    mean_difference = mean_last_50000 - mean_first_50000
    return mean_first_50000, mean_last_50000, mean_difference


def plot_data_with_means(x_data, y_data, mean_first, mean_last,mean_diff,title):
    plt.plot(x_data, y_data, label=f'{title} Work: {mean_diff:.2f} kJ/mol')
    plt.axhline(y=mean_first, color='green', linestyle='--')
    plt.axhline(y=mean_last, color='blue', linestyle='--')

# Load and process the first file
data_1 = np.loadtxt(f'WORK_{bead_names_str}')
data_1 = data_1[20::10]  # Take every 100th data point after the first 2000
mean_of_next_2000 = np.mean(data_1[:20, 1])
data_1[:, 1] -= mean_of_next_2000 
mean_first_1, mean_last_1,mean_diff_1  = calculate_means(data_1)

# Load and process the second file - Replace 'path_to_second_file' with the actual file path
data_2 = np.loadtxt('WORK_SC1')
data_2 = data_2[20::10]  # Take every 100th data point after the first 2000
mean_of_next_2000 = np.mean(data_2[:20, 1])
data_2[:, 1] -= mean_of_next_2000 
mean_first_2, mean_last_2,mean_diff_2= calculate_means(data_2)

# Plotting
plt.figure(figsize=(12, 8))
plot_data_with_means(data_1[:, 0], data_1[:, 1], mean_first_1, mean_last_1,mean_diff_1, f'Backbone_{bead_names_str}')
plot_data_with_means(data_2[:, 0], data_2[:, 1], mean_first_2, mean_last_2,mean_diff_2, 'Backbone_SC1')

plt.xlabel('Time (ps)')
plt.ylabel('Work (kJ/mol)')
plt.title('PEI vs. Backbone membrane penetration')
plt.legend()
plt.savefig("membranecomparison.png")
plt.grid(True)
plt.show()

# File to store the data
file_path_results = 'mean_diff_data.txt'

# Writing mean_diff_2 to the file
with open(file_path_results, 'a') as file_result:
    file_result.write(f'Work for:{bead_names_str}:{mean_diff_1:.3f} kJ/mol\n')

