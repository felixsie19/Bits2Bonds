 
import pandas as pd
import os 
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import sys
import textwrap
import argparse
import random
import pickle
import gc
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import signal
import psutil
import shutil
import traceback
import signal
from multithread import kill_processes_by_name


##################Set Name of Gromacs Binary##########################
binary_name="gmx_plu"
################## Creation of directory and defining the directory name################

with open('./data/DFfromRL.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)
print(df.columns)

# Initialize or set default values
lipo_beads = None
hydro_beads = None

# Attempt to assign primary choice values
lipo_beads_primary = df[df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo"]['beads_lipo_ph_8'].iloc[0]
hydro_beads_primary = df[df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro"]['beads_hydro_ph_8'].iloc[0]

# Check if primary choices are NaN and assign fallback values if necessary
if df[df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo"]['beads_lipo_ph_8'].isna().all():
    lipo_beads = df[df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo"]['beads_lipo'].iloc[0]
else:
    lipo_beads = lipo_beads_primary


if df[df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro"]['beads_hydro_ph_8'].isna().all():
    hydro_beads = df[df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro"]['beads_hydro'].iloc[0]
else:
    hydro_beads = hydro_beads_primary


lipo_dict = {}
hydro_dict = {}
bead_insertion_info_ph_8={}
collected = gc.collect()
 


# Check conditions and modify keys for lipo_dict

if lipo_beads[0] == 'SQ2p':
    lipo_dict['SQ2p_l'] = lipo_beads[1:]  # Drops the first element if it's 'SQ2p' and stores the rest under 'SQ2p'

else:
    lipo_dict['SN4'] = lipo_beads[1:]  # Stores all except the first under 'SN4'

# Check conditions and modify keys for hydro_dict
if hydro_beads[0] == 'SQ2p':
    hydro_dict['SQ2p_h'] = hydro_beads[1:]  # Same handling for hydro beads
else:
    hydro_dict['N6d'] = hydro_beads[1:]  # Stores all except the first under 'N6d'



# Merging the two dictionaries, hydro_dict values will overwrite lipo_dict values for the same key
bead_insertion_info_ph_8 = lipo_dict | hydro_dict

# Print the merged dictionary
bead_insertion_info_ph_8 = {key: [item for item in value if item is not None] for key, value in bead_insertion_info_ph_8.items()}
lipo_beads=df[df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo"]['beads_lipo_ph_4'].iloc[0] 
hydro_beads=df[df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro"]['beads_hydro_ph_4'].iloc[0]  

lipo_beads = None
hydro_beads = None

# Attempt to assign primary choice values
lipo_beads_primary = df[df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo"]['beads_lipo_ph_4'].iloc[0]
hydro_beads_primary = df[df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro"]['beads_hydro_ph_4'].iloc[0]


# Check if primary choices are NaN and assign fallback values if necessary

if df[df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo"]['beads_lipo_ph_8'].isna().all():
    lipo_beads = df[df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo"]['beads_lipo'].iloc[0]
else:
    lipo_beads = lipo_beads_primary

if df[df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro"]['beads_hydro_ph_4'].isna().all():
    hydro_beads = df[df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro"]['beads_hydro'].iloc[0]
else:
    hydro_beads = hydro_beads_primary

lipo_dict = {}
hydro_dict = {}
bead_insertion_info_ph_4={}
# Check conditions and modify keys for lipo_dict

if lipo_beads[0] == 'SQ2p':
    lipo_dict['SQ2p_l'] = lipo_beads[1:]  # Drops the first element if it's 'TQ2p' and stores the rest under 'TQ2p'

else:
    lipo_dict['SN4'] = lipo_beads[1:]  # Stores all except the first under 'SN4'

# Check conditions and modify keys for hydro_dict
if hydro_beads[0] == 'SQ2p':
    hydro_dict['SQ2p_h'] = hydro_beads[1:]  # Same handling for hydro beads
else:
    hydro_dict['N6d'] = hydro_beads[1:]  # Stores all except the first under 'N6d'

bead_insertion_info_ph_4 = lipo_dict | hydro_dict

# Filter out None values from each list in the dictionary
bead_insertion_info_ph_8 = {key: [item for item in value if item is not None] for key, value in bead_insertion_info_ph_8.items()}
bead_insertion_info_ph_4 = {key: [item for item in value if item is not None] for key, value in bead_insertion_info_ph_4.items()}



bead_names_str = "_".join([f"{key}+{'_'.join(map(str, filter(None, vals)))}" for key, vals in bead_insertion_info_ph_8.items() if isinstance(vals, list)])



#
# Now you can use hydro_beads and lipo_beads as lists

start_number = 44
for key, beads in bead_insertion_info_ph_8.items():
    bead_insertion_info_ph_8[key] = [(start_number + i, bead) for i, bead in enumerate(beads)]
    start_number += len(beads)

start_number = 44
for key, beads in bead_insertion_info_ph_4.items():
    bead_insertion_info_ph_4[key] = [(start_number + i, bead) for i, bead in enumerate(beads)]
    start_number += len(beads)

bead_names_str = "_".join([f"{key}+{'_'.join(bead[1] for bead in vals if bead and len(bead) > 1 and bead[1] is not None)}" 
                           for key, vals in bead_insertion_info_ph_8.items() if isinstance(vals, list)])



# Check if the directory exists
if not os.path.exists(f"./data/{bead_names_str}"):
    # If it doesn't exist, create the directory
    os.makedirs(f"./data/{bead_names_str}")

os.chdir(f"./data/{bead_names_str}")

class FileGenerator:
    def __init__(self, bead_insertion_info):
        self.bead_insertion_info_ph_8 = bead_insertion_info
        #self.start_number = 29
        

    ################## Creation of .gro file ##################
    print("Starting .gro and .itp file generation")
    def insert_beads_before_last_line(self, file_path,bead_insertion_info,ph,initial_distance=0.5,):
        keys_list = list(bead_insertion_info.keys())
        new_lines = []
        bead_info = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
        updated_atom_count = int(lines[1].strip())  # Initial atom count
        new_lines.extend(lines[:-1])  # Keep all lines except the last
        ### Check charge and adapt the gro file accordingly
        print(f"Target bead keys:{bead_insertion_info}, New lines:{new_lines}")

        for target_bead_key, new_bead_names in bead_insertion_info.items():
            for i, line in enumerate(new_lines):
                if "0UNK   SN4    15" in line and target_bead_key == "SQ2p_l":
                    new_lines[i] = line.replace(' SN4', 'SQ2p')
                    target_lipo_index=i
                if "0UNK   N6d    29" in line and target_bead_key == "SQ2p_h":
                    new_lines[i] = line.replace(' N6d', 'SQ2p')
                    target_hydro_index=i
                if "0UNK   SN4    15" in line and target_bead_key == "SN4":
                    target_lipo_index=i
                if "0UNK   N6d    29" in line and target_bead_key == "N6d":
                    target_hydro_index=i

        for target_bead_key, new_bead_names in bead_insertion_info.items():
            current_distance = initial_distance
            if target_bead_key == "SQ2p_l":
                target_bead="SQ2p"
                current_distance, updated_atom_count = self.insert_beads(target_bead,target_lipo_index, new_bead_names, current_distance, updated_atom_count, new_lines, bead_info,initial_distance)
            if target_bead_key == "SN4":
                target_bead="SN4"
                current_distance, updated_atom_count = self.insert_beads(target_bead,target_lipo_index,new_bead_names, current_distance, updated_atom_count, new_lines, bead_info,initial_distance)                
           
            if target_bead_key == "SQ2p_h":
                target_bead="SQ2p"
                current_distance, updated_atom_count = self.insert_beads(target_bead,target_hydro_index,new_bead_names, current_distance, updated_atom_count, new_lines, bead_info,initial_distance)
            
            if target_bead_key == "N6d":
                target_bead="N6d"
                current_distance, updated_atom_count = self.insert_beads(target_bead,target_hydro_index,new_bead_names, current_distance, updated_atom_count, new_lines, bead_info,initial_distance)
                                    
        new_lines.append(lines[-1])  # Append the last line
        new_lines[1] = f"{updated_atom_count}\n"  # Update atom count

        new_file_path = f"modified_{bead_names_str}_pH_{ph}.gro"
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(new_lines)
        return new_file_path, bead_info, bead_names_str

    def insert_beads(self,target_bead, target_index, new_bead_names, current_distance, updated_atom_count, new_lines, bead_info,initial_distance):
       
        for i, line in enumerate(new_lines):
            split_line = line.split()
            if len(split_line) > 4:  # Ensuring the line has enough parts to check
                line_bead = split_line[1]  # Assuming the bead name is the second element
                line_index = int(split_line[2])  # Assuming the index is the third element
                if line_index == target_index-1 and line_bead == target_bead:
                    found_target = True
                    x, y, z = map(float, split_line[3:6])  # Coordinates are in the fourth, fifth, and sixth positions
                    res_name = "0UNK"
                    atom_type = int(split_line[2])
                    break
        else:
            return current_distance, updated_atom_count  # Target bead not found

        for new_bead_name in new_bead_names:
            updated_atom_count += 1
            bead_info[new_bead_name] = updated_atom_count
            new_x, new_y, new_z = x, y + current_distance, z
            current_distance += initial_distance
            bead_line = f"{res_name:>8}{new_bead_name[1]:>7}{updated_atom_count:>5}{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}\n"
            new_lines.append(bead_line)

        return current_distance, updated_atom_count


    ########################################adaption of .itp atom list##################################

    def add_new_beads_to_itp(self,itp_file_path, bead_insertion_info, output_file_path):
        charged_atom_types = ['TQ4p', 'SQ3p', 'SQ2p']
        negative_charged_atom_type=['SQ5n']
        with open(itp_file_path, 'r') as file:
            lines = file.readlines()
        ################################# change anchor_beads if charged

        atoms_start, atoms_end = None, None
        for i, line in enumerate(lines):
            if '[ atoms ]' in line or '[atoms]' in line:
                atoms_start = i + 1
            elif 'SQ2p_l' in bead_insertion_info and '15     SN4' in line:
                lines[i] = line.replace(' SN4', 'SQ2p')
            elif 'SQ2p_h' in bead_insertion_info and '29     N6d' in line:
                lines[i] = line.replace(' N6d', 'SQ2p')            
            elif atoms_start and line.startswith('['):
                atoms_end = i-2
                break

        for target_bead, bead_list in bead_insertion_info.items():
            for bead in bead_list:
                number, name = bead
                if name in charged_atom_types: 
                    charge = 1
                elif name in negative_charged_atom_type:
                    charge = -1 
                else:
                    charge = 0
                
                new_line = f"{number:>8}{name[0:4]:>8}      1{name[0:4]:>7}{name[0:4]:>6}{number:>6}{charge:>7}\n" 



                lines.insert(atoms_end, new_line)
                atoms_end += 1 
        with open(output_file_path, 'w') as file:
            file.writelines(lines)

    ###########################################adaptation of .itp file [bond] section###############################

    def update_itp_with_bond_info(self,csv_file_path, itp_file_path, output_file_path, bead_insertion_info):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        # Convert DataFrame to a dictionary for easier lookup
        bond_dict = {(row['atom1'], row['atom2']): (row['bond_length'], row['force_constant']) 
                     for index, row in df.iterrows()}
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
        # Hardcoded bond details
        hardcoded_bond_length = 0.356
        hardcoded_force_constant = 1000
        # Find the lowest bead number for each key in bead_insertion_info_ph_8
        lowest_bead_numbers = {key: min(bead_list, key=lambda x: x[0])[0] for key, bead_list in bead_insertion_info.items() if bead_list}
        print(f"Bead list: {bead_insertion_info}")
        first_key = list(bead_insertion_info.keys())[0]
        second_key = list(bead_insertion_info.keys())[1]
        print(first_key)
        print(second_key)
        # Check if we have values for both keys
        if first_key in lowest_bead_numbers:
            print(f"Lowest bead numbers:{lowest_bead_numbers}")
            first_key_lowest_bead_number = lowest_bead_numbers.get(first_key)
            print(f"First key found:{first_key_lowest_bead_number}")
            hardcoded_bond_line_first = f"  14{first_key_lowest_bead_number:>6}     1  {hardcoded_bond_length:>5.3f}  {hardcoded_force_constant:>7}\n"
            lines.insert(bonds_end, hardcoded_bond_line_first)
            bonds_end += 1

        if second_key in lowest_bead_numbers:
            second_key_lowest_bead_number = lowest_bead_numbers.get(second_key)
            print(f"Second key found:{second_key_lowest_bead_number}")
            hardcoded_bond_line_second = f"  28{second_key_lowest_bead_number:>6}     1  {hardcoded_bond_length:>5.3f}  {hardcoded_force_constant:>7}\n"
            lines.insert(bonds_end, hardcoded_bond_line_second)
            bonds_end += 1
        else:
            # Do nothing if the second chain is empty
            pass
        # Add new bonds for beads in bead_insertion_info_ph_8
        for target_bead, bead_list in bead_insertion_info.items():
            for i in range(len(bead_list) - 1):
                current_bead_number, current_bead_name = bead_list[i]

                next_bead_number, next_bead_name = bead_list[i + 1]

                # Look for bond information between current bead name and next bead name
                bond_key = (current_bead_name, next_bead_name)
                if bond_key in bond_dict:
                    length, constant = bond_dict[bond_key]
                    new_bond_line = f"{current_bead_number:>4}{next_bead_number:>6}     1  {length:>4.3f}  {constant:>7}\n"
                    lines.insert(bonds_end, new_bond_line)
                    bonds_end += 1  # Update the end index as we are adding lines

        # Write to a new file
        with open(output_file_path, 'w') as file:
            file.writelines(lines)
    ###########################################adaptation of .itp file [angles] section###############################        

    def update_itp_with_hardcoded_angle_info(self,itp_file_path, output_file_path, bead_insertion_info):
        """
        Updates an ITP file with a hardcoded angle of 120 degrees and a force constant of 1000.

        Finds the [ angles ] section in the ITP file and adds/updates angle entries
        based on bead insertion information, always using the hardcoded values.

        Args:
            itp_file_path (str): Path to the original ITP file.
            output_file_path (str): Path to save the modified ITP file.
            bead_insertion_info (dict): A dictionary where keys are target beads (not used
                directly in the angle section, but used for determining where to insert
                hardcoded angles between chains) and values are lists of
                (bead_number, bead_name) tuples, representing the inserted beads.
        """

        # --- 1. Hardcoded Angle Values ---
        hardcoded_angle = 120.0
        hardcoded_force_constant = 750.0

        # --- 2. Read and Prepare ITP File ---
        with open(itp_file_path, 'r') as file:
            lines = file.readlines()

        # --- 3. Locate [ angles ] Section ---
        angles_start, angles_end = None, None
        for i, line in enumerate(lines):
            if '[ angles ]' in line or '[angles]' in line:
                angles_start = i + 1
            elif angles_start and line.startswith('['):
                angles_end = i
                break

        # If [ angles ] section doesn't exist, create it.
        if angles_start is None:
            angles_start = angles_end = len(lines)
            lines.append("[ angles ]\n")
        else:
            # Ensure no extra newline before appending new angles
            if lines[angles_end - 1].strip() == "":
                angles_end -= 1

        # --- 4. Insert Hardcoded Angles Between Chains ---
        # Find lowest bead numbers for each key (chain)
        lowest_bead_numbers = {
            key: min(bead_list, key=lambda x: x[0])[0]
            for key, bead_list in bead_insertion_info.items() if bead_list
        }
        first_key = list(bead_insertion_info.keys())[0]
        second_key = list(bead_insertion_info.keys())[1]

        # Insert hardcoded angles
        if first_key in lowest_bead_numbers:
            first_key_lowest_bead_number = lowest_bead_numbers.get(first_key)
            # Angle between bead 14, first bead of first chain, and *next* bead in first chain
            if len(bead_insertion_info[first_key]) > 1:  # Check if there's a next bead
                next_bead_number = bead_insertion_info[first_key][1][0]
                hardcoded_angle_line_first = (
                    f"  14{first_key_lowest_bead_number:>6}{next_bead_number:>6}     1"
                    f"  {hardcoded_angle:>5.1f}  {hardcoded_force_constant:>7}\n"
                )
                lines.insert(angles_end, hardcoded_angle_line_first)
                angles_end += 1

        if second_key in lowest_bead_numbers:
            second_key_lowest_bead_number = lowest_bead_numbers.get(second_key)
            # Angle between bead 28, first bead of second chain, and *next* bead in second chain
            if len(bead_insertion_info[second_key]) > 1: # Check if there's a next bead
                next_bead_number = bead_insertion_info[second_key][1][0]
                hardcoded_angle_line_second = (
                    f"  28{second_key_lowest_bead_number:>6}{next_bead_number:>6}     1"
                    f"  {hardcoded_angle:>5.1f}  {hardcoded_force_constant:>7}\n"
                )
                lines.insert(angles_end, hardcoded_angle_line_second)
                angles_end += 1

        # --- 5. Add Hardcoded Angles Within Chains ---
        for target_bead, bead_list in bead_insertion_info.items():
            for i in range(len(bead_list) - 2):  # Iterate up to the third-to-last bead
                bead1_number, _ = bead_list[i]
                bead2_number, _ = bead_list[i + 1]
                bead3_number, _ = bead_list[i + 2]

                new_angle_line = (
                    f"{bead1_number:>4}{bead2_number:>6}{bead3_number:>6}     1"
                    f"  {hardcoded_angle:>5.1f}  {hardcoded_force_constant:>7}\n"
                )
                lines.insert(angles_end, new_angle_line)
                angles_end += 1

        # --- 6. Write Modified ITP File ---
        with open(output_file_path, 'w') as file:
            file.writelines(lines)

########################################################################################################
#                                                                                                      #
#                                                                                                      #
#                               Start of Challenges                                                    #
#                                                                                                      #
########################################################################################################
    

def double_membrane(bead_names_str,modified_file_path,double_membrane_file):
    try:
        ########################Creation of  minimization Topology file ######################
        print("Starting Double Membrane Challenge")
        # Define the content of the topology file
        topology_content = f"""
        #include "../../requirements/martini_v3.0.0.itp"
        #include "../../requirements/martini_v3.0.0_phospholipids_v1.itp"
        #include "../../requirements/martini_v3.0.0_solvents_v1.itp"
        #include "../../requirements/martini_v3.0.0_ions_v1.itp"
        #include "backbone_modified_{bead_names_str}_pH_4.itp"

        [ system ]
        DPPC BILAYER SELF-ASSEMBLY in water

        [ molecules ]
        BCK 1
        """

        # Path for the topology file
        topology_file_path = f"top_{bead_names_str}_pH_8.top"

        # Writing the topology file
        with open(topology_file_path, "w") as file:
            file.write(topology_content)


        ########################Setup Minimization#############################

        # Bash script content
        bash_script_content = textwrap.dedent(f"""#!/bin/bash

        # GROMACS simulation command
        # Replace this with the actual command you need to run
        {binary_name} grompp -f ../../requirements/minimization.mdp -c {modified_file_path} -p top_{bead_names_str}_pH_8.top -o minim_{bead_names_str}.tpr 
        {binary_name} mdrun -v -deffnm minim_{bead_names_str} -nt 4
        {binary_name} solvate -cp minim_{bead_names_str}.gro -cs ../../requirements/water.gro -radius 0.21  -o solvated_{bead_names_str}.gro -p top_{bead_names_str}_pH_8.top
        {binary_name} grompp -p top_{bead_names_str}_pH_8.top -c solvated_{bead_names_str}.gro -f ../../requirements/minimization.mdp -o minimization_{bead_names_str}.tpr
        {binary_name} mdrun -deffnm minimization_{bead_names_str} -v -nt 4 
        """)

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




        ########################Creation of Double membrane Topology file ######################
        # Define the content of the topology file
        topology_content = f"""

        #define RUBBER_BANDS
        #include "../../requirements/martini_v3.0.0.itp"
        #include "../../requirements/martini_v3.0.0_ions_v1.itp"
        #include "../../requirements/martini_v3.0.0_solvents_v1.itp"
        #include "backbone_modified_{bead_names_str}_pH_4.itp"

        [ system ]
        ; name
        logP challenge in water
        [ molecules ]
        ; name  number
        HD 3501
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
        {binary_name} insert-molecules -f ../../requirements/hexadecane-md2.gro -ci minim_{bead_names_str}.gro -o dppc_{bead_names_str}.gro -nmol 1 -try 5000 -selrpos atom -ip ../../requirements/positions.dat 
        {binary_name} grompp -p top_{bead_names_str}_double_membrane.top -f ../../requirements/minimization.mdp -c dppc_{bead_names_str}.gro  -r dppc_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr -maxwarn 2
        {binary_name} mdrun -deffnm minimization-vac_{bead_names_str} -v -nt 4

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


        log_file_path = f'{bead_names_str}_setup_double_membrane.log'  # Replace with your actual log file path


        def extract_charge_from_log(file_path):
            charge = 0
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
            print(f"This is the charge: {charge}")
            return charge

        # Extract the charge
        total_charge = extract_charge_from_log(log_file_path)
        
        total_charge = int(abs(total_charge))

        ############################Charge neutralisation and solvations#################################

        if total_charge!=0:

            # Bash script content
            bash_script_content = f"""#!/bin/bash

            # GROMACS simulation command
            {binary_name} grompp -p top_{bead_names_str}_double_membrane.top -f ../../requirements/minimization.mdp -c minimization-vac_{bead_names_str}.gro -r minimization-vac_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization-vac_{bead_names_str} -v -nt 4
            {binary_name} insert-molecules -f minimization-vac_{bead_names_str}.gro -ci ../../requirements/CL.pdb -o {bead_names_str}_CL.gro -nmol {total_charge}
            echo "CL   {total_charge}" >> top_{bead_names_str}_double_membrane.top
            {binary_name} grompp -p top_{bead_names_str}_double_membrane.top -f ../../requirements/minimization.mdp -c {bead_names_str}_CL.gro  -r {bead_names_str}_CL.gro -o minimization-vac2_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization-vac2_{bead_names_str} -v -nt 4
            {binary_name} solvate -cp minimization-vac2_{bead_names_str}.gro -cs ../../requirements/water.gro -radius 0.21  -o solvated_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top
            {binary_name} grompp -p top_{bead_names_str}_double_membrane.top -c solvated_{bead_names_str}.gro -r solvated_{bead_names_str}.gro -f ../../requirements/minimization.mdp -o minimization_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization_{bead_names_str} -v -nt 4
            {binary_name} grompp -f ../../requirements/martini_md.mdp -c minimization_{bead_names_str}.gro -r minimization_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top -o dppc_md_{bead_names_str}.tpr -maxwarn 2
            """
        else:
                        # Bash script content
            bash_script_content = f"""#!/bin/bash

            # GROMACS simulation command
            {binary_name} grompp -p top_{bead_names_str}_double_membrane.top -f ../../requirements/minimization.mdp -c minimization-vac_{bead_names_str}.gro -r minimization-vac_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization-vac_{bead_names_str} -v -nt 4
            {binary_name} solvate -cp minimization-vac_{bead_names_str}.gro -cs ../../requirements/water.gro -radius 0.21  -o solvated_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top
            {binary_name} grompp -p top_{bead_names_str}_double_membrane.top -c solvated_{bead_names_str}.gro -r solvated_{bead_names_str}.gro -f ../../requirements/minimization.mdp -o minimization_{bead_names_str}.tpr -maxwarn 2
            {binary_name} mdrun -deffnm minimization_{bead_names_str} -v -nt 4
            {binary_name} grompp -f ../../requirements/martini_md.mdp -c minimization_{bead_names_str}.gro -r minimization_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top -o dppc_md_{bead_names_str}.tpr -maxwarn 2
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
        end_mol=14004+(len(beads)+41)
        plumed_content = f"""
        # a point which is on the line connecting atoms 1 and 10, so that its distance
        # from 10 is twice its distance from 1:
        c2: CENTER ATOMS=14005-{end_mol} MASS

        d2c: POSITION ATOM=c2 



        MOVINGRESTRAINT ...
        # also notice that a LABEL keyword can be used and is equivalent
        # to adding the name at the beginning of the line with colon, as we did so far
          LABEL=res
          ARG=d2c.x
          STEP0=0 AT0=0 KAPPA0=100
          STEP1=300000 AT1=5 KAPPA1=100


        ...
        PRINT ARG=res.work FILE={double_membrane_file}  STRIDE=1000
        PRINT ARG=res.bias FILE=BIAS_double_MEMBRANE
        PRINT ARG=res.force2 FILE=FORCE_double_MEMBRANE

        PRINT ARG=d2c.x FILE=COLVAR_double_Membrane STRIDE=1000
        ene: ENERGY 
        

        """

        # Path for the topology file
        plumed_file_path = f"plumed_{bead_names_str}.dat"

        # Writing the topology file
        with open(plumed_file_path, "w") as file:
            file.write(plumed_content)    

        # Bash script content
        bash_script_content = textwrap.dedent(f"""#!/bin/bash

        # GROMACS simulation command
        rm *step*
        rm *#*
        {binary_name} grompp -f ../../requirements/martini_md.mdp -c minimization_{bead_names_str}.gro -r minimization_{bead_names_str}.gro -p top_{bead_names_str}_double_membrane.top -o plumed_{bead_names_str}.tpr -maxwarn 2
        {binary_name} mdrun -deffnm plumed_{bead_names_str}  -plumed plumed_{bead_names_str}.dat -nt 4 -nsteps 125000
        rm *step*
        echo done
        """)
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


        
        # Load and process the first file
        data_1 = np.loadtxt(f'WORK_double_Membrane', usecols=(0, 1))  # Load both columns
        data_1 = data_1[:, 1] 


        # Find the minimum value in the 2nd column
        data1_min = np.min(data_1)

        # Shift the data so that the minimum value is zero
        data_1_shifted = data_1 - data1_min
        np.savetxt('WORK_double_membrane_to_zero', data_1_shifted, fmt='%.6f', newline='\n')

        # Calculate the integrated value of the shifted data
        integrated_data_double_membrane = np.trapezoid(data_1_shifted, axis=0) / 100

        # Save the minimum value to a file

        # Save the integrated value to a file
        file_path_results = '../mean_diff_data_double_membrane.txt'
        # Writing mean_diff_2 to the file
        with open(file_path_results, 'a') as file_result:
            file_result.write(f'{sys.argv[1]},{bead_names_str},{integrated_data_double_membrane :.3f}\n')

        print("Double Membrane Challenge completed")
        # Selecting rows where both conditions are met
        mask = (df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo") & (df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro")

        # Assigning the performance value to a new column 'performance' in the selected rows
        df.loc[mask, 'Performance_double_membrane'] = integrated_data_double_membrane

    except Exception as e:
        logging.error(f"Attempt failed with parameters Error: {e}\n{traceback.format_exc()}")
        return False
    else:
        return True,

def sirna_challenge(bead_names_str,modified_file_path,logfile,pH):
        #################################### Creation of top file of RNA-setup #####################
        print("Starting siRNA Challenge")
        # Define the content of the topology file
        topology_content = f"""
        #include "../../requirements/martini_v3.0.0.itp"
        #include "../../requirements/martini_v3.0.0_solvents_v1.itp"
        #include "../../requirements/martini_v3.0.0_ions_v1.itp"

        #define RUBBER_BANDS

        #include "../../requirements/Nucleic_B+Nucleic_A.itp"
        #ifdef POSRNA
        #include "../../requirements/posre.itp"
        #endif
        #include "backbone_modified_{bead_names_str}_pH_{pH}.itp"
        [ system ]
        ; name
        Martini system from egfp_sirna_meth.pdb in water

        [ molecules ]
        ; name        number
        Nucleic_B+Nucleic_A 	 1
        BCK                    1


        """

        # Path for the topology file
        topology_file_path = f"top_{bead_names_str}_siRNA_challenge.top"

        # Writing the topology file
        with open(topology_file_path, "w") as file:
            file.write(topology_content)    


        ######################################Importing molecule in  RNA-setup ##############


        # Bash script content
        bash_script_content = f"""#!/bin/bash



        {binary_name} insert-molecules -f ../../requirements/RNA-CG.gro -ci {modified_file_path} -o rna_{bead_names_str}.gro -nmol 1 -try 5000 -selrpos atom -ip ../../requirements/positions_rna.dat
        {binary_name} grompp -p top_{bead_names_str}_siRNA_challenge.top -f ../../requirements/minimization.mdp -c rna_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr -maxwarn 1
        {binary_name} mdrun -deffnm minimization-vac_{bead_names_str} -v -nt 4
        {binary_name} grompp -p top_{bead_names_str}_siRNA_challenge.top -f ../../requirements/minimization.mdp -c minimization-vac_{bead_names_str}.gro -o minimization-vac_{bead_names_str}.tpr -maxwarn 1

        """

        # Path for the bash script
        bash_script_path = "setup_siRNA_challenge.sh"

        # Writing the bash script
        with open(bash_script_path, "w") as file:
            file.write(bash_script_content)

        # Make the script executable
        os.chmod(bash_script_path, 0o755)


        ##############################Running insertion into siRNA setup ##############################

        # Path to the Bash script
        bash_script_path = 'setup_siRNA_challenge.sh'

        # Path to the log file
        log_file_path = f'{bead_names_str}_setup_siRNA_challenge.log'

        # Open the log file in write mode
        with open(log_file_path, 'w') as log_file:
            # Running the Bash script and redirecting stdout and stderr to the log file
            process = subprocess.Popen(['bash', bash_script_path], stdout=log_file, stderr=subprocess.STDOUT, text=True)

            # Wait for the process to complete
            process.wait()



        log_file_path = f'{bead_names_str}_setup_siRNA_challenge.log'  # Replace with your actual log file path


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


        ############################################Minimization and charge neutralisation
            # Bash script content
        if total_charge!=0:
        
            bash_script_content = f"""#!/bin/bash


            {binary_name} insert-molecules -f minimization-vac_{bead_names_str}.gro -ci ../../requirements/NA.pdb -o {bead_names_str}_siRNA_CL.gro -nmol {total_charge}
            echo "NA   {total_charge}" >> top_{bead_names_str}_siRNA_challenge.top
            {binary_name} grompp -p top_{bead_names_str}_siRNA_challenge.top -f ../../requirements/minimization.mdp -c {bead_names_str}_siRNA_CL.gro -o minimization-vac2_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization-vac2_{bead_names_str} -v -nt 4
            {binary_name} solvate -cp minimization-vac2_{bead_names_str}.gro -cs ../../requirements/water.gro -radius 0.21  -o solvated_siRNA_{bead_names_str}.gro -p top_{bead_names_str}_siRNA_challenge.top
            {binary_name} grompp -p top_{bead_names_str}_siRNA_challenge.top -c solvated_siRNA_{bead_names_str}.gro -f ../../requirements/minimization.mdp -o minimization_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization_{bead_names_str} -v -nt 4
            {binary_name} grompp -f ../../requirements/equilibration.mdp -c minimization_{bead_names_str}.gro -p top_{bead_names_str}_siRNA_challenge.top -o equilibration_siRNA_{bead_names_str}.tpr -maxwarn 2
            {binary_name} mdrun -deffnm equilibration_siRNA_{bead_names_str} -v -nt 4

            """
        else:
            bash_script_content = f"""#!/bin/bash

            {binary_name} grompp -p top_{bead_names_str}_siRNA_challenge.top -f ../../requirements/minimization.mdp -c minimization-vac_{bead_names_str}.gro -o minimization-vac2_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization-vac2_{bead_names_str} -v -nt 4
            {binary_name} solvate -cp minimization-vac2_{bead_names_str}.gro -cs ../../requirements/water.gro -radius 0.21  -o solvated_siRNA_{bead_names_str}.gro -p top_{bead_names_str}_siRNA_challenge.top
            {binary_name} grompp -p top_{bead_names_str}_siRNA_challenge.top -c solvated_siRNA_{bead_names_str}.gro -f ../../requirements/minimization.mdp -o minimization_{bead_names_str}.tpr
            {binary_name} mdrun -deffnm minimization_{bead_names_str} -v -nt 4
            {binary_name} grompp -f ../../requirements/equilibration.mdp -c minimization_{bead_names_str}.gro -p top_{bead_names_str}_siRNA_challenge.top -o equilibration_siRNA_{bead_names_str}.tpr -maxwarn 2
            {binary_name} mdrun -deffnm equilibration_siRNA_{bead_names_str} -v -nt 4
            """
        # Path for the bash script
        bash_script_path = "charge_neutralisation+solvation_siRNA.sh"

        # Writing the bash script
        with open(bash_script_path, "w") as file:
            file.write(bash_script_content)

        # Make the script executable
        os.chmod(bash_script_path, 0o755)

        ########################## Running charge and equilibration###########
        # Path to the Bash script
        bash_script_path = "charge_neutralisation+solvation_siRNA.sh"

        # Path to the log file
        log_file_path = f'{bead_names_str}_charge_setup_siRNA.log'

        # Open the log file in write mode
        with open(log_file_path, 'w') as log_file:
            # Running the Bash script and redirecting stdout and stderr to the log file
            process = subprocess.Popen(['bash', bash_script_path], stdout=log_file, stderr=subprocess.STDOUT, text=True)

            # Wait for the process to complete
            process.wait()




        #########################################create plumed file#################################



        # Define the content of the topology file
        end_mol=425+(len(beads)+40)
        plumed_content = f"""
            # treat each molecule as whole

        WHOLEMOLECULES ENTITY0=1-425 ENTITY1=426-{end_mol}, 
        c1: COM ATOMS=1-425
        c2: COM ATOMS=426-{end_mol}
        c3: COM ATOMS=100-110
        # define atoms for distance



        d1: PROJECTION_ON_AXIS AXIS_ATOMS=1,218 ATOM=c2
        d2: PROJECTION_ON_AXIS AXIS_ATOMS=120,341 ATOM=c2


        uwall: UPPER_WALLS ARG=d1.ext AT=7 KAPPA=100 EXP=2 EPS=1 OFFSET=0
        
        restraint: RESTRAINT ARG=d2.proj AT=0 KAPPA=100

        # also notice that a LABEL keywsord can be used and is equivalent
        # to adding the name at the beginning of the line with colon, as we did so far

        MOVINGRESTRAINT ...
          LABEL=res
          ARG=d1.ext
          STEP0=0 AT0=5.5 KAPPA0=100
          STEP1=200000 AT1=0 KAPPA1=100
        ...
        PRINT ARG=d1.ext,res.work FILE=COLVAR_siRNA STRIDE=1000
        PRINT ARG=res.bias FILE=BIAS_siRNA
        PRINT ARG=res.force2 FILE=FORCE_siRNA
        PRINT ARG=res.work FILE={logfile} STRIDE=1000
        PRINT ARG=restraint.bias 

        COMMITTOR ...
          ARG=d1.ext
          STRIDE=100
          BASIN_LL1=0
          BASIN_UL1=1.5
        ...
        """

        # Path for the topology file
        plumed_file_path = f"plumed_{bead_names_str}_siRNA.dat"

        # Writing the topology file
        with open(plumed_file_path, "w") as file:
            file.write(plumed_content)    

        # Bash script content
        bash_script_content = f"""#!/bin/bash

        # GROMACS simulation command
        rm *step*
        rm *#*
        {binary_name} grompp -f ../../requirements/dynamic.mdp -c equilibration_siRNA_{bead_names_str}.gro -p top_{bead_names_str}_siRNA_challenge.top -o plumed_{bead_names_str}_siRNA.tpr -maxwarn 2 -r equilibration_siRNA_{bead_names_str}.gro 
        {binary_name} mdrun -deffnm plumed_{bead_names_str}_siRNA  -plumed plumed_{bead_names_str}_siRNA.dat -nt 4 -nsteps 200000
        rm *step*
        """
        # Path for the bash script
        bash_script_path = "plumed_run_siRNA.sh"

        # Writing the bash script
        with open(bash_script_path, "w") as file:
            file.write(bash_script_content)

        # Make the script executable
        os.chmod(bash_script_path, 0o755)

        ##############################Running plumed run setup##############################

        # Path to the Bash script
        bash_script_path = "plumed_run_siRNA.sh"

        # Path to the log file
        log_file_path = f'{bead_names_str}_plumed_run_siRNA.log'

        # Open the log file in write mode
        with open(log_file_path, 'w') as log_file:
            # Running the Bash script and redirecting stdout and stderr to the log file
            process = subprocess.Popen(['bash', bash_script_path], stdout=log_file, stderr=subprocess.STDOUT, text=True)

            # Wait for the process to complete
            process.wait()

        ###########################Analysis and plot#########################################



        # Load and process the first file
        data = np.loadtxt(logfile)
        data_1 = data[:, 1]
        
        # Find the minimum value in the 2nd column
        num_to_drop = 35
        num_to_average = 4
        data1_mean = np.mean(data[num_to_drop : num_to_drop + num_to_average, 1])
        print(f"Method 1: Mean = {data1_mean}")
        # Shift the data so that the minimum value is zero
        data_1_shifted = data_1 - data1_mean
        
        # Calculate the integrated value of the shifted data
        integrated_data_siRNA = np.trapezoid(data_1_shifted, axis=0) / 100
        
        # Save the minimum value to a file
        np.savetxt('WORK_siRNA_min_to_zero', data_1_shifted, fmt='%.6f', newline='\n')
        
        # Save the integrated value to a file
        file_path_results = f'../mean_diff_data_siRNA_pH_{pH}.txt'

        # Writing mean_diff_2 to the file
        with open(file_path_results, 'a') as file_result:
            file_result.write(f'{sys.argv[1]},{bead_names_str},{integrated_data_siRNA :.3f}\n')
        mask = (df['model_path_lipo'] ==f"model_{sys.argv[1]}_lipo") & (df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro")

        # Assigning the performance value to a new column 'performance' in the selected rows
        df.loc[mask, f'Performance_siRNA_pH_{pH}'] = integrated_data_siRNA
        print("siRNA Challenge completed")

        return True




logging.basicConfig(level=logging.INFO)



def clean_up_files(bead_names_str):
    files_to_keep = [
        f"backbone_modified_{bead_names_str}_pH_8.itp",
        f"backbone_modified_{bead_names_str}_pH_4.itp",
        f"modified_{bead_names_str}_pH_4.gro",
        f"modified_{bead_names_str}_pH_8.gro",
        f"minimization-vac_{bead_names_str}.gro"
    ]

    for filename in os.listdir('.'):  # Iterate through all files in the current directory
        if filename not in files_to_keep:  # If the file is not in the keep list
            file_path = os.path.join('.', filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Use shutil for directories
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")



def run_simulation(func, *args,**kwargs):
    """Run a simulation with file size checking and restarts."""
    max_attempts = 4
    delay = 3

    file_name = args[-1] 
    for attempt in range(max_attempts):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args,**kwargs)  # Execute the function without the file name argument
                
                # Wait for 60 seconds
                time.sleep(30)
                
                # Check initial file size
                initial_size = os.path.getsize(file_name) if os.path.exists(file_name) else 0
                
                
                # Wait for another 30 seconds
                time.sleep(60)
                
                # Check new file size
                new_size = os.path.getsize(file_name) if os.path.exists(file_name) else 0
                print(new_size)
                if new_size > initial_size:
                    # File has grown, let the simulation continue
                    print(f"Simulation progressing as expected.")
                    future.result()  # Wait for the function to complete
                    return True
                else:
                    # File hasn't grown or doesn't exist, terminate and try again
                    future.cancel()
                    logging.warning(f"Simulation {func.__name__ } not progressing. Restarting (attempt {attempt + 1})...")
                    clean_up_files(args[0])
                    print("Files cleaned .. Starting again")

        
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with parameters: {args}. Error: {e}\n{traceback.format_exc()}")
        
        # Clean up before retrying
        time.sleep(delay)
        clean_up_files(args[0])  # Assuming the first arg is bead_names_str
        if os.path.exists(file_name):
            os.remove(file_name)
        
        if attempt == max_attempts - 1:  # We've reached the last attempt
            logging.critical(f"All {max_attempts} attempts failed for simulation {func.__name__}. Exiting script.")
            sys.exit(1)  # Exit with an error code
    
    return False

def clean_up_files(bead_names_str):
    files_to_keep = [
        f"backbone_modified_{bead_names_str}_pH_8.itp",
        f"backbone_modified_{bead_names_str}_pH_4.itp",
        f"modified_{bead_names_str}_pH_4.gro",
        f"modified_{bead_names_str}_pH_8.gro",
        f"minimization-vac_{bead_names_str}.gro"
    ]

    for filename in os.listdir('.'):  # Iterate through all files in the current directory
        if filename not in files_to_keep:  # If the file is not in the keep list
            file_path = os.path.join('.', filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Use shutil for directories
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
def timeout_handler(signum, frame):
    logging.error("Script execution timed out after 1500 seconds.")
    # Optional cleanup actions if needed
    kill_processes_by_name("{binary_name}")
    sys.exit(1)  # Exit with an error code

if __name__ == "__main__":
    start_time = time.time()  # Capture start time
    # Set the alarm to trigger after 1800 seconds (30 minutes)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800)

    # Create an instance of your FileGenerator class
    bead_info_list = [(bead_insertion_info_ph_8, 8), (bead_insertion_info_ph_4, 4)]  # List of charges to process

    for bead_insertion_info, ph in bead_info_list:
        file_generator = FileGenerator(bead_insertion_info)
        # Call methods in order for the current charge workflow
        file_generator.insert_beads_before_last_line('../../requirements/PBAE-2_Dimer_new.gro', bead_insertion_info, ph, initial_distance=0.5)
        file_generator.add_new_beads_to_itp('../../requirements/PBAE-2_Dimer_new.itp', bead_insertion_info, f"backbone_modified_atoms_pH_{ph}.itp")
        file_generator.update_itp_with_bond_info("../../requirements/bond_parameters_extended.csv", f'backbone_modified_atoms_pH_{ph}.itp', f'backbone_modified_{bead_names_str}_pH_{ph}.itp', bead_insertion_info)
        file_generator.update_itp_with_hardcoded_angle_info(f'backbone_modified_{bead_names_str}_pH_{ph}.itp', f'backbone_modified_{bead_names_str}_pH_{ph}.itp', bead_insertion_info)
    print("File generation completed")

    # Define unique file names for each challenge
    double_membrane_file = f"WORK_double_Membrane"
    sirna_challenge_file = f"WORK_siRNA"
    disso_challenge_file = f"WORK_siRNA_disso"

    success_1 = run_simulation(double_membrane, bead_names_str, f'modified_{bead_names_str}_pH_4.gro', double_membrane_file)
    
    
    if not success_1:
        logging.error("double_membrane function failed after several attempts.")
        sys.exit(1)  # Exit with error code

    if success_1:
        success_2 = run_simulation(sirna_challenge, bead_names_str,f'modified_{bead_names_str}_pH_4.gro', sirna_challenge_file,pH=4)
        if not success_2:
            logging.error("sirna_challenge function failed after several attempts.")
            sys.exit(1)  # Exit with error code
    if success_2:
        success_3 = run_simulation(sirna_challenge, bead_names_str, f'modified_{bead_names_str}_pH_8.gro',disso_challenge_file,pH=8)
        if not success_3:
            logging.error("disso_challenge function failed after several attempts.")
            sys.exit(1)  # Exit with error code
    
    signal.alarm(0)

    filtered_df = df[(df['model_path_lipo'] == f"model_{sys.argv[1]}_lipo") | (df['model_path_hydro'] == f"model_{sys.argv[1]}_hydro")]
    filtered_df.to_pickle(f"../model_results/model_{sys.argv[1]}_results_run{sys.argv[2]}.pkl")
    
    # Deletion of the folder
    os.chdir("..") 
    shutil.rmtree(f"{bead_names_str}") 
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    sys.exit(0)  # Exit with success code