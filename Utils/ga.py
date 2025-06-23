#normal reward function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.stats import pearsonr
from scipy.spatial import distance
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.stats import pearsonr
from scipy.spatial import distance
from PIL import Image
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Discrete
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.DataStructs import TanimotoSimilarity
import warnings
import rdkit.rdBase as rkrb
import random
import itertools
from rdkit.Chem import Descriptors3D, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import spearmanr
from stable_baselines3 import DQN
from rdkit.Chem import Draw
import torch
import torch_geometric
import bead_exchanger
import pandas as pd
import pickle
from datetime import datetime
import pandas as pd
import pickle


class new_GA:
    def __init__(self,input_df,nr_parents,env_hydro,env_lipo,elite):
        self.input_df = input_df
        print(self.input_df[['performance_score','Performance_double_membrane','Performance_siRNA_pH_4','Performance_siRNA_pH_4','beads_hydro', 'beads_lipo']])
        self.parents=input_df.sort_values(by='performance_score', ascending=False).head(nr_parents)
        self.pol_list = None
        self.env_hydro=env_hydro
        self.env_lipo=env_lipo
        self.elite=elite-1
    
    def export_lead(self,generation):
        
        # Selecting the first row and specific columns
        self.input_df = self.input_df.sort_values(by='performance_score', ascending=False)
        lead = self.input_df.iloc[0][['performance_score', 'beads_hydro', 'beads_lipo']].to_frame().transpose()
#
        # Adding a new column
        lead['generation'] = generation # Assuming you have a variable `generation`
#
        # Exporting to CSV
        lead.to_csv("lead_output.csv", mode='a', header=False, index=False, sep=',')


    def mutate(self,fill_nr, mutate_strength):
        """
        Function should mutate policy neuronal networks based on a mutation_strength parameter.

        Parameters:
        fill_nr (int): Number that describes how many policys should be in the dataframe in the end.
        mutate_strength (float): Number that is multiplied with the weight/bias.

        Returns:
        type: Description of the return value.

        """
        total_nr = fill_nr-len(self.parents)
        all_DQN_lipo=[]
        all_DQN_hydro=[]
        #loop over policys for hydrophilic monomers 
        for i in range (len(self.parents)):
            #extract weights
            params=self.parents["model_policy_dict_hydro"].iloc[i]
            mutated_single = {}
            #loop over single weights for ActionDQN only
            for key, value in params.items():
                if "target" in key:
                    mutated_single[key]=value
                else:
                    #mutate single weights with random mutation
                    mutated_single[key]=value + value  * (mutate_strength*torch.randn_like(value))
            #append mutated policy
            all_DQN_hydro.append(mutated_single)
            all_DQN_hydro.insert(i,self.parents["model_policy_dict_hydro"].iloc[i])
            #append parents
            policys_hydro=self.new_policys(all_DQN_hydro,self.env_hydro)
        
        #loop over policys for lipophilic monomers 
        for i in range (len(self.parents)):
            params=self.parents["model_policy_dict_lipo"].iloc[i]
            mutated_single = {}
            for key, value in params.items():
                if "target" in key:
                    mutated_single[key]=value
                else:
                    mutated_single[key]=value + value  * (mutate_strength*torch.randn_like(value))
            all_DQN_lipo.append(mutated_single)
            all_DQN_lipo.insert(i,self.parents["model_policy_dict_lipo"].iloc[i])

            
        policys_lipo=self.new_policys(all_DQN_lipo,self.env_lipo)
        
        #zip models with or without shuffling
        all_DQN=self.zip_models(policys_hydro, policys_lipo,shuffle=True)
        self.pol_list=all_DQN
        return all_DQN
    def new_policys(self,pol_list,env):
        """
        Function creates new DQN objects based on the mutated policys.

        Parameters:
        pol_list (list): List of policys
        env (object): environment used for the DQN object

        Returns:
        list: list of new DQN objects



        """
        new_models=[]
        for policy in pol_list:
            new_model= DQN("MlpPolicy", env, verbose=0,buffer_size=500000)
            new_model.policy.load_state_dict(policy)
            new_models.append(new_model)

        return new_models
    def zip_models(self,policys_hydro,policys_lipo,shuffle):
        """
        Function zips thy hydrophilic and hydrophobic models together.

        Parameters:
        policys_hydro (dict): List of policys of hydrophilic monomers
        policys_lipo (dict): List of policys of lipophilic monomers
        shuffle (bool): Information if the policys will be shuffled, elite will not be shuffled
        
        Returns:
        list: list of new concatenated DQN objects


        """
        if shuffle == True:
            
            #hydro_elite=policys_hydro[:self.elite]
            random.shuffle(policys_hydro)
            #policys_hydro=hydro_elite+policys_hydro
            
            #lipo_elite=policys_lipo[:self.elite]
            random.shuffle(policys_lipo)
            #policys_lipo=lipo_elite+policys_lipo
            
        zipped=[]
        for i,j in zip(policys_hydro,policys_lipo):
            temp=[i,j]
            zipped.append(temp)
   
        return zipped    


