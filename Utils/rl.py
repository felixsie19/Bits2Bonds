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

dict_iter = {
"SC1": 'CCC',
"TC1": 'CC',
"TP1": "CO",
"TN6d": "CN",
"SN4":"CNC",
"SN3a":"CN(C)C",
"TC4":"C=C",
"TC6":"CS",
"N5a":"CC(=O)C",
"SP2": "C(=O)O",
None:""}

rkrb.DisableLog("rdApp.error")
# Molecule Descriptor Calculator
selected_desc_list_hydro = ["MolWt",
"FpDensityMorgan1",
"InertialShapeFactor",
"BCUT2D_MWHI",
"NPR2",
"VSA_EState4",
"BCUT2D_CHGLO",
"BCUT2D_MWLOW",
"BalabanJ",
"BCUT2D_LOGPHI",
"Eccentricity",
"VSA_EState2",
"PEOE_VSA10",
"PEOE_VSA8",
"VSA_EState3",
"BCUT2D_MRLOW",
"PMI2",
"BCUT2D_MRHI",
"FpDensityMorgan2"
]

descriptor_names = [
    'RadiusOfGyration',
    'InertialShapeFactor',
    'Eccentricity',
    'Asphericity',
    'SpherocityIndex',
    'PMI1',
    'PMI2',
    'PMI3',
    'NPR1',
    'NPR2',
]
selected_desc_list_lipo = [
"MolWt",
"BCUT2D_LOGPHI",
"BCUT2D_LOGPLOW",
"FpDensityMorgan1",
"InertialShapeFactor"
"BCUT2D_MWHI",
"VSA_EState5",
"FpDensityMorgan2",
"BCUT2D_MWLOW",
"VSA_EState3",
"FpDensityMorgan3",
"NPR2",
"VSA_EState2",
"Eccentricity",
"PEOE_VSA8",
"BalabanJ",
"BCUT2D_CHGHI",
"PEOE_VSA9",
"HeavyAtomMolWt",
"RadiusOfGyration"
]
class MoleculeDesigner:
    def __init__(self):

        pass



    def make_mol_models(self,nr_models,env,total_timesteps=100000,show_mol=True):
        #initialize model
        """
        Function to train DQN models.

        Parameters:
        nr_models(int) : number of models that is supposed to be trained
        env(object) : environment that is used for training of models
        total_timesteps(int) : timesteps the model is trained with
        show_mol(bool) : if True the monomer will be outputed
        
        Returns:
        dict: key is model and values are beads and smiles 



        """
        mol_list=[]
        bead_list=[]
        model_dict={}
        for i in range(nr_models):
            # Train the DQN model
            model = DQN("MlpPolicy", env, verbose=1,buffer_size=500000)
            model.learn(total_timesteps=total_timesteps)


            #render molecule based on trained model DQN
            
            obs, info  = env.reset()
            while True:
                #for usage of DQN model
                action, _ = model.predict(obs, deterministic=False)

                obs, rew, done, truncated, _ = env.step(action)
                if done == True:
                    break

            final_mol = env.state
            final_beads = env.bead_list
            final_mol_end = "CCC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CC"+final_mol+")cc2)cc1"
            #target_smiles = "C=CC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CCNCCN3CCOCC3)cc2)cc1"
            mol_list.append(final_mol)
            bead_list.append(final_beads)
            #CHNAGED THE FINAL MOL_END TO FINAL MOL FOR THE LIST TO RUN THE PKA_PRED PROPERLY!
            model_dict[model]=[final_beads,final_mol]
            #render molecule
            if show_mol==True:
                img=Chem.MolFromSmiles(final_mol_end)
                img=Draw.MolToImage(img)
                #img.show()  

            #output models and beads 
        return model_dict
    def new_mol_from_single_models(self,model_dict1,model_dict2,show_mol):
        """
        Function concatenates both sidechain models and renders molecule.

        Parameters:
        model_dict1 (dict): dict containing information about smiles, beads and model parameters for model1
        model_dict2 (dict): dict containing information about smiles, beads and model parameters for model2
        show_mol (string): "lead" to render only lead candidate, "always" to show every MOL, "never" to show nothing
        Returns:
        list: list containing concatenated information like this : [[model_hydro,model_lipo],[beads_hydro,beads_lipo],[mol_hydro],[mol_lipo]]


        """
        #make list so that [[model_hydro,model_lipo],[beads_hydro,beads_lipo],[mol_hydro],[mol_lipo]]
        final_list=[]
        last=len(model_dict1)-1
        lead=True
        if len(model_dict1)==len(model_dict2):
            for (key1, value1), (key2, value2) in zip(model_dict1.items(), model_dict2.items()):
                
                final_list.append([[key1,key2],[value1[0],value2[0]],[value1[1],value2[1]]])
                hydro_full_monomer="CCC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CC"+value1[1]+")cc2)cc1"
                lipo_full_monomer ="CCC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CC"+value2[1]+")cc2)cc1"
                #render mol
                if show_mol=="lead" and lead == True:
                    mol1=Chem.MolFromSmiles(hydro_full_monomer)
                    mol2=Chem.MolFromSmiles(lipo_full_monomer)
                    mol = Chem.CombineMols(mol1,mol2)
                    mol = Chem.RWMol(mol)
                    atoms = mol.GetAtoms()
                    mol.AddBond(0, 31+mol1.GetNumAtoms(), order=Chem.BondType.SINGLE)
                    img=Draw.MolToImage(mol,size=(800,600))
                    img.save("../Results/lead.png") 
                    #img.show()
                    lead=False
                if show_mol=="always":
                    mol1=Chem.MolFromSmiles(hydro_full_monomer)
                    mol2=Chem.MolFromSmiles(lipo_full_monomer)
                    mol = Chem.CombineMols(mol1,mol2)
                    mol = Chem.RWMol(mol)
                    atoms = mol.GetAtoms()
                    mol.AddBond(0, 31+mol1.GetNumAtoms(), order=Chem.BondType.SINGLE)
                    img=Draw.MolToImage(mol,size=(800,600))
                    #img.show()


                
        else:
            print("Error in model_dict_length")
        data = []
        for x in range(0, len(final_list)):
            model_data = {}  # Initialize the dictionary outside the model loop, for each `x`

            for i, model in enumerate(final_list[x][0]):
                # Determine model type
                model_type = "hydro" if i == 0 else "lipo"

                # Collect data for each model and append it to the model_data dictionary with type-specific keys
                model_data[f"model_policy_dict_{model_type}"] = model.policy.state_dict()
                model_data[f"beads_{model_type}"] = final_list[x][1][i]
                model_data[f"smiles_{model_type}"] = final_list[x][2][i]
                model_data[f"model_path_{model_type}"] = f"model_{x}_{model_type}"

            # After collecting both models' data, append the consolidated dictionary to the data list
            data.append(model_data)

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
        return df
    def mol_from_concat_models(self,concat_list, env_hydro,env_lipo,show_mol):
        """
        Function creates new df of candidates.

        Parameters:
        concat_list (list): List of concatenated_DQN_objects
        env_hydro (object): environment used for the DQN object that is supossed to generate a hydrophilic polymer
        env_lipo (object) : environment used for the DQN object that is supossed to generate a lipophilic polymer
        show_mol (string) : see new_mol_from_single_models() docstring
        Returns:
        dataframe: new pandas dataframe is returned


        """
        #concat list should look like [['m1', 'M1'], ['m2', 'M2'], ['m3', 'M3']]
        model_dict1={}
        model_dict2={}
        
        for i in range(len(concat_list)):
            model1=concat_list[i][0]
            model2=concat_list[i][1]
            #run_hydrophilic_model
            obs, info  = env_hydro.reset()
            while True:
                #for usage of DQN model
                action, _ = model1.predict(obs, deterministic=False)

                obs, rew, done, truncated, _ = env_hydro.step(action)
                if done == True:
                    break
            final_mol = env_hydro.state
            final_beads = env_hydro.bead_list
            
            final_mol_end = "CCC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CC"+final_mol+")cc2)cc1"
            #target_smiles = "C=CC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CCNCCN3CCOCC3)cc2)cc1"
            #mol_list.append(final_mol)
            #bead_list.append(final_beads)
            #CHNAGED THE FINAL MOL_END TO FINAL MOL FOR THE LIST TO RUN THE PKA_PRED PROPERLY!
            model_dict1[model1]=[final_beads,final_mol]
            #run lipophilic_model
            obs, info  = env_lipo.reset()
            while True:
                #for usage of DQN model
                action, _ = model2.predict(obs, deterministic=False)

                obs, rew, done, truncated, _ = env_lipo.step(action)
                if done == True:
                    break
            final_mol = env_lipo.state
            final_beads = env_lipo.bead_list
            final_mol_end = "CCC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CC"+final_mol+")cc2)cc1"
            #target_smiles = "C=CC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CCNCCN3CCOCC3)cc2)cc1"
            #mol_list.append(final_mol)
            #bead_list.append(final_beads)
            #CHNAGED THE FINAL MOL_END TO FINAL MOL FOR THE LIST TO RUN THE PKA_PRED PROPERLY!
            model_dict2[model2]=[final_beads,final_mol]
            #render molecule
        final_list=self.new_mol_from_single_models(model_dict1,model_dict2,show_mol=show_mol)

        return final_list
        
class MoleculeEnvironment(gym.Env):
    def __init__(self,max_steps):
        super().__init__()
        
        # initialize state
        self.state = "N"                                 
        self.target_descriptors=None
        descriptor_list=None
        self.calc=None
        #action and obs space
        self.action_space = spaces.Discrete(len(dict_iter))
        self.observation_space = spaces.Box(low=0, high=1,shape=(2048,), dtype=np.int32)
                 
        #initialize steps
        self.max_steps = max_steps
        self.current_step = 0    
        self.last_reward=0
        self.bead_list=["anchor_bead"]
    def load_target_desc(self,target_smiles,typ):
        """
        Function loads target smiles and converts them into descriptors.

        Parameters:
        target_smiles(string): Smiles of target structure monomer
        typ (string): determines which descriptor list is loaded

        Returns:
        np.array: array contains all selected descriptors


        """
        if typ=="hydro":
            self.descriptor_list=selected_desc_list_hydro
        elif typ=="lipo":
            self.descriptor_list=selected_desc_list_lipo
            
        desc_names = [desc_name[0] for desc_name in Descriptors.descList if desc_name[0] in self.descriptor_list]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
        target_mol = Chem.MolFromSmiles(target_smiles)
        target_mol = Chem.AddHs(target_mol)
        Chem.SanitizeMol(target_mol)
        AllChem.EmbedMolecule(target_mol, randomSeed=42,useRandomCoords=True) 
        AllChem.MMFFOptimizeMolecule(target_mol)
        
        desc_values_target = self.calc.CalcDescriptors(target_mol)
        descriptors = []
        for descriptor_name in descriptor_names:
            if descriptor_name in self.descriptor_list:
                descriptor_function = getattr(Descriptors3D, descriptor_name)
                descriptor_value = descriptor_function(target_mol)
                descriptors.append(descriptor_value)
        combined_values = np.concatenate((desc_values_target, descriptors))
        self.target_descriptors=combined_values
        return combined_values
    
    def calculate_morgan_fingerprints(self, molecule, radius=2, n_bits=2048):
        molecule = Chem.MolFromSmiles(molecule)
        Chem.SanitizeMol(molecule)
        #compute fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)


        return np.array(fp)
    #calculate reward
    def calculate_reward(self,mol_reward, descv_target):
        #target
        mol_reward1 = "C=CC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CC"+mol_reward+")cc2)cc1"
        mol_reward1 = Chem.MolFromSmiles(mol_reward1)   
        if mol_reward1 is None:
            return None
        else:
            state_mol = Chem.AddHs(mol_reward1)
            Chem.SanitizeMol(state_mol)
            AllChem.EmbedMolecule(state_mol, randomSeed=42,useRandomCoords=True) 
            AllChem.MMFFOptimizeMolecule(state_mol)
            desc_values_state = self.calc.CalcDescriptors(state_mol)
            descriptors_state = []
            for descriptor_name_state in descriptor_names:
                if descriptor_name_state in self.descriptor_list:
                    descriptor_function = getattr(Descriptors3D, descriptor_name_state)
                    descriptor_value_state = descriptor_function(state_mol)
                    descriptors_state.append(descriptor_value_state)

            desc_values_state_combined = np.concatenate((desc_values_state, descriptors_state))
            #print("state",desc_values_state)
            #print("target",desc_target)
            cosine_dist = distance.cosine(desc_values_state_combined, descv_target)
            reward = 1-cosine_dist
            return reward
   #check if mol is valid
    def is_valid_molecule(self, mol):
        mol = Chem.MolFromSmiles(mol)
        return mol is not None and mol.GetNumAtoms() > 0
    
    def reset(self,seed=None):
        #reset and start again with continous representation
        self.current_step = 0
        self.state = "N"
        self.action_space = spaces.Discrete(len(dict_iter))
        initial_fp = self.calculate_morgan_fingerprints(self.state)
        self.last_reward=0
        self.bead_list=["anchor_bead"]
        return initial_fp,{}
    #step_function, that defines actions, rewards and next_state
    def step(self, action):
        reward = 0
        done = False
        truncated = False

        
        bead = list(dict_iter.keys())[action]
        group = list(dict_iter.values())[action]
        self.state = self.state+group
        self.bead_list.append(bead)
        combined_mol = Chem.MolFromSmiles(self.state)
        if list(dict_iter.keys())[action]=="TP1" or list(dict_iter.keys())[action]=="TN6d" or list(dict_iter.keys())[action]=="SP2"  or list(dict_iter.keys())[action]=="TC6":
            done = True  
        # calculate reward
        reward = (self.calculate_reward(self.state, self.target_descriptors)-self.last_reward)
        self.last_reward=reward
        if not self.is_valid_molecule(self.state):
            reward = -5
            done = True
            return mol_fp, reward, done,truncated, {}

                
        mol_fp = self.calculate_morgan_fingerprints(self.state)
        self.current_step += 1
        
        # define terminated 
        if self.current_step >= self.max_steps:

            
            done = True      
     
        return mol_fp, reward, done,  truncated, {}
    
    def seed(self, seed=None):
        pass
    def render(self, mode='human'):
        # render
        pass

    def close(self):
        # close
        pass