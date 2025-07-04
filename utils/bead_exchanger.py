import copy
import random
from . import PkaPred
from .PkaPred import GCN
import torch
import pandas as pd
import numpy as np

def helper_function(smile):


    '''
    helper function laods trained GCN and calculates pka_list.
    Parameters:
    smiles(string): SMILES to pass.
    
    Returns: pka_values(list): List of pka values for every deprotonable COOH and every protonable N.
    '''
    
    #load model acid
    model_acid = GCN()
    model_acid.load_state_dict(torch.load('./params/CGNN_params_carbox.pth'))
    #predict pka values
    #pka_values=PkaPred.predict_pka(smiles,model)
    pka_values_acid = PkaPred.predict_pka(smile,model_acid,mask_type="acid")

    model_base = GCN()
    model_base.load_state_dict(torch.load('./params/CGNN_params_new_amine.pth'))
    #predict pka values
    #pka_values=PkaPred.predict_pka(smiles,model)
    pka_values_base= PkaPred.predict_pka(smile,model_base,mask_type="base")
    
    return pka_values_base, pka_values_acid

        
def transform_beads_ph(pH_value,df,thresh,typ):
    

    '''
    Function transforms a bead structure into a protonated version.
    
    Parameters:
    pH_value(float): pH value to transform bead
    df(dataframe): Dataframe that contains 4 necesarry columns ("beads_hydro","beads_lipo","smiles_hydro","smiles_lipo")
    thresh (float): Probability threshold were mol gets protonated
    typ (string): which sidechain is treated-"hydro" or "lipo"

    Returns:
    df_copy(pd.DataFrame(object)): Updated dataframe
    '''
    #think about appending protonated structures beads as well, otherwise the bead_idx will fail
    protonable_beads= ["TN6d","SN4","SN3a","anchor_bead"]
    deprotonable_beads= ["SP2"]
    #calculate pka from Mol
    df_copy=copy.deepcopy(df)
    changes_made = False
    if f"beads_{typ}_ph_{pH_value}" in df_copy:
        smile = df_copy[f"smiles_{typ}_ph_{pH_value}"]
        beads = df_copy[f"beads_{typ}_ph_{pH_value}"]
    else:
        smile = df_copy[f"smiles_{typ}"]
        beads = df_copy[f"beads_{typ}"]
    #CONVERT SO THAT HELPER FUNCTION OUTPUTS ACID AND BASE PKA_LIST   

    pka_base,pka_acid=helper_function(smile)
    #print("PKA_BASE: ",pka_base)
    #print("PKA_ACID: ",pka_acid)
    bead_list=list(beads)
    N_idx=[]

    O_idx=[]
    
    #ADD O IDX
    for idx,atom in enumerate(smile):
        if atom == "N":
            if idx < len(smile) - 1:
                if smile[idx+1]=="H":
                    continue
                else:
                    N_idx.append(idx)
            else:
                N_idx.append(idx)
        elif atom =="O":
            if idx==len(smile)-1 and smile[idx-2]=="O" and smile[idx-1]==")":

                O_idx.append(idx)
                
    #print("N_IDX: ", N_idx)      
    #print("O_IDX: ", O_idx)
    #DIVIDE INTO N AND O PROBAS
    probabilitys_N=[]
    probabilitys_O=[]
    #use cap value to avoid overflow
    max_exp = 20
    #REPEAT PROCESS FOR PROT AND DEPROT
    for pka_b in pka_base:       
        #check protonation proba
        exp_value_b = min(pka_b - pH_value, max_exp)
        N_acid = round(np.power(10.0, exp_value_b), 5)
        proba_N = N_acid / (1 + N_acid)
        probabilitys_N.append(proba_N)
        #check deprotonation proba
    for pka_a in pka_acid:
        exp_value_a = min(pH_value-pka_a, max_exp)
        O_acid = round(np.power(10.0, exp_value_a), 5)
        proba_O = O_acid / (1 + O_acid)
        probabilitys_O.append(proba_O)
    #print("proba_N: ",probabilitys_N)
    #print("proba_O: ",probabilitys_O)
    smile_list = list(smile)    
    
    if probabilitys_N and max(probabilitys_N) >= 0.5:
        changes_made=True
        
        protonable_beads_idx=[]
        for idx,b in enumerate(beads):
            if b in protonable_beads:
                protonable_beads_idx.append(idx)
          
        #check for max_proba_nitrogen
        
        nitrogen_idx = probabilitys_N.index(max(probabilitys_N))  

        #save bead that is most proba protonated
        bead_idx = protonable_beads_idx[nitrogen_idx]

        #save idx of nitrogen that is protonated most proba
        atom_index = N_idx[nitrogen_idx]
 
        # NAME OF BEADS STILL NEEDS TO BE ADJUSTED
        if beads[bead_idx] == "TN6d":
            bead_list[bead_idx] = "TQ4p"
        if beads[bead_idx] == "TN6d":
            bead_list[bead_idx] = "TQ4p"
            smile_list[atom_index] = "[NH3+]"
            if bead_idx + 1 < len(beads) and beads[bead_idx + 1] is not None:
            
                smile_list[atom_index] = "[NH2+]"
        elif beads[bead_idx] == "SN4":
            bead_list[bead_idx] = "SQ3p"
            bead_list[bead_idx] = "SQ3p"
            smile_list[atom_index] = "[NH2+]"
            if bead_idx + 1 < len(beads) and beads[bead_idx + 1] is not None:
            
                smile_list[atom_index] = "[NH+]"
        elif beads[bead_idx] == "SN3a":
            bead_list[bead_idx] = "SQ2p"
            bead_list[bead_idx] = "SQ2p"
            smile_list[atom_index] = "[NH+]"
        elif beads[bead_idx]== "anchor_bead":
            bead_list[bead_idx] = "SQ2p"
            smile_list[atom_index] = "[NH2+]"
        elif beads[bead_idx]== "anchor_bead":
            bead_list[bead_idx] = "SQ2p"
            smile_list[atom_index] = "[NH2+]"
            if bead_idx + 1 < len(beads) and beads[bead_idx + 1] is not None:
                smile_list[atom_index] = "[NH+]"
        
        
        
        #ADD IF beads[deprot_bead_idx]:...
    elif probabilitys_O and max(probabilitys_O) >= 0.5:
        changes_made=True
        
        deprotonable_beads_idx=[]
        for idx,b in enumerate(beads):
            if b in deprotonable_beads:
                deprotonable_beads_idx.append(idx)
          
        #check for max_proba_nitrogen
        oxygen_idx = probabilitys_O.index(max(probabilitys_O))  
        #save bead that is most proba protonated
        bead_idx = deprotonable_beads_idx[oxygen_idx]
        #save idx of nitrogen that is protonated most proba
        atom_index = O_idx[oxygen_idx]
 
        
        if beads[bead_idx] == "SP2":
            bead_list[bead_idx] = "SQ5n"
            smile_list[atom_index] = "[O-]"


    # Convert the list back to a string
    smile = ''.join(smile_list)
        

    if changes_made:
        df_copy[f"beads_{typ}_ph_{pH_value}"] = bead_list
        df_copy[f"smiles_{typ}_ph_{pH_value}"] = smile
        return transform_beads_ph(pH_value, df_copy, thresh, typ)
    
    return df_copy


def exchange_beads(pH_value, df, thresh):
    new_df = pd.DataFrame()
    for i in range(len(df)):
        exchange_hydro = transform_beads_ph(pH_value, df.iloc[i], thresh, "hydro")
        exchange_lipo = transform_beads_ph(pH_value, df.iloc[i], thresh, "lipo")

        # Ensure both are DataFrames
        if isinstance(exchange_hydro, pd.Series):
            exchange_hydro = exchange_hydro.to_frame().T
        if isinstance(exchange_lipo, pd.Series):
            exchange_lipo = exchange_lipo.to_frame().T
        
        # Find columns in exchange_lipo not in exchange_hydro
        unique_lipo_columns = [col for col in exchange_lipo.columns if col not in exchange_hydro.columns]

        # Check if the unique columns exist in exchange_hydro before trying to access them
        #existing_unique_columns = [col for col in unique_columns if col in exchange_hydro.columns]
        df_concat = pd.concat([exchange_hydro, exchange_lipo[unique_lipo_columns]], axis=1)
        new_df = pd.concat([new_df, df_concat], ignore_index=True)

    return new_df
