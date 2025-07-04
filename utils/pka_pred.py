#!/usr/bin/env python
# coding: utf-8

# # Import data and define classes and functions
# 

# In[1]:


import pickle
import random
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
from torch_geometric.loader import DataLoader
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
import math
from rdkit import Chem
from torch_geometric.data import Batch

#load data-every molecule is a graph
def load_molecule(smiles):
    
    mols=[]
    for smile in smiles:
        mol=Chem.MolFromSmiles(smile)
        Chem.SanitizeMol(mol)
        #mol=Chem.AddHs(mol)
        mols.append(mol)
    return mols
def make_label_list(molecules,pka_list):
# Let's assume molecule_list contains your SMILES strings for the molecules


    # Initialize an empty list to hold the nitrogen presence data
    nitrogen_presence = []

    # Iterate over each molecule in the molecule list
    for smiles,pkas in zip(molecules,pka_list):
        # Convert the SMILES string to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        # Initialize an empty list to store the nitrogen presence for this molecule
        nitrogen_pka = []

        # Iterate over each atom in the molecule
        N_idx_counter=0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() =="N":
                if math.isnan(pkas[N_idx_counter]):
                    value=40
                elif atom.GetFormalCharge() >=1:
                    value=40
                else: 
                    value= pkas[N_idx_counter]
                nitrogen_pka.append(value)

                N_idx_counter += 1
            else: 
                nitrogen_pka.append(40)
                

        # Add the list for this molecule to the main list
        nitrogen_presence.append(nitrogen_pka)
    return nitrogen_presence   
def get_nitrogen_mask(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mask = [1 if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0 else 0 for atom in mol.GetAtoms()]
    return mask
def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def get_atom_features(mol):
    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)

    acceptor_smarts_one = '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
    acceptor_smarts_two = "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    donor_smarts_one = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
    donor_smarts_two = "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]"

    hydrogen_donor_one = Chem.MolFromSmarts(donor_smarts_one)
    hydrogen_donor_two = Chem.MolFromSmarts(donor_smarts_two)
    hydrogen_acceptor_one = Chem.MolFromSmarts(acceptor_smarts_one)
    hydrogen_acceptor_two = Chem.MolFromSmarts(acceptor_smarts_two)

    hydrogen_donor_match_one = mol.GetSubstructMatches(hydrogen_donor_one)
    hydrogen_donor_match_two = mol.GetSubstructMatches(hydrogen_donor_two)
    hydrogen_donor_match = []
    hydrogen_donor_match.extend(hydrogen_donor_match_one)
    hydrogen_donor_match.extend(hydrogen_donor_match_two)
    hydrogen_donor_match = list(set(hydrogen_donor_match))

    hydrogen_acceptor_match_one = mol.GetSubstructMatches(hydrogen_acceptor_one)
    hydrogen_acceptor_match_two = mol.GetSubstructMatches(hydrogen_acceptor_two)
    hydrogen_acceptor_match = []
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_one)
    hydrogen_acceptor_match.extend(hydrogen_acceptor_match_two)
    hydrogen_acceptor_match = list(set(hydrogen_acceptor_match))

    ring = mol.GetRingInfo()

    m = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        o = []
        # Ensure all possible atom types are listed
        o += one_hot(atom.GetSymbol(), ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I', 'B', 'Si'])
        o += [atom.GetDegree()]
        # Ensure all possible hybridization states are covered
        o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'Other'])
        o += [atom.GetImplicitValence()]
        o += [int(atom.GetIsAromatic())]
        o += [int(ring.IsAtomInRingOfSize(atom_idx, size)) for size in range(3, 9)]
        o += [int(atom_idx in hydrogen_donor_match)]
        o += [int(atom_idx in hydrogen_acceptor_match)]
        o += [atom.GetFormalCharge()]
        # Ensure the list 'o' has a consistent length for each atom
        m.append(o)


    return m

def mol2vec(mol):
    node_f = get_atom_features(mol)
    node_f = torch.tensor(node_f, dtype=torch.float)
    edge_index = get_bond_pair(mol)
    edge_index=torch.tensor(edge_index, dtype=torch.long)
    mask = torch.tensor([1 if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0 else 0 for atom in mol.GetAtoms()], dtype=torch.float).unsqueeze(1)
  
    return node_f,edge_index,mask

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(30, 1024, cached=False) 
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GCNConv(1024, 512, cached=False)
        self.bn2 = BatchNorm1d(512)
        self.conv3 = GCNConv(512, 256, cached=False)
        self.bn3 = BatchNorm1d(256)
        self.conv4 = GCNConv(256, 512, cached=False)
        self.bn4 = BatchNorm1d(512)
        self.conv5 = GCNConv(512, 1024, cached=False)
        self.bn5 = BatchNorm1d(1024)
        self.fc2 = Linear(1024, 128)
        self.fc3 = Linear(128, 16)
        self.fc4 = Linear(16, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



def predict_pka(smiles, model):
    print("Smile passed to pka_pred: ", smiles)
    model.eval()  
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    

    x, edge_index, mask = mol2vec(mol)

    data = Data(x=x, edge_index=edge_index, mask=mask)
    

    batch_data = Batch.from_data_list([data])
    

    with torch.no_grad():
        prediction = model(batch_data.x, batch_data.edge_index)
    

    masked_prediction = prediction * mask
    pka_values = masked_prediction[masked_prediction != 0].flatten().tolist()
    
    return pka_values








