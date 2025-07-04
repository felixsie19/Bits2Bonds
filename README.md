Polymer Design with Reinforcement Learning and Genetic Algorithms
This project utilizes a combination of reinforcement learning (RL), genetic algorithms (GA), and molecular dynamics (MD) simulations to design and optimize novel polymer structures for specific tasks, likely related to drug delivery (e.g., siRNA and membrane interactions).

The workflow appears to involve generating polymer sidechains using RL, evaluating their performance through GROMACS and PLUMED simulations, and then using a genetic algorithm to evolve the RL policies to create better-performing polymers over generations.

Core Features
Reinforcement Learning (RL) for Monomer Design: The rl.py script defines a Gym environment for growing polymer sidechains. A Deep Q-Network (DQN) model from stable-baselines3 is trained to select monomers to build a polymer that matches target molecular descriptors.

Genetic Algorithm (GA) for Policy Optimization: The ga.py script implements a genetic algorithm to evolve the RL policies. It takes the best-performing policies (parents), applies mutation, and creates a new generation of policies to be evaluated.

Molecular Dynamics (MD) Simulations: The chains.py script is a complex workflow that takes the generated polymer bead structures, creates GROMACS topologies (.itp, .gro), and runs MD simulations for different "challenges," such as:

Double membrane penetration.

siRNA binding at different pH values.

pKa Prediction: PkaPred.py uses a Graph Convolutional Network (GCN) built with PyTorch Geometric to predict the pKa values of the generated molecules, which is crucial for simulating behavior at different pH levels.

Automated Workflow: The multithread.py script appears to orchestrate the entire process, running the MD simulations in parallel, aggregating the results, calculating a final performance score, and feeding the results back into the GA/RL loop.

Reporting: The make_pdf.py script generates a PDF report visualizing the performance of the lead candidate over generations.

Installation
To set up the environment for this project, install the required Python packages using the requirements.txt file:

pip install -r requirements.txt

External Dependencies:

This project also has significant external dependencies that must be installed and configured correctly:

GROMACS: A molecular dynamics package. It seems a specific version compiled with PLUMED is required (gmx_plu).

PLUMED: A plugin for free-energy calculations in molecular simulations.

How It Works
Initialization: The process likely starts with an initial set of RL models (policies).

Molecule Generation: For each generation, the RL models (rl.py) generate lipophilic and hydrophilic sidechains.

Simulation & Evaluation:

The bead_exchanger.py script adjusts the protonation state of the polymer beads based on the pH.

The chains.py script takes these bead structures and runs them through a series of GROMACS simulations to evaluate their performance in the predefined challenges (e.g., membrane permeability, siRNA binding).

The results (e.g., binding energy, work) are saved.

Scoring & Selection:

multithread.py gathers the results from the simulations and calculates a performance_score for each candidate.

The results are saved to a DataFrame (DFfromRL.pkl).

Evolution:

The ga.py script selects the top-performing models ("elites" or "parents") based on the performance_score.

It then creates a new generation of models by mutating the weights of the parent policies.

Iteration: The process repeats, with the new generation of models being used to generate and evaluate the next set of molecules.

Reporting: Throughout the process, the lead candidate from each generation is tracked, and a final PDF report can be generated with make_pdf.py.

Scripts Overview
rl.py: Defines the RL environment and model training for generating polymer sidechains.

ga.py: Implements the genetic algorithm for evolving the RL policies.

chains.py: The core simulation script. Prepares and runs GROMACS simulations.

multithread.py: The main orchestrator that runs the simulation pipeline in parallel and processes the results.

PkaPred.py: Predicts pKa values using a GCN model.

bead_exchanger.py: Modifies polymer bead structures based on pH and predicted pKa.

make_pdf.py: Creates a PDF report of the results.

process_killer.sh: A utility script to terminate any lingering GROMACS processes.# Bits2Bonds
