import subprocess
import os.path
from ga import new_GA
import rl
import pandas as pd
from bead_exchanger import exchange_beads
import sys
from Utils.make_pdf import make_pdf_file

# Specify the output file
output_file = "output.txt"

# Open the file in write mode 'w'
with open(output_file, 'w') as f:
    # Redirect standard output
    sys.stdout = f

# 1. Run the RL script
    env_hydro=rl.MoleculeEnvironment(5)
    ##define targets
    lipo_target="C=CC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CCNCCCCCCCCCCCCC)cc2)cc1"
    hydro_target="C=CC(=O)OCC(O)COc1ccc(C(C)(C)c2ccc(OCC(O)COC(=O)CCNCCN3CCOCC3)cc2)cc1"
    env_hydro.load_target_desc(hydro_target,typ="hydro")
    ##initialize class,load targets and calculate N models each
    moldesigner=rl.MoleculeDesigner()
    env_lipo=rl.MoleculeEnvironment(5)
    env_lipo.load_target_desc(lipo_target,typ="lipo")

  ########Comment if continue

    hydro_models=moldesigner.make_mol_models(1,env_hydro,total_timesteps=10)
    lipo_models=moldesigner.make_mol_models(1,env_lipo,total_timesteps=10)
    df=moldesigner.new_mol_from_single_models(hydro_models,lipo_models,show_mol="lead")
    print("Moldesigner ready")
    df=exchange_beads(8,df,0.5)
    df=exchange_beads(4,df,0.5)
    df.to_pickle("DFfromRL.pkl")

## 2. Check for the presence of 'model1.pkl'
    for i in range(0,50):

        if os.path.exists("DFfromRL.pkl"):
            try:
                os.environ['ITERATION_NUMBER'] = str(i)
                subprocess.run(["python3", "multithread3.py"] ,check=True)  # Will raise exception on error
                print(f"Finished iteration {i+1}")
            except subprocess.CalledProcessError as e:
                print(f"Error in multithread.py: {e}")
        else:
            print("Error: 'DFfromRL.pkl' not found. Please ensure the RL script generated it.")
            sys.exit(1)  # Exit with an error code
             
        # Perform GA on challengeout.pkl
#
#
        ## Load the dataframe and remove duplicates initially
        df = pd.read_pickle("DFfromRL.pkl")
    
        # Check if the 'performance_score' column exists
        if 'performance_score' not in df.columns:
            print("Error: 'performance_score' column not found in the DataFrame.")
            sys.exit(1)  # Exit with an error code (non-zero)
    
        # Check if *any* value in the 'performance_score' column is greater than 9
        if (df['performance_score'] > 17).any():
            print("Success: At least one performance score is greater than 9.")
            df[['performance_score', 'Performance_siRNA_pH_4', 'Performance_double_membrane', 'Performance_siRNA_pH_8',
               'beads_hydro', 'beads_lipo']].to_csv(f"output_df_episode_{i}.csv")
            
            #make PDF
            make_PDF_file()
            sys.exit(0)  # Exit with a success code (0)
        else:
            print("Continuing: No performance score is greater than 9.")
        # The rest of your script would go here
            # Convert the list columns to tuples
            df['beads_hydro_tuple'] = df['beads_hydro'].apply(tuple)
            df['beads_lipo_tuple'] = df['beads_lipo'].apply(tuple)
            print(df.columns)
            # Drop duplicates based on the tuple columns
            df.drop_duplicates(subset=['beads_hydro_tuple', 'beads_lipo_tuple'], inplace=True)
            
            # Drop the helper tuple columns
            df.drop(columns=['beads_hydro_tuple', 'beads_lipo_tuple'], inplace=True)
            # Load target descriptions
            env_lipo.load_target_desc(lipo_target, typ="lipo")
            env_hydro.load_target_desc(hydro_target, typ="hydro")
            # Initialize the GA
            ga = new_GA(df, 2, env_hydro, env_lipo, elite=1)
            # Print performance scores
            print(ga.input_df[["performance_score"]])
            print(ga.parents["performance_score"])
            output_df=ga.input_df
            output_df[['performance_score', 'Performance_siRNA_pH_4', 'Performance_double_membrane', 'Performance_siRNA_pH_8',
               'beads_hydro', 'beads_lipo']].to_csv(f"output_df_episode_{i}.csv")
            # Export lead and mutate
            ga.export_lead(i)
            new_pol = ga.mutate(2, 1)
            # Process the dataframe with moldesigner and exchange_beads
            df = moldesigner.mol_from_concat_models(new_pol, env_hydro, env_lipo, show_mol="lead")
            df = exchange_beads(8, df, 0.5)
            df = exchange_beads(4, df, 0.5)
            # Remove duplicates before saving
            
            df.to_pickle("DFfromRL.pkl")
            print(f"Starting Challenges in iteration {i+1}")
            print(f"Starting iteration {i+1}",flush=True)

