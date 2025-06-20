import pandas as pd
import pickle # Import pickle for more specific error handling

# Define the name of your pickle file
pickle_filename = "top_performer_26.pkl"

try:
    # Attempt to load the object from the pickle file using pandas
    loaded_data = pd.read_pickle(pickle_filename)

    # Check if the loaded object is a pandas DataFrame
    if isinstance(loaded_data, pd.DataFrame):
        print(f"Successfully loaded a DataFrame from '{pickle_filename}'.")
        print("\nColumns in the DataFrame:")
        # Print the columns (converting to a list for cleaner output)
        print(list(loaded_data.columns))
        print(loaded_data["performance_score"])
        # Alternatively, print one column per line:
        # for col in loaded_data.columns:
        #    print(f"- {col}")
    else:
        # If it's not a DataFrame, report the type found
        print(f"Loaded object from '{pickle_filename}' is not a pandas DataFrame.")
        print(f"Object type found: {type(loaded_data)}")
        # You could try checking if it has a 'columns' attribute anyway
        if hasattr(loaded_data, 'columns'):
            print("However, it has a '.columns' attribute:")
            try:
                print(loaded_data.columns)
            except Exception as e:
                print(f"(Could not display columns: {e})")
        else:
             print("It does not have a '.columns' attribute.")


except FileNotFoundError:
    print(f"Error: The file '{pickle_filename}' was not found.")
except (pickle.UnpicklingError, EOFError):
    print(f"Error: Could not read or unpickle '{pickle_filename}'. The file might be corrupted or empty.")
except Exception as e:
    # Catch other potential errors during loading (e.g., module not found if it contains custom classes)
    print(f"An unexpected error occurred: {e}") 

selected_columns_df = loaded_data[[
    'performance_score',
    'Performance_siRNA_pH_4',
    'Performance_double_membrane',
    'Performance_siRNA_pH_7',
    'beads_hydro',
    'beads_lipo'
]]
output_filename = f"output_df_episode_26.csv"
selected_columns_df.to_csv(output_filename)
print(f"DataFrame columns saved to {output_filename}")