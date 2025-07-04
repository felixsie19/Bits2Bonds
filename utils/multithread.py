import pandas as pd
import threading
import subprocess
import time
import os
import signal
import psutil
import numpy as np
from scipy.stats import trim_mean, iqr
from .chains import binary_name

iteration_number = int(os.environ.get('ITERATION_NUMBER'))
def process_iteration(iteration_num, run):
    try:
        args = ["python3", "./utils/chains.py", str(iteration_num), str(run)]
        print(f"Starting process for iteration {iteration_num}, run {run}: {args}")
        # Create the subprocess with a timeout
        process = subprocess.Popen(args)
        # Wait for the process, up to the timeout
        timeout_seconds = 45 * 60  # 45 minutes in seconds
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"Iteration {iteration_num}, run {run}: Process timed out after 45 minutes. Terminating...")
        process.kill()  
    except Exception as e: 
        print(f"Iteration {iteration_num}, run {run}: An unexpected error occurred: {e}")


def load_and_average_results(model_num):  # Removed the aggregation_method
    """
    Loads and aggregates results from multiple pickle files, calculating the median.

    Args:
        model_num (int): The model number to load results for.

    Returns:
        pd.DataFrame: A DataFrame containing the median results.  Returns an empty
            DataFrame if no valid files are found.
    """
    dfs = []
    for run in range(3):
        model_path = f"data/model_results/model_{model_num}_results_run{run}.pkl"
        if os.path.exists(model_path):
            df = pd.read_pickle(model_path)
            # Debugging: Check for expected columns
            print(f"Columns in {model_path}: {df.columns}")
            dfs.append(df)
        else:
            print(f"Warning: {model_path} does not exist and will be skipped.")

    if not dfs:
        print(f"No valid DataFrames found for model {model_num}.")
        return pd.DataFrame()

    df_combined = pd.concat(dfs)
    # --- Median Calculation (with agg_funcs)---
    # Assign agg_funcs here, so it is *always* defined.
    agg_funcs = "median"  # Use the string "median" for future compatibility

    df_result = df_combined.groupby(df_combined.index)[
        ['Performance_siRNA_pH_4', 'Performance_siRNA_pH_8', 'Performance_double_membrane']
    ].agg(agg_funcs).copy() 



    # --- Column Handling ---
    performance_columns = ['Performance_siRNA_pH_4', 'Performance_siRNA_pH_8', 'Performance_double_membrane']
    columns_to_drop = [col for col in performance_columns if col in df_combined.columns]
    df_rest = df_combined.drop(columns=columns_to_drop).groupby(df_combined.index).first().copy()
    df_result = df_rest.join(df_result)

    return df_result

def kill_processes_by_name(process_name, target_pid=None):
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if proc.info['name'] == process_name:
            if target_pid is None or proc.info['pid'] == target_pid:  # Kill only if target_pid matches or is not provided
                try:
                    os.kill(proc.info['pid'], signal.SIGKILL)
                    print(f"Killed process {proc.info['pid']} with name {process_name}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    print(f"Failed to kill process {proc.info['pid']} with name {process_name}")
if __name__ == "__main__":
    while True:  # Main loop to retry if needed
        start_time = time.time()  # Capture start time
        for run in range(3):
            threads = []
            for thread_id in range(1):
                thread = threading.Thread(target=process_iteration, args=(thread_id, run))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            print(f"Finished run {run}")
        end_time = time.time()
        total_time = end_time - start_time
        print(f"All iterations completed in {total_time:.2f} seconds... killing leftover processes")
        kill_processes_by_name('gmx_plu')
        df_mean_results = []
        for model_num in range(1):
            print(model_num)
            df_mean = load_and_average_results(model_num)
            df_mean_results.append(df_mean)
        # Check if the list is still empty after the loop
        if not df_mean_results:
            print("No data loaded, retrying parallel processing...")
            # Reset start_time for accurate timing of the retry
            continue  # Go back to the beginning of the while loop
        break
    # Concatenate the averaged DataFrames vertically
    df_mean_results = [df.reset_index(drop=True) for df in df_mean_results if not df.empty]  # Ensure non-empty DataFrames
    df_concatenated = pd.concat(df_mean_results, ignore_index=True)
    # Handling NaN values by using forward fill which propagates last valid observation forward
    df_filled = df_concatenated.fillna(method='ffill')
    # Compute performance score

    x = df_filled['Performance_siRNA_pH_4']
    y = df_filled['Performance_siRNA_pH_8']
    z = df_filled['Performance_double_membrane']

    df_filled["performance_score"]=29.6052 * np.exp(-((x - (-40))**2 / (2 * 30**2) + (z - 155)**2 / (2 * 15**2) + (y - (-40))**2 / (2 * 25**2)))


    ############ Conversion block
    
    if iteration_number > 0:
        df_external = pd.read_pickle(f"data/top_performer_{iteration_number-1}.pkl")
        first_row = df_external.iloc[[0]]
        df_filtered = df_filled
        df_filtered = pd.concat([first_row, df_filtered], ignore_index=True)
        df_filtered.sort_values(by='performance_score', key=lambda x: x.abs(), ascending=False, inplace=True)
        top_performer = df_filtered.iloc[[0]].copy()  
        top_performer.to_pickle(f"data/top_performer_{iteration_number}.pkl")
    else:
        df_filtered = df_filled
        df_filtered.sort_values(by='performance_score', key=lambda x: x.abs(), ascending=False, inplace=True)
        top_performer = df_filtered.iloc[[0]].copy()  
        top_performer.to_pickle(f"data/top_performer_{iteration_number}.pkl")
    
    if 'performance_score' not in df_filled.columns:
        raise ValueError("The DataFrame does not contain a 'performance_score' column.")



    # Save the filtered DataFrame back to a pickle file
    df_filtered.to_pickle("data/DFfromRL.pkl")
    print(df_filtered["performance_score"])
    print("Successfully pickled")
    # Cleanup old model files
    for model_num in range(1):
        for run in range(3):
            model_path = f"data/model_results/model_{model_num}_results_run{run}.pkl"
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"Deleted {model_path}")
 
