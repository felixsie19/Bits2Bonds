import pandas as pd
import threading
import subprocess
import time
import os
import signal
import psutil

iteration_number = int(os.environ.get('ITERATION_NUMBER'))
def process_iteration(iteration_num, run):
    try:
        args = ["python3", "chains.py", str(iteration_num), str(run)]
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
def load_and_average_results(model_num):
    dfs = []
    for run in range(3):
        model_path = f"model_results/model_{model_num}_results_run{run}.pkl"
        if os.path.exists(model_path):
            df = pd.read_pickle(model_path)
            dfs.append(df)
        else:
            print(f"Warning: {model_path} does not exist and will be skipped.")
    if not dfs:
        print(f"No valid DataFrames found for model {model_num}.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid files
    df_combined = pd.concat(dfs)
    df_mean_std = df_combined.groupby(df_combined.index)[['Performance_siRNA_pH_4','Performance_siRNA_pH_7', 'Performance_double_membrane']].agg(['mean', 'std'])
    # Ensure you're working on a copy
    df_mean = df_mean_std.xs('mean', axis=1, level=1).copy()
    # Add standard deviation columns safely
    df_mean['Performance_siRNA_pH_4_std'] = df_mean_std[('Performance_siRNA_pH_4', 'std')].values
    df_mean['Performance_double_membrane_std'] = df_mean_std[('Performance_double_membrane', 'std')].values
    df_mean['Performance_siRNA_pH_7_std'] = df_mean_std[('Performance_siRNA_pH_7', 'std')].values
    # List of columns to potentially drop
    columns_to_drop = ['Performance_siRNA_pH_4', 'Performance_double_membrane', 'Performance_siRNA_pH_7',
                       'Performance_siRNA_pH_4_std', 'Performance_double_membrane_std', 'Performance_siRNA_pH_7_std']
    # Only drop columns that exist in df_combined
    columns_to_drop = [col for col in columns_to_drop if col in df_combined.columns]
    # Reintegrate the rest of the columns
    df_rest = df_combined.drop(columns=columns_to_drop).groupby(df_combined.index).first().copy()
    # Merge df_mean with the rest of the columns
    df_mean = df_rest.join(df_mean)
    return df_mean
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
    df_filled["performance_score"]=(-0.14674291*df_filled['Performance_siRNA_pH_4']-0.05741036*df_filled['Performance_double_membrane']-0.01894954*df_filled['Performance_siRNA_pH_7'])**2    #only self interaction added 
    #df_filled["performance_score"]=(-1.0029492*df_filled['Performance_siRNA_pH_4']+0.70042235*df_filled['Performance_double_membrane']+0.9998894*df_filled['Performance_siRNA_pH_7'])+(-0.01080037*(df_filled['Performance_siRNA_pH_4']**2)+-0.00086912 *(df_filled['Performance_double_membrane']**2)+0.01473979*(df_filled['Performance_siRNA_pH_7']**2))
    #completely second degree with all interactions solved parameters
    #df_filled["performance_score"] = 0.22576037+1.0998312*df_filled['Performance_siRNA_pH_4']-0.31390563*df_filled['Performance_double_membrane']+0.17914405*df_filled['Performance_siRNA_pH_7']+0.36222485*(df_filled['Performance_siRNA_pH_4']**2)+0.06296889*(df_filled['Performance_siRNA_pH_4']*df_filled['Performance_double_membrane'])-0.22621362*(df_filled['Performance_siRNA_pH_4']*df_filled['Performance_siRNA_pH_7'])+0.00120991*(df_filled['Performance_double_membrane']**2)-0.05338603*(df_filled['Performance_double_membrane']*df_filled['Performance_siRNA_pH_7'])-0.12180143*(df_filled['Performance_siRNA_pH_7']**2)
    #### Calculation of std performance_score

    #### Calculation of std performance_score
    df_filled["performance_score_std"] = df_filled.apply(
        lambda row: ((3 * row['Performance_siRNA_pH_4_std'])**2 + row['Performance_siRNA_pH_7_std']**2)**0.5, axis=1)
    
    ############ Conversion block
    
    if iteration_number > 0:
        df_external = pd.read_pickle(f"top_performer_{iteration_number-1}.pkl")
        first_row = df_external.iloc[[0]]
        df_filtered = df_filled
        df_filtered = pd.concat([first_row, df_filtered], ignore_index=True)
        df_filtered.sort_values(by='performance_score', key=lambda x: x.abs(), ascending=True, inplace=True)
        top_performer = df_filtered.iloc[[0]].copy()  
        top_performer.to_pickle(f"top_performer_{iteration_number}.pkl")
    else:
        df_filtered = df_filled
        df_filtered.sort_values(by='performance_score', key=lambda x: x.abs(), ascending=True, inplace=True)
        top_performer = df_filtered.iloc[[0]].copy()  
        top_performer.to_pickle(f"top_performer_{iteration_number}.pkl")
    # Save the filtered DataFrame back to a pickle file
    df_filtered.to_pickle("DFfromRL.pkl")
    print(df_filtered["performance_score"])
    print("Successfully pickled")
    # Cleanup old model files
    for model_num in range(1):
        for run in range(3):
            model_path = f"model_results/model_{model_num}_results_run{run}.pkl"
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"Deleted {model_path}")
