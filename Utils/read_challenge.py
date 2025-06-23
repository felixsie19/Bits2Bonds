import pandas as pd
import pickle


with open('SN4+TP1_TQ2p_h+SN3a_TQ2p_SN3a_TQ2p/model_1_results.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)
print(df.columns) 
print(df[['Performance_double_membrane', 'Performance_siRNA']])
