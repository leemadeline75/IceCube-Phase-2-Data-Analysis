import pandas as pd
import json
from datetime import datetime
import os
import random
import numpy as np

def calculate_time_spent(metadata):
    try:
        meta = json.loads(metadata)
        started_at = datetime.fromisoformat(meta['started_at'].replace('Z', '+00:00'))
        finished_at = datetime.fromisoformat(meta['finished_at'].replace('Z', '+00:00'))
        time_spent = (finished_at - started_at).total_seconds()
        return time_spent
    except Exception:
        return None

# Set the working directory
#os.chdir(r"C:\Users\xomad\ICECUBE\Phase_2_data")


os.chdir(r"C:\Users\xomad\ICECUBE\data\output_retirement_lim20")

subj = pd.read_csv(r"C:\Users\xomad\ICECUBE\data\output_retirement_lim20\consolidated_data.csv")
df = pd.read_csv('name-that-neutrino-classifications.csv')

df = df[df['workflow_id'] == 27046] 
#23715


subj_ids = np.array(subj['subject_id'])

rows_to_keep = []
rows_to_filter = []
timespent = []

for index, row in df.iterrows():
    time_spent = calculate_time_spent(row.get("metadata", "{}"))

    if time_spent is None:
        print("ERROR TIME SPENT")
        continue  # Skip if time_spent could not be calculated
    
    if time_spent > 0:

        row_copy = row.copy()  # Make sure we're not modifying the original row
        row_copy["time_spent"] = time_spent  # Add the new column
        rows_to_keep.append(row_copy)
        #rows_to_keep.append(row)
        
    else:
        rows_to_filter.append(row)  


df_filtered_keep = pd.DataFrame(rows_to_keep)
df_filtered_keep = df_filtered_keep[df_filtered_keep['subject_ids'].isin(subj_ids)]
output_path = r"C:\Users\xomad\ICECUBE\data\output_retirement_lim20\filtered_time_6_seconds_20.csv"
df_filtered_keep.to_csv(output_path, index=False)


print(f"Filtered rows with time spent greater than 6 seconds saved to 'filtered_time_spent_greater_than_6_seconds.csv'")
