#Code to determine user accuracy and number of correct classifications
import pandas as pd
import os

#input_dir = '/Users/evamankos/Documents/Python' #CHANGE directory 

#input_dir = '/Users/xomad/ICECUBE/data/output_retirement_lim15'

input_dir = "/Users/xomad/ICECUBE/data/output_retirement_lim15"


#"C:\Users\xomad\ICECUBE\data\output_retirement_lim15\name-that-neutrino-classifications.csv"

# Read data from excel files (classif being file of origin, correct_classif being correct data set)
classif = pd.read_excel(os.path.join(input_dir, 'classified_output.xlsx'))  
#classif = pd.read_csv(os.path.join(input_dir, 'name-that-neutrino-classifications.csv'))  
correct_classif = pd.read_csv(os.path.join(input_dir, 'consolidated_data_raw.csv')) 

# Create a dictionary for correct answers by subject to lookup within
correct_lookup = dict(zip(correct_classif['subject_id'], correct_classif['ntn_category']))


# Create dictionary of all user accuracies
user_accuracies = {}
count = 0


# Iterate over each classification row
for index, row in classif.iterrows():
    user_name = row['user_name']
    subject_id = row['subject_ids']
    user_answer = row['user_classification']
    
    correct_answer = correct_lookup.get(subject_id)

    # Initialize user stats
    if user_name not in user_accuracies:
        user_accuracies[user_name] = {'correct': 0, 'total': 0}  
    #add one to total for every iteration
    user_accuracies[user_name]['total'] += 1    
    
    
    # Determine correctness (treat missing answer as incorrect)
    if correct_answer is not None:
        is_correct = int(user_answer == correct_answer) #evaluates to 1 if true, 0 if false

    # Update correct values
    user_accuracies[user_name]['correct'] += is_correct


# Format results for export
export_data = []
for user_name, stats in user_accuracies.items():
    total = stats['total']
    correct = stats['correct']
    accuracy = (correct / total) * 100
    rounded_accuracy = round(accuracy, 2)
    #To remove users with under 5 classifications before including in excel sheet
    if total >=5: 
       #format for excel sheet
       export_data.append({
            'Username': user_name,
           'Correct': correct,
           'Total Classifications': total,
           'Accuracy %': rounded_accuracy})

# Export to new Excel sheet
df_output = pd.DataFrame(export_data)
output_path = os.path.join(input_dir, 'user_accuracy_results.xlsx')
df_output.to_excel(output_path, index=False)

print(f"User accuracy results saved to: {output_path}")
