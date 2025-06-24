import pandas as pd
import numpy as np
import argparse
import json
import os, os.path

##############################################################################################
#                                       reducer.py
##############################################################################################
# 

##############################################################################################


def reduce():

    # Read in the first file
    subj = pd.read_csv(r"C:\Users\xomad\ICECUBE\data\output_retirement_lim20\consolidated_data_55.csv")

    # Read in the second file
    #classif = pd.read_csv(r"C:\Users\xomad\ICECUBE\data\filtered_time_spent_greater_than_6_seconds.csv")
    classif = pd.read_csv(r"C:\Users\xomad\ICECUBE\data\output_retirement_lim20\filtered_time_6_seconds_20.csv")

    # Read in the user accuracy scores (replace with your actual file path and column names)
    user_acc_df = pd.read_excel(r"C:\Users\xomad\ICECUBE\data\user_accuracy_results.xlsx")  
    user_acc_dict = dict(zip(user_acc_df['Username'], user_acc_df['Accuracy']))

# Initialize dictionary for subject votes
    subj_ids = np.array(subj['subject_id'])
    subj_dict = {id: {'THROUGHGOINGTRACK': 0, 'STOPPINGTRACK': 0, 'STARTINGTRACK': 0, 'CASCADE': 0, 'SKIMMING': 0} for id in subj_ids}

    # Extract subject data, annotations, and user names
    subj_data = np.array(classif['subject_data'])
    annotations = np.array(classif['annotations'])
    users = np.array(classif['user_name']) 

    for i in range(len(subj_data)):
        metadata = json.loads(subj_data[i])
        annot = json.loads(annotations[i])
        user = users[i]

        # Get subject key
        key = int(list(metadata.keys())[0])

        # Get user accuracy and weight
        accuracy = user_acc_dict.get(user, 0)  # Default to 0 if user not found
        if accuracy > 0.3:
            weight = 1
        elif accuracy >= 0.2:
            weight = 1
        else:
            weight = 0

        # Apply weighted vote
        if key in subj_dict:
            try:
                user_choice = annot[0]['value'][0]['choice']
                if user_choice in ['CASCADE', 'SKIMMING']:
                    subj_dict[key][user_choice] += weight
                elif user_choice == 'TRACK':
                    track_type = annot[0]['value'][0]['answers'].get('WHATTYPEOFTRACKISIT')
                    if track_type in subj_dict[key]:
                        subj_dict[key][track_type] += weight
            except Exception as e:
                print(f"Skipping record {i} due to error: {e}")

    # Build output data
    MAX_VOTES, MOST_LIKELY, AGREEMENT, TOTAL = [], [], [], []

    for key in subj_ids:
        cat_votes = subj_dict[key]
        total_votes = sum(cat_votes.values())
        max_votes = max(cat_votes.values())
        most_likely = max(cat_votes, key=cat_votes.get)
        agreement = max_votes / total_votes if total_votes > 0 else 0

        MAX_VOTES.append(max_votes)
        MOST_LIKELY.append(most_likely)
        AGREEMENT.append(agreement)
        TOTAL.append(total_votes)

    # Create DataFrame and export
    data = {
        'subject_id': subj_ids,
        'data.num_votes': MAX_VOTES,
        'data.most_likely': MOST_LIKELY,
        'data.agreement': AGREEMENT,
        'total.votes': TOTAL
    }

    df = pd.DataFrame(data)
    df = df[df['total.votes'] >= 10]
    df.to_csv(r'C:\Users\xomad\ICECUBE\data\output_retirement_lim20\user_consensus_data_weighted_2.csv', index=False)

    return
                                        #run reducer func

reduce()