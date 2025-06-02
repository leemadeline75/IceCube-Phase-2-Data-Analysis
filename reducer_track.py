import pandas as pd
import numpy as np
import argparse
import json
import os, os.path

##############################################################################################
#                                       reducer.py
##############################################################################################
# Purpose: Takes in classification list, and reduces into user consensus choices and vote totals
# Usage: python reduce.py <input_dir> <output_dir>
# Author: Andrew Phillips
# Date: 4/2/24

# Code adapted to 2nd iteration due to change in classification data handling
# Madeline Lee
# 10/11/2024

##############################################################################################


class Reducer:

    def __init__(self, input_dir, output_dir, retirement_lim):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.retirement_lim = retirement_lim

    def reduce_track(self):

        lim = self.retirement_lim                                                            #extract retirement limit
        input_dir = os.path.join(os.getcwd(), self.input_dir)                                    #extract input directory (where to get ntn data exports)
        output_dir = os.path.join(os.getcwd(), self.output_dir)                                  #extract output directory (where to put modified ntn datasets)

        ''' Read in data '''
        classif = pd.read_csv(os.path.join(input_dir, 'name-that-neutrino-classifications.csv')) #read in classification set
        subj = pd.read_csv(os.path.join(input_dir, 'name-that-neutrino-subjects.csv'))     
        matched = pd.read_csv(os.path.join(input_dir, r'..\matched_sim_data.csv'))  #read in subject set


        ''' Set up hash map '''
        #create hash map for subject ids, where each entry is a dict with the 5 categories.
        subj_ids = np.array(matched['subject_id'])
        #subj_set_ids = np.array(matched['subject_set_id'])
        event_ids = subj_ids
        subj_dict = {}                                                                           #create empty dict
        for id in subj_ids:
            subj_dict[id] = {'TRACK': 0, 'CASCADE': 0, 'SKIMMING': 0}  #initialize dict key to subj id, value to 0

        subj_data = np.array(classif['subject_data'])                                           #read in subject data
        annotations = np.array(classif['annotations'])  #read in annotation (tells us what the user picked)

        
        '''loop over all the classifications, extract user choice '''
        for i in range(0, len(subj_data)):  
            metadata = json.loads(subj_data[i])
            annot = json.loads(annotations[i])
            key = int(list(metadata.keys())[0])

    #extract user choice ML
    #accounts for three types: cascade, skimming, track and three subtypes of tracks: starting, stopping, and through-going ML
            # Uses same hash map situation as previous code, should handle the different classification method the same way ML

            if key in subj_dict.keys():
            #if annot:
                user_choice = annot[0]['value'][0]['choice'] # Get the user choice from the first item

                if user_choice in ['CASCADE', 'SKIMMING']: # Increases count if user choice is cascade or skimming ML
                    subj_dict[key][user_choice] += 1
                elif user_choice == 'TRACK':  #If the type was track, then asks what type of track
                    subj_dict[key]['TRACK'] += 1 
                        
          
                #now get the winning categorizations
        '''Extract winning category, its number of votes, and user agreement'''
        MAX_VOTES = []
        MOST_LIKELY = []
        AGREEMENT = []
        for key in subj_dict.keys():
            max_votes = 0
            most_likely = None
            for cat in subj_dict[key].keys(): #find which category had the most votes
                if subj_dict[key][cat] > max_votes:
                    max_votes = subj_dict[key][cat]
                    most_likely = cat
                    
            MAX_VOTES.append(max_votes)
            MOST_LIKELY.append(most_likely)
            AGREEMENT.append(max_votes/lim)
            
        data = {'subject_id': subj_ids, 'event_id':event_ids, 'data.num_votes': MAX_VOTES, 'data.most_likely':MOST_LIKELY, 'data.agreement':AGREEMENT}
            
        df = pd.DataFrame(data)
        csv_name = os.path.join(output_dir, 'user_consensus_data.csv')
        
        df.to_csv(csv_name, index=False)
        return csv_name

if __name__ == '__main__':

    ''' Parsing command line args '''
    parser = argparse.ArgumentParser(
                    prog='reducer.py',
                    description='Reduction script for NtN data')
    parser.add_argument('retirement_lim', metavar='lim', type=int, nargs='+',
                    help='desired retirement limit')
    parser.add_argument('in_dir', metavar='indir', type=str, nargs=1, 
                    help='input directory')
    parser.add_argument('out_dir', metavar='outdir', type=str, nargs=1, 
                    help = 'output directory')

    args = parser.parse_args()
    lim = args.retirement_lim[0]                                                             #extract retirement limit
    input_dir = os.path.join(os.getcwd(), args.in_dir[0])                                    #extract input directory (where to get ntn data exports)
    output_dir = os.path.join(os.getcwd(), args.out_dir[0])                                  #extract output directory (where to put modified ntn datasets)

    reducer_instance = Reducer(input_dir, output_dir, lim)
    reducer_instance.reduce_track()                                               #run reducer func