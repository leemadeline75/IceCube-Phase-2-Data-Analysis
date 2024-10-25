import pandas as pd
import os
import argparse
from reducer import Reducer
from phase1_data_analysis import consolidateData, makePlots
from get_retired import getRetired

##############################################################################################
#                                   do_analysis.py 
##############################################################################################
# Purpose: perform analysis on aggregated NtN data
# Usage: python do_analysis.py <retirement_limit(s)> <output_directory>
# Arguments:    
#           retirement_limit(s) -> int: Desired retirement limit for data. Can be singular or
#            a list
#           output_directory -> string: Directory where you want all output (csvs, plots) to go
# Requirements: reducer.py, get_retired.py, phase1_data_analysis.py

#In Summary ML
# get_retired.py is used to extract the classification data at the time at which a particular retirement was achieved, sorts through users inputs
# reducer.py extracts only the relevant information from our ntn data exports (current workflow) This will give a new csv tabulating the subject id of every subject along with the user consensus choice, and the fraction agreement
# phase1_data_analysis.py performs analysis of the resultant dataset, should take into account MC classifications and data, this is where an additional step is needed to read the data from the separate i3 files

# do_analysis runs everything together, calls reducer to filter only current workflow, get_retired to count the classifications, then phase1_data_analysis to compare it to the DNN






#
##############################################################################################

if __name__ == '__main__':

    ''' Parsing command line args '''
    parser = argparse.ArgumentParser(
                    prog='phase1_data_analysis',
                    description='Data anlysis for phase 1 name that neutrino data')
    parser.add_argument('retirement_lims', metavar='lim', type=int, nargs='+',
                    help='desired retirement limit(s) (can put multiple)')
    parser.add_argument('exports_dir', metavar='exp_dir', type=str, nargs=1, 
                    help='location of raw NTN data exports')
    

    args = parser.parse_args()
    retirement_lims = args.retirement_lims                                                              #extract retirement lims desired
    exports_dir = os.path.join(os.getcwd(), args.exports_dir[0])                                        #location of raw NTN data exports

    for lim in retirement_lims:

        print(f'Performing phase 1 analysis for retirement limit of {lim}\n')
        outdir = f'../data/output_retirement_lim{lim}'
        if (os.path.isdir(os.path.join(os.getcwd(), outdir)) == False):                                 #create directory for plots, if not existent
            os.mkdir(os.path.join(os.getcwd(), outdir))
            print(f'    Directory {os.path.join(os.getcwd(), outdir)} created.\n')
        reducer = Reducer(outdir, outdir, lim)                                                          #initialize reducer
        print('     Extracting retired data...\n')
        getRetired(exports_dir, outdir, lim)                                                            #extract retired
        print('     Running reducer...\n')
        reducer.reduce()                                                                                #reduce the dataset

        data_consensus = pd.read_csv(os.path.join(outdir, 'consensus_reduced.csv'))                     #read in reduced data
        ntn_subjects = pd.read_csv(os.path.join(outdir, 'name-that-neutrino-classifications.csv'))      #read in classification data
        print('     Consolidating user, dnn, and mc data...\n')
        consolidateData(data_consensus, ntn_subjects, lim, outdir)                                      #consolidate 

        result_consensus = pd.read_csv(os.path.join(outdir, 'ntn-result-consensus.csv'))
        print('     Creating plots...\n')
        makePlots(result_consensus, lim, outdir)                                                        #make plots
        print('Complete!')

