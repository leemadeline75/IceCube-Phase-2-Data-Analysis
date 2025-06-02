import pandas as pd
import os
import argparse
import sys
from reducer import Reducer
from phase2_data_analysis import consolidateData, makePlots, plot_stacked_histogram, plot_correct_fraction_histogram, plotuserenergy_2d, plotDNNenergy_2d, plot_accuracy_colored_by_confidence, scatterplot_iter2, plot_accuracy_heatmaps, userDNNaccuracy, accuracybycat_subset
from get_retired import getRetired
from energy_histo_plot import qtot2d_hist

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

# Code adapted to 2nd iteration due to change in classification data handling
# Madeline Lee
# 10/11/2024

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

        print(f'Performing phase 2 analysis for retirement limit of {lim}\n')
        outdir = f'../data/output_retirement_lim{lim}'
        if (os.path.isdir(os.path.join(os.getcwd(), outdir)) == False):                                 #create directory for plots, if not existent
            os.mkdir(os.path.join(os.getcwd(), outdir))
            print(f'    Directory {os.path.join(os.getcwd(), outdir)} created.\n')
        reducer = Reducer(outdir, outdir, lim)                                                          #initialize reducer
        print('     Extracting retired data...\n')
        #getRetired(exports_dir, outdir, lim)                                                            #extract retired
        print('     Running reducer...\n')
        #reducer.reduce()                                                                                #reduce the dataset

        user_data = pd.read_csv(os.path.join(outdir, 'user_consensus_data.csv'))                     #read in reduced data
        dnn_sim_data = pd.read_csv(os.path.join(outdir, r'..\matched_sim_data.csv'))      #read in classification data
        print('     Consolidating user, dnn, and mc data...\n')
        #consolidateData(user_data, dnn_sim_data, lim, outdir)#consolidate 
        
    
        result_consensus = pd.read_csv(os.path.join(outdir, 'consolidated_data.csv'))
        #result_consensus = pd.read_csv(os.path.join(outdir, 'results_consensus.csv'))
        print('     Creating plots...\n')
        #makePlots(result_consensus, lim, outdir)        #make plots
        
        print('Complete!')

        qtot = result_consensus['qtot'] #make histogram w accuracy
        qsig = result_consensus['signal_charge']
        qbg = result_consensus['bg_charge']
        accuracy = result_consensus['user_accuracy']
        energy = result_consensus['energy']
        userconf = result_consensus['data.agreement']
        DNNconfidence = result_consensus['max_score_val']
        DNNaccuracy = result_consensus['DNN_accuracy']
        ntn_category = result_consensus['ntn_category']
        #qratio = result_consensus['qratio']
        #plot_correct_fraction_histogram(qtot, accuracy)
        #qtot2d_hist(qtot, energy, accuracy, userconf)
        #plotuserenergy_2d(energy,userconf)
        #plotDNNenergy_2d(energy, DNNconfidence)
        #plot_accuracy_colored_by_confidence(qtot, accuracy, userconf)
        #scatterplot_iter2(qtot, energy, accuracy, userconf, qratio)
        #plot_accuracy_heatmaps(qtot, DNNaccuracy, DNNconfidence)
        #userDNNaccuracy(accuracy, DNNaccuracy, qtot)
        #accuracybycat1(accuracy,qtot,ntn_category, DNNaccuracy, num_bins=10)
#cascade bin size was 65 to 4250 0.165 
        bin_config = {
            'CASCADE': {'range': (65, 8500), 'bin_width': 0.195},
            'SKIMMING': {'range': (85, 415), 'bin_width': 0.065},
            'THROUGHGOINGTRACK': {'range': (95, 310), 'bin_width': 0.065},
            'STARTINGTRACK': {'range': (70, 500), 'bin_width': 0.13},
            'STOPPINGTRACK': {'range': (85, 180), 'bin_width': 0.065},}
        accuracybycat_subset(accuracy=accuracy,qtot=qtot, ntn_category=ntn_category, DNNaccuracy=DNNaccuracy, bin_config=bin_config)


