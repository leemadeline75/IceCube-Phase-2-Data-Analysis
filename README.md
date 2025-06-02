## IceCube-Phase-2-Data-Analysis repository

# Author: Madeline Lee

Here is a collection of code for analysis of the Name that Neutrino Zooniverse data. This code has been mostly written by Elizabeth Warrick and Andrew Phillips. I have edited this code to be applied to the second iteration of this project. 


Reducing NTN data exports
The relevant files are located in the iteration2 directory. The following code is to get from the raw NtN data export files name-that-neutrino-classifications, name-that-neutrino-workflows, and name-that-neutrino-subjects to a consolidated_data.csv file and confusion matrices

name-that-neutrino-classifications: classification level data (ex, graham classifed event 10289368 as a cascade at this time), includes user name, ip address, workflow, date, start time, end time, users choice (what they picked), and subject id

name-that-neutrino-workflows: information about the current and previous workflows, including primary language and retirement limit

name-that-neutrino-subjects: subject data, information about each event/video (ex 7840), can be used as definitive list of all subject ids

get_retired.py
The input for this files are the raw Ntn data exports, and the output is three new files, also named name-that-neutrino-classifications, name-that-neutrino-workflows, and name-that-neutrino-subjects that only includes classifications for the second iteration, and excludes classifications done above the specified retirement limit

reducer.py
The input files are the output files of get_retired: name-that-neutrino-classifications, name-that-neutrino-workflows, and name-that-neutrino-subjects (they should be in a different location than the raw NtN files). reducer.py is where the user classifications counted to determine the winning category. The subject id, data_num_votes (number of votes the winning classification has), data_most_likely (winning category), and data.agreement (user confidence, ex 11/15 votes) are saved to a new file user_consensus_data.csv. Keep in mind this solely contains the user data, the DNN and Simulation data come from the i3 files and will need to be matched to the user data using the directory iteration2matching

phase2_data_analysis_2.py


do_analysis.py
This is a script that wraps everything together. You can use it to run phase1_data_analysis.py on a series of different retirement limits and auto-generate plots.
