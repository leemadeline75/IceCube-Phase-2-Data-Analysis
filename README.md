## IceCube-Phase-2-Data-Analysis repository

# Author: Madeline Lee

Here is a collection of code for analysis of the second iteration of Name that Neutrino Zooniverse data. Some of this code was written by Elizabeth Warrick and Andrew Phillips. I have edited this code to be applied to the second iteration of this project. 


Reducing NTN data exports
The relevant files are located in the iteration2 directory. The following code is to get from the raw NtN data export files name-that-neutrino-classifications, name-that-neutrino-workflows, and name-that-neutrino-subjects to a consolidated_data.csv file and confusion matrices

### name-that-neutrino-classifications

classification level data (ex, graham classifed event 10289368 as a cascade at this time), includes user name, ip address, workflow, date, start time, end time, users choice (what they picked), and subject id

### name-that-neutrino-workflows

information about the current and previous workflows, including primary language and retirement limit

### name-that-neutrino-subjects

subject data, information about each event/video (ex 7840), can be used as definitive list of all subject ids

## get_retired.py

The input for this files are the raw Ntn data exports, and the output is three new files, also named name-that-neutrino-classifications, name-that-neutrino-workflows, and name-that-neutrino-subjects that only includes classifications for the second iteration, and excludes classifications done above the specified retirement limit

## reducer.py

The input files are the output files of get_retired: name-that-neutrino-classifications, name-that-neutrino-workflows, and name-that-neutrino-subjects (they should be in a different location than the raw NtN files). reducer.py is where the user classifications counted to determine the winning category. The subject id, data_num_votes (number of votes the winning classification has), data_most_likely (winning category), and data.agreement (user confidence, ex 11/15 votes) are saved to a new file user_consensus_data.csv. Keep in mind this solely contains the user data, the DNN and Simulation data come from the i3 files and will need to be matched to the user data using match_dnn_user

## phase2_data_analysis_2.py

The input files for phase2_data_analysis_2.py are user_consensus_data.csv (the users data) and combined_sim_DNN_data.csv (the DNN and simulation data). The consolidateData function takes in the information from both files, matches them based on subject id, and removes irrelevant columns that may have been carried over. Additionally, this is where any filters on qratio, qtot, or data.agreement (user confidence) can be done. This saves the desired info into a new file, consolidated_data

The file consolidated_data is then used to make the confusion matrices and for further analysis

# do_analysis.py
This is a script that runs everything together. This is where you specify and input and output, and retirement limit. Line 73 specifies the completion of the above analyis, the remaining lines I used for further analysis.


## Match_DNN_User
For the second iteration, the DNN and simulation data was not added to the name-that-neutrino-subjects file, and needed to be manually added from the i3 files. The i3 files are from the IceCube cobalt server and contain all the relevant simulation and DNN data (such as the DNN classification and the truth label). I was provided the i3 files for the second iteration events, and the following is how I matched them to each event.

1. Extract the run number, event number, and filename from each unique subject/event in the Zooniverse data file and save it as a new file, subject_data.csv
   
2. Combine all DNN/simulation data files from a directory and save them into one CSV, combined_sim_DNN_data.csv

3. Match dnn/sim to user data based on run, event, and origidx value. The DNN/simulation data is only labeled by run number and event number. On Zooniverses end (the user data), each event has a run number, event number, and origidx number. Each combination of run, event, and origidx is entirely unique to that event, some events have the same run and event, making them impossible to match to corresponding DNN and simulation data. For this reason, I only included event with a unique run and event value, any events with the same run/event combo were excluded from further analysis, bringing the total number of events down from 7840 to 6385.
   
combined_sim_DNN_data.csv contains the DNN and simulation data matched to their appropriate subject id, and is now used in phase2_data_analysis. This should only need to be done once!


# consolidated_data by column

### subject_id
9 digit number assigned to an event by Zooniverse, ex 101504604

### data.num_votes
**User Data** the number of votes the "winning" category has

### data.most_likely
**User Data** the "winning" category that recieved the highest number of votes, if listed as 0,1,2,3,4 mapping is under ntn_category

### data.agreement
**User Data** the ratio of votes the "winning" category got to the total number of votes, ex 11/15 = 0.73

### filename
full filename that includes the run, event, and origidx numbers, run and event are also separate columns

### truth_classification
**Simulation Data** True event topology out of the following (+ mapping to the five NtN topologies):

unclassified = 0  
throughgoing_track = 1                 ---> 2 ---> THROUGHGOINGTRACK  
starting_track = 2                     ---> 3 ---> STARTINGTRACK  
stopping_track = 3                     ---> 4 ---> STOPPINGTRACK  
skimming_track = 4                     ---> 0 ---> SKIMMING  
contained_track = 5    
contained_em_hadr_cascade = 6          ---> 1 ---> CASCADE  
contained_hadron_cascade = 7           ---> 1 ---> CASCADE  
uncontained_cascade = 8                ---> 0 ---> SKIMMING  
glashow_starting_track = 9  
glashow_electron = 10  
glashow_tau_double_bang = 11  
glashow_tau_lollipop = 12  
glashow_hadronic = 13  
throughgoing_tau = 14  
skimming_tau = 15  
double_bang = 16  
lollipop = 17  
inverted_lollipop = 18  
throughgoing_bundle = 19               ---> 2 ---> THROUGHGOINGTRACK
stopping_bundle = 20                   ---> 4 ---> STOPPINGTRACK
tau_to_mu = 21

### pred_skim,	pred_cascade,	pred_tgtrack,	pred_starttrack,	pred_stoptrack
**DNN Data** DNN confidence of whether that event belongs to the five topologies

### energy,	zenith,	oneweight
**Simulation Data** properties of the event given by the simulation

### signal_charge
**Simulation Data** value of charge deposited in the detector that came from signal (neutrino) from NuGen

### bg_charge
**Simulation Data** value of charge deposited in the detector that came from background (ex cosmic rays) from Corsika

### qratio
**Simulation Data** measured as signal_charge / (signal_charge + bg_charge), provides a measure of the signals contribution to the total charge of the event

### qtot
**Simulation Data** sum of signal_charge, bg_charge, and Qnoise(not in file), is the value of total charge deposited in the detector

### max_score_val
**DNN Data** value of highest DNN confidence out of the 5 topologies

### idx_max_score
**DNN Data** category that has the highest DNN confidence, the DNNs classification of that event, mapped by:

pred_skim: SKIMMING
pred_cascade: CASCADE
pred_tgtrack: THROUGHGOINGTRACK
pred_starttrack = STARTINGTRACK
pred_stoptrack = STOPPINGTRACK

### ntn_category
**Simulation Data** truth label / "correct answer", which out of the 5 topologies the event actually should be classified as

0 = SKIMMING
1 = CASCADE
2 = THROUGHGOINGTRACK
3 = STARTINGTRACK
4 = STOPPINGTRACK

### user_accuracy
1 if users classified event correctly, 0 if users classified incorrectly, determined if data.most_likely matches ntn_category

### DNN_accuracy
1 if the DNN classified event correctly, 0 is DNN classified incorrectly, determined if idx_max_score matches ntn_category







