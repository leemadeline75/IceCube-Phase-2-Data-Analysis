import pandas as pd
import os
import sys
import json #reading java strings into python dictionaries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import re
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.colors as mcolors
import scipy.stats as stats

##############################################################################################
#                                   phase2_data_analysis.py 
##############################################################################################
# Code adapted to 2nd iteration due to change in classification data handling
# Madeline Lee
# 10/11/2024


''' Function to generate confusion matrix labels '''

def getUncertaintyLabels(matrix_df):
    labels = []
    nrows, ncols = matrix_df.shape
    for i in range(0,nrows):
        col_sum = matrix_df.iloc[:,i].sum()
        for k in range(0,ncols):
            val = matrix_df.iloc[k,i]
            n_val = (val/col_sum.astype(float))*100
            val_percent_round = "%.1f" % n_val
            lab_n = '%.1f%%'%n_val
            lab_v = '%d/%d'%(val,col_sum)
            lab='%.1f\%%\n%d/%d' % (n_val, val, col_sum)
            labelc= str(lab_n)+"\n"+str(lab_v)
            label = f"{str(lab_n)} {str(lab_v)}"
            labels.append(str(lab))
    labels_array = np.asarray(labels)
    labels_val = labels_array.reshape(5,5)
    labels_new = labels_val.T
    return labels_new



def consolidateData(user_data, dnn_sim_data, retirement_lim, outdir):
    # Lists to store user data
    user_filenames = []
    user_runs = []
    user_events = []
    user_subj_ids = []
    #user_subject_set_ids = []
    user_num_votes = []
    user_most_likely = []
    user_agreement = []

    # iterate through the user data
    for i in range(len(user_data)):
        subject_id = user_data['subject_id'][i]
        event_id = user_data['event_id'][i]
        #subject_set_id = user_data['subject_set_id'][i]
        num_votes = user_data['data.num_votes'][i]
        most_likely = user_data['data.most_likely'][i]
        agreement = user_data['data.agreement'][i]

        # Add to lists for later DataFrame creation
        user_subj_ids.append(subject_id)
        user_filenames.append(f"subject_{subject_id}_event_{event_id}.txt")  # Example filename format
        user_runs.append(None)  # Set None for now; modify as needed
        user_events.append(event_id)
        #user_subject_set_ids.append(subject_set_id)
        user_num_votes.append(num_votes)
        user_most_likely.append(most_likely)
        user_agreement.append(agreement)

    # Create DataFrame from user data
    subj_user_data = pd.DataFrame({
        'subject_id': user_subj_ids,
        'filename': user_filenames,
        'run': user_runs,
        'event': user_events,
        #'subject_set_id': user_subject_set_ids,
        'data.num_votes': user_num_votes,
        'data.most_likely': user_most_likely,
        'data.agreement': user_agreement
    })

    # Lists to store data from DNN and simulation data
    dnn_subj_ids = []
    dnn_runs = []
    dnn_events = []
    dnn_filenames = []
    dnn_truth_classification = []
    dnn_pred_skims = []
    dnn_pred_cascades = []
    dnn_pred_tgtracks = []
    dnn_pred_starttracks = []
    dnn_pred_stoptracks = []
    dnn_energies = []
    dnn_zeniths = []
    dnn_oneweights = []
    dnn_signal_charge = []
    dnn_bg_charge = []
    dnn_qratio = []
    dnn_qtot = []
    dnn_max_score_vals = []
    dnn_idx_max_scores = []
    dnn_ntn_category = []

    # collect DNN and simulation data
    for i in range(len(dnn_sim_data)):
        subject_id = dnn_sim_data['subject_id'][i]
        run = dnn_sim_data['run'][i]
        event = dnn_sim_data['event'][i]
        filename = dnn_sim_data['filename'][i]
        truth_classification = dnn_sim_data['truth_classification'][i]
        pred_skim = dnn_sim_data['pred_skim'][i]
        pred_cascade = dnn_sim_data['pred_cascade'][i]
        pred_tgtrack = dnn_sim_data['pred_tgtrack'][i]
        pred_starttrack = dnn_sim_data['pred_starttrack'][i]
        pred_stoptrack = dnn_sim_data['pred_stoptrack'][i]
        energy = dnn_sim_data['energy'][i]
        zenith = dnn_sim_data['zenith'][i]
        oneweight = dnn_sim_data['oneweight'][i]
        signal_charge = dnn_sim_data['signal_charge'][i]
        bg_charge = dnn_sim_data['bg_charge'][i]
        qratio = dnn_sim_data['qratio'][i]
        qtot = dnn_sim_data['qtot'][i]
        max_score_val = dnn_sim_data['max_score_val'][i]
        idx_max_score = dnn_sim_data['idx_max_score'][i]
        ntn_category = dnn_sim_data['ntn_category'][i]

        # Add to lists for later DataFrame creation
        dnn_subj_ids.append(subject_id)
        dnn_runs.append(run)
        dnn_events.append(event)
        dnn_filenames.append(filename)
        dnn_truth_classification.append(truth_classification)
        dnn_pred_skims.append(pred_skim)
        dnn_pred_cascades.append(pred_cascade)
        dnn_pred_tgtracks.append(pred_tgtrack)
        dnn_pred_starttracks.append(pred_starttrack)
        dnn_pred_stoptracks.append(pred_stoptrack)
        dnn_energies.append(energy)
        dnn_zeniths.append(zenith)
        dnn_oneweights.append(oneweight)
        dnn_signal_charge.append(signal_charge)
        dnn_bg_charge.append(bg_charge)
        dnn_qratio.append(qratio)
        dnn_qtot.append(qtot)
        dnn_max_score_vals.append(max_score_val)
        dnn_idx_max_scores.append(idx_max_score)
        dnn_ntn_category.append(ntn_category)

    # Create DataFrame from DNN simulation data
    dnn_data = pd.DataFrame({
        'subject_id': dnn_subj_ids,
        'filename': dnn_filenames,
        'run': dnn_runs,
        'event': dnn_events,
        'truth_classification': dnn_truth_classification,
        'pred_skim': dnn_pred_skims,
        'pred_cascade': dnn_pred_cascades,
        'pred_tgtrack': dnn_pred_tgtracks,
        'pred_starttrack': dnn_pred_starttracks,
        'pred_stoptrack': dnn_pred_stoptracks,
        'energy': dnn_energies,
        'zenith': dnn_zeniths,
        'oneweight': dnn_oneweights,
        'signal_charge': dnn_signal_charge,
        'bg_charge': dnn_bg_charge,
        'qratio': dnn_qratio,
        'qtot': dnn_qtot,
        'max_score_val': dnn_max_score_vals,
        'idx_max_score': dnn_idx_max_scores,
        'ntn_category': dnn_ntn_category
    })

    # Merge the user data and DNN simulation data on 'subject_id'
    result_consensus = pd.merge(subj_user_data, dnn_data, on='subject_id', how='outer')

    # Drop unwanted columns
    result_consensus = result_consensus.drop(columns=[
        'filename_x',  # Drop filename from user data
        'run_x',        # Drop run from user data
        'event_x',      # Drop event from user data
        'run_y',        # Drop run from DNN data
        'event_y'       # Drop event from DNN data
    ])


###################### to create accuracy column#######################
    # Map 'data.most_likely' to numerical categories
   # Create a mapping from integer labels to string labels
    label_mapping = {
    0: 'SKIMMING',
    1: 'CASCADE',
    2: 'THROUGHGOINGTRACK',
    3: 'STARTINGTRACK',
    4: 'STOPPINGTRACK'
}

# Apply the mapping to 'ntn_category'
    result_consensus['ntn_category'] = result_consensus['ntn_category'].map(label_mapping)
    
    result_consensus['user_accuracy'] = (result_consensus['data.most_likely'] == result_consensus['ntn_category']).astype(int)
    
    DNNlabel_mapping = {
    'pred_skim': 'SKIMMING',
    'pred_cascade': 'CASCADE',
    'pred_tgtrack': 'THROUGHGOINGTRACK',
    'pred_starttrack': 'STARTINGTRACK',
    'pred_stoptrack': 'STOPPINGTRACK'
}

    result_consensus['idx_max_score'] = result_consensus['idx_max_score'].map(DNNlabel_mapping)
    result_consensus['DNN_accuracy'] = (result_consensus['idx_max_score'] == result_consensus['ntn_category']).astype(int)

# ONLY USING NEXT FEW LINES FOR FILTERING USER CONFIDENCE > 55% AND QRATIO BETWEEN 0.05 AND 0.95

#    result_consensus_qtot = result_consensus[
#        (result_consensus['qtot'] >= 200) |
#        ((result_consensus['qtot'] <= 250) & (result_consensus['data.agreement'] > 0.55))
#    ]
    
 #   result_consensus_filtered = result_consensus[result_consensus['data.agreement'] >= 0.55]

    #result_consensus_qtot = result_consensus_filtered[   (result_consensus_filtered['qtot'] >= 0) & (result_consensus_filtered['qtot'] <= 300)]





#    result_consensus_qtot = result_consensus_filtered[
#(
#    (result_consensus_filtered['ntn_category'] == 'SKIMMING') &
#    (result_consensus_filtered['qtot'] >= 80) &
#    (result_consensus_filtered['qtot'] <= 150)
#)|
#(
#    (result_consensus_filtered['ntn_category'] == 'CASCADE') &
#    (result_consensus_filtered['qtot'] >= 2000) &
#    (result_consensus_filtered['qtot'] <= 10000)
#)|
#(
#    (result_consensus_filtered['ntn_category'] == 'THROUGHGOINGTRACK') &
#    (result_consensus_filtered['qtot'] >= 160) &
#    (result_consensus_filtered['qtot'] <= 210)
#)
#]



# Apply the filter for qratio < 0.05 or qratio > 0.95
#    result_consensus_qcut = result_consensus_filtered[
#    (result_consensus_filtered['qratio'] < 0.05) | (result_consensus_filtered['qratio'] > 0.95)
#]   
    # Save the consolidated data to a CSV file
    csv_name = os.path.join(outdir, 'consolidated_data.csv')
    result_consensus_filtered.to_csv(csv_name, index=False)

    return csv_name


def makePlots(result_consensus, retirement_lim, outdir):

    
# Ensure that 'data.agreement' is numeric (e.g., float)
    result_consensus['data.agreement'] = pd.to_numeric(result_consensus['data.agreement'], errors='coerce')
    print(len(result_consensus))

# Ensure 'max_score_val' is numeric (e.g., float)
    result_consensus['max_score_val'] = pd.to_numeric(result_consensus['max_score_val'], errors='coerce')

# Ensure 'truth_classification' is categorical (e.g., string or category type)
    result_consensus['ntn_category'] = result_consensus['ntn_category'].astype('category')

# Ensure 'idx_max_score' is categorical (e.g., string or category type)
    result_consensus['idx_max_score'] = result_consensus['idx_max_score'].astype('category')

    types = ['Skimming', 'Cascade', 'Through-Going\nTrack', 'Starting\nTrack', 'Stopping\nTrack']

    if not os.path.isdir(os.path.join(os.getcwd(), outdir, 'plots')):  # create directory for plots if not existent
        os.mkdir(os.path.join(os.getcwd(), outdir, 'plots'))

    '''Make bin labels, digitize data'''
    # Get the possible value of the consensus voter fractions
    x = result_consensus['data.agreement'].value_counts(ascending=False).keys().tolist()
    xtick_labels = []
    min_val = retirement_lim / 5
    bin_edges = np.histogram_bin_edges(result_consensus['data.agreement'][:], bins=int(retirement_lim * 4 / 5) + 1, range=((min_val - 0.5) / retirement_lim, (retirement_lim + 0.5) / retirement_lim))

    for i in np.arange(0, len(bin_edges)):
        xtick_labels.append("{:0.2f}".format(bin_edges[i]))

    # Digitize the agreement values into bins
    np.digitize(result_consensus['data.agreement'], bin_edges, right=True)
    new_bin_labels = np.arange((min_val - 1) / retirement_lim, 1 + 1 / retirement_lim, 1 / retirement_lim).round(2)

    ########################################## HISTOGRAM OF MAX USER SCORES ###################################################
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica", "font.size": 20})
    plt.figure()
    result_consensus['data.agreement'].plot(kind='hist', bins=bin_edges, logy=False, edgecolor='black', figsize=(12, 12))
    plt.xlabel(f'Consensus User Vote Fraction (Retirement Limit = {retirement_lim})', fontsize=25, labelpad=15)
    plt.ylabel('Number of Events', fontsize=25, labelpad=15)
    plt.xticks(new_bin_labels, labels=new_bin_labels, rotation=45, fontsize=15)
    plt.xlim(0.15, 1.05)
    plt.ylim(0, 1200)
    plt.savefig(os.path.join(outdir, 'plots', 'user_max_score.png'), bbox_inches='tight')
    plt.close()


########################################## CONFUSION MATRIX FOR DNN ########################################################
    
# Create a mapping from integer labels to string labels
    label_mapping = {
    0: 'SKIMMING',
    1: 'CASCADE',
    2: 'THROUGHGOINGTRACK',
    3: 'STARTINGTRACK',
    4: 'STOPPINGTRACK'
}

    categories = ['SKIMMING', 'CASCADE', 'THROUGHGOINGTRACK', 'STARTINGTRACK', 'STOPPINGTRACK']

# Apply the mapping to 'ntn_category'
    result_consensus['ntn_category'] = pd.Categorical(result_consensus['ntn_category'],categories=categories,ordered=True)


# Define the categories explicitly to ensure order consistency

    # Reorder the 'data.most_likely' to match the MC Truth Labels (ntn_category)
    result_consensus['idx_max_score'] = pd.Categorical(
    result_consensus['idx_max_score'], 
    categories=categories, 
    ordered=True
)

    
# Create the confusion matrix and normalize by columns
    confusion_matrix_ml_truth_norm = pd.crosstab(result_consensus['idx_max_score'], result_consensus['ntn_category'], 
                                              rownames=['DNN Max Category'], colnames=['MC Truth Label'], 
                                              margins=False, normalize='columns')


# Round the normalized values to 1 decimal place (percentage) and multiply by 100 to get percentages
    confusion_matrix_ml_truth_norm = (confusion_matrix_ml_truth_norm * 100).round(1)

# Create a matrix for raw counts (for displaying as "count/total")
    confusion_matrix_counts = pd.crosstab(result_consensus['idx_max_score'], result_consensus['ntn_category'], 
                                      rownames=['DNN Max Category'], colnames=['MC Truth Label'], 
                                      margins=False)

# Create a custom annotation combining percentage and raw count

    annot = confusion_matrix_ml_truth_norm.apply(lambda col: col.apply(lambda x: f"{x}\%")) +'%' +  '\n' + confusion_matrix_counts.astype(str) + '/' + confusion_matrix_counts.sum(axis=0).astype(str)

    #print(result_consensus[['ntn_category', 'idx_max_score']].head())

# Plot the heatmap with the custom annotations
    fig, ax = plt.subplots(figsize=(13, 13))
    ax = sns.heatmap(confusion_matrix_ml_truth_norm, annot=annot, fmt='', annot_kws={"size": 15}, 
                 cmap='Blues', xticklabels=types, yticklabels=types, vmin=0.0, vmax=100.0, 
                 cbar_kws={'label': 'Percentage'})
    plt.ylabel('DNN Max Category', fontsize=25, labelpad=15)
    plt.xlabel('MC Truth Label', fontsize=25, labelpad=15)
    plt.savefig(os.path.join(outdir, 'plots', 'dnn_mc_confusion.png'), bbox_inches='tight')
    plt.close()

########################################## CONFUSION MATRIX FOR USER DATA######################################################
# Create the confusion matrix for the user data and normalize by columns using 'data.most_likely'

# Define the categories explicitly to ensure order consistency
    categories = ['SKIMMING', 'CASCADE', 'THROUGHGOINGTRACK', 'STARTINGTRACK', 'STOPPINGTRACK']
    # Reorder the 'data.most_likely' to match the MC Truth Labels (ntn_category)
    result_consensus['data.most_likely'] = pd.Categorical(
    result_consensus['data.most_likely'], 
    categories=categories, 
    ordered=True
)


    confusion_matrix_user_norm = pd.crosstab(result_consensus['data.most_likely'], result_consensus['ntn_category'], 
                                         rownames=['User Agreement'], colnames=['MC Truth Label'], 
                                         margins=False, normalize='columns')

# Round the normalized values to 1 decimal place (percentage) and multiply by 100 to get percentages
    confusion_matrix_user_norm = (confusion_matrix_user_norm * 100).round(1)

# Create a matrix for raw counts using 'data.most_likely'
    confusion_matrix_user_counts = pd.crosstab(result_consensus['data.most_likely'], result_consensus['ntn_category'], 
                                           rownames=['User Agreement'], colnames=['MC Truth Label'], 
                                           margins=False)

#create annotation to display percent and fraction on matrix
    annot_user = confusion_matrix_user_norm.map(lambda x: f"{x}\%") + '\n' + confusion_matrix_user_counts.astype(str) + '/' + confusion_matrix_user_counts.sum(axis=0).astype(str)

# Plot with the custom annotations for the user data
    fig, ax = plt.subplots(figsize=(13, 13))
    ax = sns.heatmap(confusion_matrix_user_norm, annot=annot_user, fmt='', annot_kws={"size": 15}, 
                 cmap='Blues', xticklabels=types, yticklabels=types, vmin=0.0, vmax=100.0, 
                 cbar_kws={'label': 'Percentage'})

# Set labels and save the plot
    plt.ylabel('User Agreement', fontsize=25, labelpad=15)
    plt.xlabel('MC Truth Label', fontsize=25, labelpad=15)
    plt.savefig(os.path.join(outdir, 'plots', 'user_mc_confusion.png'), bbox_inches='tight')
    plt.close()


########################################## CONFUSION MATRIX USER VS DNN######################################################
# Create the confusion matrix for the user and DNN data and normalize by columns using 'data.most_likely'
    confusion_dnn_user_norm = pd.crosstab(result_consensus['idx_max_score'], result_consensus['data.most_likely'], 
                                         rownames=['DNN Max Category'], colnames=['User Agreement'], 
                                         margins=False, normalize='columns')

# Round the normalized values to 1 decimal place (percentage) and multiply by 100 to get percentages
    confusion_dnn_user_norm = (confusion_dnn_user_norm * 100).round(1)

# Create a matrix for raw counts using 'data.most_likely'
    confusion_dnn_user_counts = pd.crosstab(result_consensus['idx_max_score'], result_consensus['data.most_likely'], 
                                           rownames=['DNN Max Category'], colnames=['User Agreement'], 
                                           margins=False)

#create annotation to display percent and fraction on matrix
    annot_dnn_user = confusion_dnn_user_norm.map(lambda x: f"{x}\%") + '\n' + confusion_dnn_user_counts.astype(str) + '/' + confusion_dnn_user_counts.sum(axis=0).astype(str)

# Plot with the custom annotations for the user data
    fig, ax = plt.subplots(figsize=(13, 13))
    ax = sns.heatmap(confusion_dnn_user_norm, annot=annot_dnn_user, fmt='', annot_kws={"size": 15}, 
                 cmap='Blues', xticklabels=types, yticklabels=types, vmin=0.0, vmax=100.0, 
                 cbar_kws={'label': 'Percentage'})

# Set labels and save the plot
    plt.ylabel('DNN Max Category', fontsize=25, labelpad=15)
    plt.xlabel('User Agreement', fontsize=25, labelpad=15)
    plt.savefig(os.path.join(outdir, 'plots', 'user_dnn_confusion.png'), bbox_inches='tight')
    plt.close()

    # Print out the confusion matrix raw counts
    confusion_matrix_raw = pd.crosstab( result_consensus['data.most_likely'], 
    result_consensus['ntn_category'], 
    rownames=['User Agreement'], 
    colnames=['MC Truth Label'], 
    margins=False
)
    print("Raw Confusion Matrix (User):")
    print(confusion_matrix_raw)


def userDNNaccuracy(useraccuracy, DNNaccuracy, qtot, bins=13, log_scale=True):
    """
    Compare user vs DNN accuracy across regions of qtot or log10(qtot).

    Parameters:
    - useraccuracy: np.array of user accuracy values
    - DNNaccuracy: np.array of DNN accuracy values
    - qtot: np.array of qtot values
    - bins: number of bins for qtot
    - log_scale: whether to use log10(qtot) for binning

    Returns:
    - None (displays a plot)


    
    """

    # Filter qtot to only include values under 5000
    filter_mask = qtot < 3200
    qtot = qtot[filter_mask]
    useraccuracy = useraccuracy[filter_mask]
    DNNaccuracy = DNNaccuracy[filter_mask]
    if log_scale:
        q = np.log10(qtot)
        xlabel = 'log10(qtot)'
    else:
        q = qtot
        xlabel = 'qtot'

    # Bin data
    bin_edges = np.linspace(np.min(q), np.max(q), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    user_avg = []
    DNN_avg = []
    event_counts = []

    for i in range(bins):
        in_bin = (q >= bin_edges[i]) & (q < bin_edges[i + 1])
        event_counts.append(np.sum(in_bin))
        if np.any(in_bin):
            user_avg.append(np.mean(useraccuracy[in_bin]))
            DNN_avg.append(np.mean(DNNaccuracy[in_bin]))
        else:
            user_avg.append(np.nan)
            DNN_avg.append(np.nan)


    crossover_index = np.where(np.diff(np.sign(np.array(user_avg) - np.array(DNN_avg))))[0]
    if len(crossover_index) > 0:
        crossover_qtot = bin_centers[crossover_index[0]]
        print(f"Crossover point at {xlabel} = {crossover_qtot}")
    else:
        print("No crossover point found.")


    # Print the number of events in each bin
    for i, count in enumerate(event_counts):
        print(f"Bin {i + 1}: {count} events")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, user_avg, 'o-', label='User Accuracy')
    plt.plot(bin_centers, DNN_avg, 'o-', label='DNN Accuracy')

    for i, (x, count) in enumerate(zip(bin_centers, event_counts)):
        plt.text(x, 0.05, str(count), ha='center', va='bottom', fontsize=9)  
    
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.title('User vs DNN Accuracy by {}'.format(xlabel), 'userconf > .55')
    plt.legend()
    plt.grid(True)
    plt.show()

def accuracybycat1(accuracy, qtot, ntn_category, DNNaccuracy, bin_config=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    df = pd.DataFrame({
        'accuracy': accuracy,
        'qtot': qtot,
        'ntn_category': ntn_category
    })

    df['log_qtot'] = np.log10(df['qtot'])

    categories = ['STARTINGTRACK', 'STOPPINGTRACK', 'THROUGHGOINGTRACK', 'CASCADE', 'SKIMMING']

    category_colors = {
        'STARTINGTRACK': 'purple',
        'STOPPINGTRACK': 'orange',
        'THROUGHGOINGTRACK': 'blue',
        'CASCADE': 'green',
        'SKIMMING': 'red'
    }

    plt.figure(figsize=(10, 6))
    z = stats.norm.ppf(0.975)  # 95% confidence interval

    for cat in categories:
        sub_df = df[df['ntn_category'] == cat]

        # Apply custom qtot range and bin width if provided
        if bin_config and cat in bin_config:
            qmin, qmax = bin_config[cat].get('range', (df['qtot'].min(), df['qtot'].max()))
            print(qmin)
            bin_width = bin_config[cat].get('bin_width', 0.5)
            sub_df = sub_df[(sub_df['qtot'] >= qmin) & (sub_df['qtot'] <= qmax)]
        else:
            qmin, qmax = df['qtot'].min(), df['qtot'].max()
            bin_width = 0.5  # default

        if len(sub_df) == 0:
            continue

        min_log_qtot = np.log10(qmin)
        max_log_qtot = np.log10(qmax)

        bins = np.arange(np.floor(min_log_qtot), np.ceil(max_log_qtot) + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        binned_accuracy = []
        binned_errors = []
        bin_counts = []

        for i in range(len(bins) - 1):
            bin_data = sub_df[(sub_df['log_qtot'] >= bins[i]) & (sub_df['log_qtot'] < bins[i + 1])]
            n = len(bin_data)
            bin_counts.append(n)
            if n > 0:
                p = bin_data['accuracy'].mean()
                stderr = np.sqrt(p * (1 - p) / n)
                error = z * stderr
            else:
                p = np.nan
                error = np.nan
            binned_accuracy.append(p)
            binned_errors.append(error)

        plt.errorbar(
            bin_centers,
            binned_accuracy,
            yerr=binned_errors,
            fmt='o-',
            label=cat,
            color=category_colors.get(cat, 'black'),
            linewidth=1.375,
            markersize=4,
            capsize=2
        )
        
        
        #for x, y, count in zip(bin_centers, binned_accuracy, bin_counts):
        #    if not np.isnan(y):
        #        plt.text(x, y + 0.01, str(count), ha='center', va='bottom', fontsize=8, alpha=0.6)

    plt.xlabel(r'$\log_{10}(Qtot)$')
    plt.ylabel('Average User Accuracy')
    plt.title('Average User Accuracy vs Qtot by Topology')
    plt.legend()
    plt.grid(True)

    # Determine the global log_qtot range for setting ticks
    plt.xticks(np.arange(1.5, 4.75, 0.25))
    plt.xlim(1.75, 4)

    
    plt.tight_layout()
    plt.show()



def accuracybycat(accuracy, qtot, ntn_category, DNNaccuracy, bin_config=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    df = pd.DataFrame({
        'accuracy': accuracy,
        'DNN_accuracy': DNNaccuracy,
        'qtot': qtot,
        'ntn_category': ntn_category
    })

    df['log_qtot'] = np.log10(df['qtot'])

    # Limit to 3 categories
    categories = ['THROUGHGOINGTRACK', 'CASCADE', 'SKIMMING']

    category_colors = {
        'THROUGHGOINGTRACK': 'blue',
        'CASCADE': 'green',
        'SKIMMING': 'red'
    }

    plt.figure(figsize=(10, 6))
    z = stats.norm.ppf(0.975)  # 95% confidence interval

    for cat in categories:
        sub_df = df[df['ntn_category'] == cat]

        if bin_config and cat in bin_config:
            qmin, qmax = bin_config[cat].get('range', (df['qtot'].min(), df['qtot'].max()))
            bin_width = bin_config[cat].get('bin_width', 0.25)
            sub_df = sub_df[(sub_df['qtot'] >= qmin) & (sub_df['qtot'] <= qmax)]
        else:
            qmin, qmax = df['qtot'].min(), df['qtot'].max()
            bin_width = 0.25  # default

        if len(sub_df) == 0:
            continue

        min_log_qtot = np.log10(qmin)
        max_log_qtot = np.log10(qmax)

        bins = np.arange(min_log_qtot, max_log_qtot + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        user_acc, user_err, dnn_acc = [], [], []
        bin_counts = []

        for i in range(len(bins) - 1):
            bin_data = sub_df[(sub_df['log_qtot'] >= bins[i]) & (sub_df['log_qtot'] < bins[i + 1])]
            n = len(bin_data)
            bin_counts.append(n)

            if n > 0:
                p_user = bin_data['accuracy'].mean()
                p_dnn = bin_data['DNN_accuracy'].mean()
                stderr = np.sqrt(p_user * (1 - p_user) / n)
                error = z * stderr
            else:
                p_user = np.nan
                p_dnn = np.nan
                error = np.nan

            user_acc.append(p_user)
            user_err.append(error)
            dnn_acc.append(p_dnn)

        # Plot user accuracy with error bars
        plt.errorbar(
            bin_centers,
            user_acc,
            yerr=user_err,
            fmt='o-',
            label=f'{cat} User Accuracy',
            color=category_colors.get(cat, 'black'),
            linewidth=1.375,
            markersize=4,
            capsize=2
        )

        # Plot DNN accuracy as dashed line
        plt.plot(
            bin_centers,
            dnn_acc,
            linestyle='--',
            color=category_colors.get(cat, 'black'),
            alpha=0.4,
            label=f'{cat} DNN Accuracy'
        )

        # Optionally annotate bin counts
        for x, y, count in zip(bin_centers, user_acc, bin_counts):
            if not np.isnan(y):
                plt.text(x, y + 0.01, str(count), ha='center', va='bottom', fontsize=8, alpha=0.6)

    plt.xlabel(r'$\log_{10}(Q_{tot}\ [\mathrm{GeV}])$')
    plt.ylabel('Average Accuracy')
    plt.title('User vs DNN Accuracy vs $Q_{tot}$ by Topology')
    plt.legend()
    plt.grid(True)

    # Determine the global log_qtot range for setting ticks
    plt.xticks(np.arange(1.5, 4.75, 0.25))
    plt.xlim(1.75, 4)

    plt.tight_layout()
    plt.show()


def accuracybycat_subset(accuracy, qtot, ntn_category, DNNaccuracy, bin_config=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    df = pd.DataFrame({
        'accuracy': accuracy,
        'qtot': qtot,
        'ntn_category': ntn_category,
        'dnn_accuracy': DNNaccuracy
    })

    df['log_qtot'] = np.log10(df['qtot'])

    categories = ['SKIMMING', 'CASCADE', 'THROUGHGOINGTRACK']

    category_colors = {
        'SKIMMING': 'red',
        'CASCADE': 'green',
        'THROUGHGOINGTRACK': 'blue'
    }

    plt.figure(figsize=(10, 6))
    z = stats.norm.ppf(0.975)  # 95% confidence interval

    for cat in categories:
        sub_df = df[df['ntn_category'] == cat]

        if bin_config and cat in bin_config:
            qmin, qmax = bin_config[cat].get('range', (df['qtot'].min(), df['qtot'].max()))
            bin_width = bin_config[cat].get('bin_width', 0.5)
            sub_df = sub_df[(sub_df['qtot'] >= qmin) & (sub_df['qtot'] <= qmax)]
        else:
            qmin, qmax = df['qtot'].min(), df['qtot'].max()
            bin_width = 0.5

        if len(sub_df) == 0:
            continue

        min_log_qtot = np.log10(qmin)
        max_log_qtot = np.log10(qmax)

        bins = np.arange(np.floor(min_log_qtot), np.ceil(max_log_qtot) + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        user_acc = []
        user_err = []

        dnn_acc = []
        dnn_err = []

        for i in range(len(bins) - 1):
            bin_data = sub_df[(sub_df['log_qtot'] >= bins[i]) & (sub_df['log_qtot'] < bins[i + 1])]
            n = len(bin_data)

            if n > 0:
                # User accuracy
                p_user = bin_data['accuracy'].mean()
                stderr_user = np.sqrt(p_user * (1 - p_user) / n)
                err_user = z * stderr_user
                user_acc.append(p_user)
                user_err.append(err_user)

                # DNN accuracy
                p_dnn = bin_data['dnn_accuracy'].mean()
                stderr_dnn = np.sqrt(p_dnn * (1 - p_dnn) / n)
                err_dnn = z * stderr_dnn
                dnn_acc.append(p_dnn)
                dnn_err.append(err_dnn)
            else:
                user_acc.append(np.nan)
                user_err.append(np.nan)
                dnn_acc.append(np.nan)
                dnn_err.append(np.nan)

        # User accuracy plot
        plt.errorbar(
            bin_centers,
            user_acc,
            yerr=user_err,
            fmt='o-',
            label=cat,
            color=category_colors[cat],
            linewidth=1.375,
            markersize=4,
            capsize=2
        )

        # DNN accuracy plot
        plt.errorbar(
            bin_centers,
            dnn_acc,
            yerr=None,
            fmt='o--',
            label=f'{cat} DNN',
            color=category_colors[cat],
            linewidth=1.375,
            markersize=0,
            capsize=2,
            alpha=0.7
        )

    plt.xlabel(r'$\log_{10}(Qtot)$')
    plt.ylabel('Average Accuracy')
    plt.title('Average User vs DNN Accuracy by Topology')
    plt.legend()
    plt.grid(True)

    plt.xticks(np.arange(1.5, 4.75, 0.25))
    plt.xlim(1.75, 4)

    plt.tight_layout()
    plt.show()
