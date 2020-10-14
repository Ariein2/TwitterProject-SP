# %% LOAD PACKAGES
import GetOldTweets3 as got
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.svm import OneClassSVM

# %% RETREIVE TWEETS (Run in terminal)

# 10-15th May 2020
#  GetOldTweets3 --querysearch "#kitten" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "kitten10M.csv"
#  GetOldTweets3 --querysearch "#pet" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "pet10M.csv"
#  GetOldTweets3 --querysearch "#COVID19" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "COVID10M.csv"
#  GetOldTweets3 --querysearch "#BlackLivesMatter" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "BLM10M.csv"

# 26- 31 May 2020
#  GetOldTweets3 --querysearch "#BlackLivesMatter" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "BLM26M.csv"
#  GetOldTweets3 --querysearch "#kitten" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "kitten26M.csv"
#  GetOldTweets3 --querysearch "#pet" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "pet26M.csv" 
#  GetOldTweets3 --querysearch "#COVID19" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "COVID26M.csv" 

# %% LOAD DATA FROM CSV FILES

# Replace the elements in namefiles with the name of the CSV files to load.
# (All files must be located in the same folder as the python script) 

namefiles = [
            'BLM10M', 'BLM26M', 
            'COVID10M', 'COVID26M', 
            'kitten10M', 'kitten26M', 
            'pet10M', 'pet26M'
            ]

# Load all csv files
all_data = {}

for dataset in namefiles:
    with open(dataset + '.csv', newline='') as f:
        reader = csv.reader(f)
        list_reader = list(reader)
        all_data[dataset] = list_reader

# Each csv file is stored as a list in the all_data dictionary. 


# %% PREPROCESSING : 
# Remove errors in samples loaded. It checks that the date of the tweet is 
# May 2020, the presence of the # searched in the tweet text and  checks for repeated tweets. 
# It is necessary to introduce the hashtag associated to each list in the all_data dictionary. 
# NOTE: This step will always remove at least 1 sample as it corresponds to the first row 
# from the list of each dataset, which contains the labels of the columns. 

all_data_removed = {}
filtered_data = {}
# List of the hashtags used in each dataset
hashtag = [
           '#BlackLivesMatter', '#BlackLivesMatter',
           '#COVID19', '#COVID19', 
           '#kitten', '#kitten', 
           '#pet', '#pet'
          ]

length_datasets = []

# Preprocessing loop
for i, datas in enumerate(all_data):
    # Select data to filter
    prep_data = pd.DataFrame(all_data[datas])

    # Check data is in may of 2020
    may_check = prep_data[0].str.contains('2020-05')
    filtered_data[datas] = prep_data[may_check]
    removed_data = prep_data[~may_check]

    # Check data contains the # of interest (CASE INSENSITIVE)
    hashtag_check = filtered_data[datas][6].str.contains(hashtag[i], flags=re.IGNORECASE)
    filtered_data[datas] = filtered_data[datas][hashtag_check]
    removed_data.append(filtered_data[datas][~hashtag_check])

    # Check repeated tweets (using URL)
    equal_check = filtered_data[datas][11].duplicated()
    filtered_data[datas] = filtered_data[datas][~equal_check]
    removed_data.append (filtered_data[datas][equal_check])
    
    # Save removed samples in dictionary
    all_data_removed[datas] = removed_data

    # Display info on number of samples removed
    count_removed = len(removed_data)
    print('Original size of data ' + datas + ': ' + str(len(prep_data)))
    print('New size of data ' + datas + ': ' + str(len(filtered_data[datas])))



#%% SENTIMENT ANALYSIS: Identify how positive, negative or neutral is a tweet. 

analyzer = SentimentIntensityAnalyzer()
# Perform sentiment analysis for each dataset
scores_all = {}
for i, datas in enumerate (filtered_data): # For each dataset
    selected = filtered_data[datas]
    scores = []
    row_names = selected.index.values # Save original sample ids
    
    # Check sentiment of each sentence
    for ind, sentence in enumerate(selected[6]): 
        vs = analyzer.polarity_scores(sentence) # Obtain sentiment scores
        # Add scores to the previous data
        scores.append([sentence, selected[3][row_names[ind]], 
                      selected[4][row_names[ind]], selected[5][row_names[ind]], 
                      vs['neg'], vs['neu'], vs['pos'], vs['compound']])

    # Save all info as a dataframe maintaining the original labels (for traceability) 
    scores_all[datas] = pd.DataFrame(scores, columns=['Text', 'reply', 'rts', 
                                                      'fav', 'neg', 'neu', 
                                                      'pos', 'compound'])

    scores_all[datas].index = list(row_names) # Assign saved labels to new dataframe
    
    # Add a marker to differentiate samples in the control set (as they will be merged)
    if (datas == 'kitten10M') or (datas == 'kitten26M'): 
        scores_all[datas].index = list ('k'+ scores_all[datas].index.astype(str))

print('Sentiment analysis completed.')

#%% PARALEL SENTIMENT ANALYSIS: Identify how positive, negative or neutral is a tweet. 
import multiprocessing as mp
pool = mp.Pool (mp.cpu_count())
analyzer = SentimentIntensityAnalyzer()
scores_all = {}

def func_analysis (ind, sentence, analyzer, selected, row_names):
    # Check sentiment of each sentence
    vs = analyzer.polarity_scores(sentence) # Obtain sentiment scores
    # Add scores to the previous data
    scores.append([sentence, selected[3][row_names[ind]], 
                   selected[4][row_names[ind]], selected[5][row_names[ind]], 
                   vs['neg'], vs['neu'], vs['pos'], vs['compound']])
    return scores

for i, datas in enumerate (filtered_data): # For each dataset
    selected = filtered_data[datas]
    scores = []
    row_names = selected.index.values # Save original sample ids
    
    # Check sentiment of each sentence
    scores = [pool.apply(func_analysis, args=(ind, sentence, analyzer,selected,row_names)) for ind, sentence in enumerate(selected[6])]
    pool.close()
    # Save all info as a dataframe maintaining the original labels (for traceability) 
    scores_all[datas] = pd.DataFrame(scores, columns=['Text', 'reply', 'rts', 
                                                      'fav', 'neg', 'neu', 
                                                      'pos', 'compound'])

    scores_all[datas].index = list(row_names) # Assign saved labels to new dataframe
    
    # Add a marker to differentiate samples in the control set (as they will be merged)
    if (datas == 'kitten10M') or (datas == 'kitten26M'): 
        scores_all[datas].index = list ('k'+ scores_all[datas].index.astype(str))

print(scores_all)
print('Sentiment analysis is completed.')
# %% CHECK DATA TO MERGE:
# Check if the datasets are homogeneous enough to be merged

def variable_boxplot(variable, title, label):
    _, ax = plt.subplots()
    ax.axhline(y = 0,linewidth=1.5,color='grey')
    plt.boxplot(variable, labels=label,
                patch_artist=True, widths= 0.6,
                medianprops=dict(color='darkorange',linewidth=1.5),
                boxprops=dict(facecolor='gainsboro', color='k'))
    
    plt.title(title)
    ax.yaxis.grid(True)
    plt.xticks(rotation=90)
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.show()

control_keys = ['kitten10M', 'pet10M', 'kitten26M', 'pet26M']
compound_control = []

# Merge scores of all datasets
for val in control_keys:
    compound_control.append(scores_all[val]['compound'])

# Check if controls are equal 
title= 'Compound scores CONTROL data'

variable_boxplot(compound_control, title, control_keys)

# %% MERGE DATA 

scores_all['control10M'] = pd.concat([scores_all['kitten10M'], scores_all['pet10M']])
scores_all['control26M'] = pd.concat([scores_all['kitten26M'], scores_all['pet26M']])

# Delete old keys (now merged in control 10M and control26M)
del scores_all['kitten10M'], scores_all['kitten26M']
del scores_all['pet10M'], scores_all['pet26M']

# Plot merged data 
label=['control10M', 'control26M']
title= 'Compound scores CONTROL data after merging'
variable_boxplot([scores_all['control10M']['compound'], 
                  scores_all['control26M']['compound']],
                  title,label) 

#%% TEST RESAMPLING
# Test if the downsampling performed is representative of the whole population. 

num_resamp = 100

length_datasets = []
#Find sizes of datasets 
for datas in scores_all:
    length_datasets.append(len(scores_all[datas]))  

# Find smallest dataset
min_sample = min(length_datasets) 

sampled_data = {}
av_repeated = {}

# Repeat resampling multiple times (num_resamp)
for resamp in range (num_resamp):
    save_comp_mean = []
    for data in scores_all:   
        sampled_data[data] =  scores_all[data].sample(min_sample) 
        save_comp_mean.append(np.mean(sampled_data[data]['compound']))
    av_repeated[resamp] = save_comp_mean

av_repeated = pd.DataFrame(av_repeated).transpose() #necessary for boxplot
av_repeated.columns = list(sampled_data.keys())

fig, ax = plt.subplots()
ax.axhline(y = 0, linewidth=1.5, color='grey')
av_repeated.boxplot(patch_artist=True, widths= 0.6,
                medianprops=dict(color='darkorange',linewidth=1.5),
                boxprops=dict(facecolor='gainsboro', color='k'))
plt.xlabel('Dataset')
plt.xticks(rotation=90)
plt.ylabel('Score')
plt.title('Compound scores ' + str(num_resamp) + ' resampling')
plt.show()

#%% DATA SAMPLING: Select a subset of data to work with. The subset must be as big as the smallest dataset. 
# NOTE:A constant seed will be used for reproducibility.
# It also changes the type of the variables reply,rts and fav to numeric

length_datasets = []
# Find sizes of datasets 
for datas in scores_all:
    length_datasets.append(len(scores_all[datas])) 

# Find smallest dataset
min_sample = min(length_datasets) 
sampled_data = {}

# Select a random subset of samples of size of the dataset with less samples.
# NOTE: The variables 'reply','rts' and 'fav' are also being converted to numeric values. 
for data in scores_all:  
    sampled_data[data] = scores_all[data].sample(min_sample, random_state=1) 
    sampled_data[data]['reply'] = pd.to_numeric(sampled_data[data]['reply'])
    sampled_data[data]['rts'] = pd.to_numeric(sampled_data[data]['rts'])
    sampled_data[data]['fav'] = pd.to_numeric(sampled_data[data]['fav'])
print('Sampling completed.')

#%% DATA INSPECTION

save_comp = []
save_rts = []
save_fav = []
save_rep = []

# Plot boxplots for each dataset (variables: neg, neu, pos)
for data in sampled_data:   
    plt.figure()
    sampled_data[data].boxplot(column=['neg', 'neu', 'pos'], 
                               widths= 0.6,
                               patch_artist=True, 
                               medianprops=dict(color='darkorange',linewidth=1.5),
                               whiskerprops= dict(color='k'),
                               boxprops=dict(facecolor='gainsboro', color='k'))
    plt.title(data + ' all variables boxplot')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.show()

    # Save individual variables for plotting
    save_comp.append(sampled_data[data]['compound'])
    save_rts.append(sampled_data[data]['rts'])
    save_fav.append(sampled_data[data]['fav'])
    save_rep.append(sampled_data[data]['reply'])

# Plot boxplot for the compound score of each dataset
title= 'Compound scores'
label= sampled_data.keys()
variable_boxplot(save_comp, title, label)

# Plot RTs  
title= 'Number of Retweets'
variable_boxplot(save_rts, title, label)

# Plot Favs  
title= 'Number of favourites'
variable_boxplot(save_fav, title, label)

# Plot Reply  
title= 'Number of replies'
variable_boxplot(save_rep, title, label)


# %% 3D plot for all samples 
all_sampled = pd.DataFrame()
for data in sampled_data:
    sampled_data[data]['Dataset'] = [data] * len(sampled_data[data])

# Concatenate all samples and add a variable reflecting the class
all_sampled = pd.concat ([sampled_data['BLM10M'], sampled_data['BLM26M'], 
                          sampled_data['COVID10M'], sampled_data['COVID26M'],
                          sampled_data['control10M'], sampled_data['control26M']])

# 3D scatter plot all samles
fig = px.scatter_3d(all_sampled, x='neu', y='pos', 
                    z='neg', color='Dataset', 
                    opacity=0.7, title='Sample space (all datasets)')
fig.show()

# %% FUNCTION DEFINITION: ONE CLASS

def one_class_parameters(data_s, kernel, nu_list, gamma='scale'):

    # Define SVM parameters 
    num_out = [None] * len(nu_list) 

    for n, nus in enumerate(nu_list): #check param nu 
        
        if (kernel == 'rbf'):
            svm = OneClassSVM(kernel=kernel, nu=nus, gamma=gamma)
        else:
            svm = OneClassSVM(kernel=kernel, nu=nus)

        # Define training and test data 
        train_data = sampled_data[data_s + '10M'][['neg', 'neu', 'pos', 'compound']]
        test_data = sampled_data[data_s + '26M'][['neg', 'neu', 'pos', 'compound']]

        # Train algorithm
        BLM_svm = svm.fit(train_data)

        # Test algorithm
        pred = BLM_svm.predict(test_data)
        anom_index = np.where(pred == -1) # Class -1 = outlier 
        values = np.array(test_data)[anom_index] # Identify outlying samples
        df_values = pd.DataFrame(values, columns=['neg', 'neu', 'pos', 'compound'])

        # Identify which sample IDs have been marked as outliers
        ids = [None] * len(values) 
        for v, _ in enumerate(values):
            ids[v] = test_data.loc[(test_data['neg'] == df_values['neg'][v]) 
                                    & (test_data['neu'] == df_values['neu'][v])
                                    & (test_data['pos'] == df_values['pos'][v])
                                    & (test_data['compound'] == df_values['compound'][v])].index.values[0]
        num_out[n] = len(ids) # Count outliers

    # Histogram outliers
    plt.bar(np.arange(len(nu_list)), num_out)
    plt.xticks(np.arange(len(nu_list)), nu_list, rotation=90)
    plt.ylabel('Number of outliers')
    plt.xlabel('Value of parameter nu')
    plt.title(kernel + ': Num outliers depending on nu ' + data_s)
    plt.show()

def one_class (data_s, kernel, nu_selected, gamma='scale'):
    num_out = [] 
    if (kernel == 'rbf'):
        svm = OneClassSVM(kernel=kernel, nu=nu_selected, gamma=gamma)
    else:
        svm = OneClassSVM(kernel=kernel, nu=nu_selected)
   
    print(svm)
    # define training and test data 
    train_data = sampled_data[data_s + '10M'][['neg', 'neu', 'pos', 'compound']]
    test_data = sampled_data[data_s + '26M'][['neg', 'neu', 'pos', 'compound']]

    # Train algorithm
    BLM_svm = svm.fit(train_data)
    # Test algorithm
    pred = BLM_svm.predict(test_data)
    anom_index = np.where(pred == -1) # class -1 = outlier 
    values = np.array(test_data)[anom_index] #identify outlier samples
    df_values = pd.DataFrame(values, columns=['neg', 'neu', 'pos', 'compound'])

    # identify which sample ID has been marked as outlier
    ids = [None] * len(values) 
    for v, _ in enumerate(values):
        ids[v] = test_data.loc[(test_data['neg'] == df_values['neg'][v]) 
                                & (test_data['neu'] == df_values['neu'][v])
                                & (test_data['pos'] == df_values['pos'][v])
                                & (test_data['compound'] == df_values['compound'][v])].index.values[0]
    num_out= len(ids)

    #prepare data for 3D plot 
    test_data ['out'] = [0]*len(test_data) #class label
    test_data.loc[ids, 'out'] = 1 # outlier label 

    fig = px.scatter_3d(test_data, x='neu', y='pos', z='neg',
                        color='out', opacity=0.7, 
                     title=data_s + ' one-class ('
                            + kernel + ': ' 
                            + gamma + ')' 
                            + ' nu: ' + str(nu_selected))
    fig.show() 

    extracted_data = sampled_data[data_s + '26M'].loc[ids]
    # Print results: 
    print('The number of discrepant samples are: ' + str(num_out))
    print(extracted_data[['Text','compound']])

# Display full-size dataframe 
pd.set_option('display.max_colwidth', -1) 

# %% One class BLM 

# Linear kernel one_class: find optimal nu: 
data_s = 'BLM' # BLM, COVID, CONTROL
kernel = 'linear' # linear or rbf
nu_list = [
            0.1, 0.09, 0.08, 0.07, 0.06, 
            0.05, 0.03, 0.02, 0.01, 0.005, 
            0.003, 0.001, 0.0005
          ]

one_class_parameters(data_s, kernel, nu_list)

# Linear kernel one_class tested with optimal nu

nu_selected = 0.01
one_class(data_s, kernel, nu_selected)

#RBF kernel one_class: find optimal nu: 
data_s = 'BLM'
kernel = 'rbf' 
gamma = 'scale' #auto or rbf
one_class_parameters(data_s, kernel, nu_list, gamma=gamma)

nu_selected = 0.04
one_class(data_s, kernel, nu_selected, gamma=gamma)

# %% One-class COVID
data_s = 'COVID' # BLM, COVID, CONTROL
kernel = 'linear' # linear or rbf
one_class_parameters(data_s, kernel, nu_list)

# Linear kernel one_class tested with optimal nu

nu_selected = 0.01
one_class(data_s, kernel, nu_selected)

#RBF kernel one_class: find optimal nu: 
data_s = 'COVID'
kernel = 'rbf' 
gamma = 'scale' #auto or rbf
one_class_parameters(data_s, kernel, nu_list, gamma=gamma)

nu_selected = 0.01
one_class(data_s, kernel, nu_selected, gamma=gamma)
# %% One-class control
data_s = 'control' # BLM, COVID, control
kernel = 'linear' # linear or rbf
one_class_parameters(data_s, kernel, nu_list)

# Linear kernel one_class tested with optimal nu

nu_selected = 0.01
one_class(data_s, kernel, nu_selected)

#RBF kernel one_class: find optimal nu: 
data_s = 'control'
kernel = 'rbf' 
gamma = 'scale' #auto or rbf
one_class_parameters(data_s, kernel, nu_list, gamma=gamma)

nu_selected = 0.01
one_class(data_s, kernel, nu_selected, gamma=gamma)

#%%