# %% LOAD PACKAGES
import GetOldTweets3 as got
import numpy as np
import csv, re
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.svm import OneClassSVM
from numpy import quantile, where, random

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

namefiles = ['BLM10M', 'BLM26M', 
            'COVID10M','COVID26M',
            'kitten10M','kitten26M',
            'pet10M', 'pet26M']

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
hashtag = ['#BlackLivesMatter', '#BlackLivesMatter',
        '#COVID19','#COVID19',
        '#kitten','#kitten',
        '#pet','#pet']

length_datasets = []

# Preprocessing loop
for i, datas in enumerate(all_data):
    # Select data to filter
    prep_data = pd.DataFrame(all_data[datas])

    # Check data is in may of 2020
    may_check = prep_data[0].str.contains('2020-05') 
    filtered_data[datas] = prep_data[may_check] # save filtered dataset
    removed_data = prep_data[~may_check] # save samples removed in a list

    # Check data contains the # of interest (CASE INSENSITIVE)
    hashtag_check = filtered_data[datas][6].str.contains(hashtag[i], flags = re.IGNORECASE)
    filtered_data [datas] = filtered_data[datas][hashtag_check] # store filtered dataset
    removed_data.append(filtered_data[datas][~hashtag_check]) # add samples to the removed list

    # Check repeated tweets (using URL)
    equal_check = filtered_data[datas][11].duplicated()
    filtered_data[datas] = filtered_data[datas][~equal_check] # save filtered dataset
    removed_data.append (filtered_data[datas][equal_check])# add samples to the removed list
    
    #Save removed samples in dictionary
    all_data_removed [datas] = removed_data 

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
        scores.append([sentence, selected[3][row_names[ind]], selected[4][row_names[ind]], selected[5][row_names[ind]], 
                    vs['neg'], vs['neu'], vs['pos'], vs['compound']])

    # Save all info as a dataframe maintaining the original labels (for traceability) 
    scores_all[datas] = pd.DataFrame(scores, columns = ['Text', 'reply', 'rts', 'fav', 'neg', 'neu', 'pos', 'compound'])
    scores_all[datas].index = list(row_names) # Assign saved labels to new dataframe
    
    # Add a marker to differentiate samples in the control set (as they will be merged)
    if (datas == 'kitten10M') or (datas == 'kitten26M'): # Add a k in the label
        scores_all[datas].index = list ('k'+ scores_all[datas].index.astype(str))

print('Sentiment analysis completed.')

#%% CHECK DATA TO MERGE:
# Check if the datasets are homogeneous enough to be merged
control_keys = ['kitten10M', 'pet10M', 'kitten26M', 'pet26M']
compound_control = []

# Merge scores of all datasets
for val in control_keys:
    compound_control.append(scores_all[val]['compound'])

# Check if controls are equal 
fig, ax = plt.subplots()
plt.boxplot(compound_control, labels = control_keys)
plt.title('Compound scores CONTROL data')
plt.xlabel('Datasets')
plt.ylabel('Score')
plt.show()

#%% MERGE DATA 
scores_all['control10M'] = pd.concat([scores_all['kitten10M'], scores_all['pet10M']])
scores_all['control26M'] = pd.concat([scores_all['kitten26M'], scores_all['pet26M']])

# Delete old keys (now merged in control 10M and control26M)
del scores_all['kitten10M'], scores_all['kitten26M'], scores_all['pet10M'], scores_all['pet26M']

# Plot merged data 
fig, ax= plt.subplots()
plt.boxplot([scores_all['control10M']['compound'], scores_all['control10M']['compound']],
            labels=['control10M','control26M'])
plt.title('Compound scores CONTROL data after merging')
plt.show()

#%% TEST RESAMPLING
# Test if the downsampling performed is representative of the whole population. 

num_resamp= 100

length_datasets =[]
#Find sizes of datasets 
for datas in scores_all:
    length_datasets.append(len(scores_all[datas])) # Store sample size of each dataset 

min_sample = min(length_datasets) # Find smallest dataset

sampled_data = {}
av_repeated = {}
# Repeat resampling multiple times (num_resamp)
for resamp in range (num_resamp):
    save_comp_mean = []
    for data in scores_all:   
        sampled_data[data] =  scores_all[data].sample(min_sample) #random_state=1
        save_comp_mean.append(np.mean(sampled_data[data]['compound']))
    av_repeated[resamp] = save_comp_mean

av_repeated=pd.DataFrame(av_repeated).transpose()
av_repeated.columns = list(sampled_data.keys())
fig, ax= plt.subplots()
ax.axhline(y=0)
av_repeated.boxplot()
plt.xlabel('Dataset')
plt.xticks(rotation=90)
plt.ylabel('Score')
plt.title('Compound scores ' + str(num_resamp) + ' resampling')
plt.show()

#%% DATA SAMPLING: Select a subset of data to work with. The subset must be as big as the smallest dataset. 
# NOTE:A constant seed will be used for reproducibility.
# It also changes the type of the variables reply,rts and fav to numeric

length_datasets =[]
# Find sizes of datasets 
for datas in scores_all:
    length_datasets.append(len(scores_all[datas])) # Store sample size of each dataset 

min_sample = min(length_datasets) # Find smallest dataset
sampled_data = {}

# Select a random subset of samples of size of the dataset with less samples.
# NOTE: The variables 'reply','rts' and 'fav' are also being converted to numeric values. 
for data in scores_all:  
    sampled_data[data] = scores_all[data].sample(min_sample,random_state=1) 
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
    figs = sampled_data[data].boxplot(column= ['neg','neu','pos'])
    plt.title(data + ' all variables boxplot')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.show()

    #save individual variables for plotting
    save_comp.append(sampled_data[data]['compound'])
    save_rts.append(sampled_data[data]['rts'])
    save_fav.append(sampled_data[data]['fav'])
    save_rep.append(sampled_data[data]['reply'])


# Plot boxplot for the compound score of each dataset
fig, ax = plt.subplots()
ax.axhline(y = 0)
plt.boxplot(save_comp ,labels = sampled_data.keys())
plt.title('Compound scores')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.show()

#Plot RTs  
fig, ax = plt.subplots()
ax.axhline(y = 0)
plt.boxplot(save_rts ,labels = sampled_data.keys())
plt.title('Number of RTs')
plt.xlabel('Dataset')
plt.ylabel('Num RTs')
plt.show()

#Plot Favs  
fig, ax = plt.subplots()
ax.axhline(y = 0)
plt.boxplot(save_fav ,labels = sampled_data.keys())
plt.title('Number of favourites')
plt.xlabel('Dataset')
plt.ylabel('Num favs')
plt.show()

#Plot Reply  
fig, ax = plt.subplots()
ax.axhline(y = 0)
plt.boxplot(save_rep ,labels = sampled_data.keys())
plt.title('Number of replies')
plt.xlabel('Dataset')
plt.ylabel('Num of replies')
plt.show()

#%% 3D plot for all samples 
sampled_data_all = pd.DataFrame()
for data in sampled_data:
    sampled_data[data]['data'] = [data] * len(sampled_data[data])
#Concatenate all samples and add a variable reflecting the class
sampled_data_all = pd.concat ([sampled_data['BLM10M'],sampled_data['BLM26M'],sampled_data['COVID10M'],sampled_data['COVID26M'],sampled_data['control10M'],sampled_data['control26M']])
fig = px.scatter_3d(sampled_data_all, x = 'neu', y = 'pos', z = 'neg', 
                    color = 'data', opacity = 0.7, title = 'Sample space all datasets')
fig.show()

# %% One class modelling : check nu
# nu: An upper bound on the fraction of training errors 
# and a lower bound of the fraction of support vectors. 
# Should be in the interval (0, 1]. By default 0.5 will be taken.

# Define SVM parameters 
data_s = 'BLM'
kernel = 'linear' 
nu_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]

num_out = [None] * len(nu_list) 
for n, nus in enumerate(nu_list): #check param nu 
    svm = OneClassSVM(kernel = kernel, nu = nus)

    # Define training and test data 
    sampled_data_numeric= sampled_data[data_s + '10M'][['neg','neu','pos','compound']]
    sampled_data_numeric_test=sampled_data[data_s + '26M'][['neg','neu','pos','compound']]

    # Train algorithm
    BLM_svm= svm.fit(sampled_data_numeric)

    # Test algorithm
    pred = BLM_svm.predict(sampled_data_numeric_test)
    anom_index = where(pred==-1) # class -1 = outlier 
    values = np.array(sampled_data_numeric_test)[anom_index] #identify outlier samples
    df_values= pd.DataFrame(values,columns= ['neg','neu','pos','compound'])

    # Identify which sample IDs have been marked as outliers
    ids= [None] * len(values) 
    for v, val in enumerate(values):
        ids[v]=sampled_data_numeric_test.loc[(sampled_data_numeric_test['neg'] == df_values['neg'][v]) 
        & (sampled_data_numeric_test['neu'] == df_values['neu'][v])
        & (sampled_data_numeric_test['pos'] == df_values['pos'][v])
        & (sampled_data_numeric_test['compound'] == df_values['compound'][v])].index.values[0]
    num_out[n]= len(ids) #count outliers

# Histogram outliers
plt.bar(np.arange(len(nu_list)),num_out)
plt.xticks(np.arange(len(nu_list)), nu_list,rotation=90)
plt.ylabel('Number of outliers')
plt.xlabel('Value of parameter nu')
plt.title(kernel + ': Num outliers depending on nu '+ data_s)
plt.show()

# %% One class modelling right nu: 

# Define SVM parameters 
data_s= 'BLM'
kernel = 'linear'
nu_selected = 0.01
num_out= [] 
svm = OneClassSVM(kernel='linear', nu=nu_selected)
print(svm)
# define training and test data 
sampled_data_numeric= sampled_data[data_s + '10M'][['neg','neu','pos','compound']]
sampled_data_numeric_test=sampled_data[data_s + '26M'][['neg','neu','pos','compound']]

# Train algorithm
BLM_svm= svm.fit(sampled_data_numeric)
# Test algorithm
pred = BLM_svm.predict(sampled_data_numeric_test)
anom_index = where(pred==-1) # class -1 = outlier 
values = np.array(sampled_data_numeric_test)[anom_index] #identify outlier samples
df_values= pd.DataFrame(values,columns= ['neg','neu','pos','compound'])

# identify which sample ID has been marked as outlier
ids= [None] * len(values) 
for v, val in enumerate(values):
    ids[v]=sampled_data_numeric_test.loc[(sampled_data_numeric_test['neg'] == df_values['neg'][v]) 
    & (sampled_data_numeric_test['neu'] == df_values['neu'][v])
    & (sampled_data_numeric_test['pos'] == df_values['pos'][v])
    & (sampled_data_numeric_test['compound'] == df_values['compound'][v])].index.values[0]
num_out= len(ids)

#prepare data for 3D plot 
sampled_data_numeric_test ['out']= [0]*len(sampled_data_numeric_test) #class label
sampled_data_numeric_test.loc[ids,'out']= 1 # outlier label 

fig = px.scatter_3d(sampled_data_numeric_test, x='neu', y='pos', z='neg',color='out',opacity=0.7,
title= data_s + ' one-class ('+ kernel +')')
fig.show() 

extracted_data = sampled_data[data_s + '26M'].loc[ids]
extracted_data = extracted_data.drop(columns=['pos','neg','neu'])

c=1
# Print results: 
print('The number of discrepant samples are: '+ str(num_out))
print(extracted_data)
c=1


# %% One class modelling : RBF check different nu and gamma values 

# RBF: 
# Define SVM parameters 
data_s= 'BLM'
nu_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]
gamma = 'scale' #scale= 1 / (n_features * X.var()), #auto= 1 / n_features

num_out= [None] * len(nu_list) 
for n, nu1 in enumerate(nu_list): #check param nu 
    svm = OneClassSVM(kernel='rbf', nu=nu1,gamma ='scale')

    # define training and test data 
    sampled_data_numeric= sampled_data[data_s + '10M'][['neg','neu','pos','compound']]
    sampled_data_numeric_test=sampled_data[data_s + '26M'][['neg','neu','pos','compound']]

    # Train algorithm
    BLM_svm= svm.fit(sampled_data_numeric)
    # Test algorithm
    pred = BLM_svm.predict(sampled_data_numeric_test)
    anom_index = where(pred==-1) # class -1 = outlier 
    values = np.array(sampled_data_numeric_test)[anom_index] #identify outlier samples
    df_values= pd.DataFrame(values,columns= ['neg','neu','pos','compound'])

    # identify which sample ID has been marked as outlier
    ids= [None] * len(values) 
    for v, val in enumerate(values):
        ids[v]=sampled_data_numeric_test.loc[(sampled_data_numeric_test['neg'] == df_values['neg'][v]) 
        & (sampled_data_numeric_test['neu'] == df_values['neu'][v])
        & (sampled_data_numeric_test['pos'] == df_values['pos'][v])
        & (sampled_data_numeric_test['compound'] == df_values['compound'][v])].index.values[0]
    num_out[n]= len(ids)

#Histogram 
his= plt.bar(np.arange(len(nu_list)),num_out)
plt.xticks(np.arange(len(nu_list)), nu_list)
plt.ylabel('Number of outliers')
plt.xlabel('Value of parameter nu')
plt.title('Num outliers depending on nu: '+ data_s)
plt.show()


# %% One class modelling right nu: intra sample 

# Define SVM parameters  try with RBF
data_s = 'BLM'
kernel = 'rbf'
gamma = 'scale' #auto or scale 
nu_selected = 0.01

num_out= [] 
svm = OneClassSVM(kernel=kernel, nu=nu_selected, gamma=gamma)
print(svm)
# define training and test data 
sampled_data_numeric= sampled_data[data_s + '10M'][['neg','neu','pos','compound']]
sampled_data_numeric_test=sampled_data[data_s + '26M'][['neg','neu','pos','compound']]

# Train algorithm
BLM_svm= svm.fit(sampled_data_numeric)
# Test algorithm
pred = BLM_svm.predict(sampled_data_numeric_test)
anom_index = where(pred==-1) # class -1 = outlier 
values = np.array(sampled_data_numeric_test)[anom_index] #identify outlier samples
df_values= pd.DataFrame(values,columns= ['neg','neu','pos','compound'])

# identify which sample ID has been marked as outlier
ids= [None] * len(values) 
for v, val in enumerate(values):
    ids[v]=sampled_data_numeric_test.loc[(sampled_data_numeric_test['neg'] == df_values['neg'][v]) 
    & (sampled_data_numeric_test['neu'] == df_values['neu'][v])
    & (sampled_data_numeric_test['pos'] == df_values['pos'][v])
    & (sampled_data_numeric_test['compound'] == df_values['compound'][v])].index.values[0]
num_out= len(ids)

#prepare data for 3D plot 
sampled_data_numeric_test ['out']= [0]*len(sampled_data_numeric_test) #class label
sampled_data_numeric_test.loc[ids,'out']= 1 # outlier label 

fig = px.scatter_3d(sampled_data_numeric_test, x='neu', y='pos', z='neg',color='out',opacity=0.7,
title= data_s + ' one-class ('+ kernel + ': ' + gamma + ')' + ' nu: ' + str(nu_selected))
fig.show() 

extracted_data = sampled_data[data_s + '26M'].loc[ids]
extracted_data = extracted_data.drop(columns=['pos','neg','neu'])

# Print results: 
print('The number of discrepant samples are: '+ str(num_out))
print(extracted_data)
c=1
# %% FUNCTION ONE-CLASS MODEL 

    
