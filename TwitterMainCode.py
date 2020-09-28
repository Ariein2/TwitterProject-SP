#%% LOAD PACKAGES
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

#%% RETRIEVE TWEETS: 
""" 
### RETRIEVE TWEETS (script): 

# searchTerm = input("Enter Keywords (Space Seperated): ")
# start=input("Enter Start date in format YYYY-MM-DD: ")
# end=input("Enter End date in format YYYY-MM-DD: ")


searchTerm = "#COVID19"
start = "2020-05-10"
end = "2020-05-11"

tweetCriteria = got.manager.TweetCriteria().setQuerySearch(searchTerm)\
                                            .setSince(start)\
                                            .setUntil(end)\
                                            .setMaxTweets(50)
                                            
                                        

tweet = got.manager.TweetManager.getTweets(tweetCriteria)
 """


### RETREIVE TWEETS (Run in terminal)

#10-15th May 2020
# GetOldTweets3 --querysearch "#kitten" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "kitten10M.csv"
# GetOldTweets3 --querysearch "#pet" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "pet10M.csv"
# GetOldTweets3 --querysearch "#COVID19" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "COVID10M.csv"
# GetOldTweets3 --querysearch "#BlackLivesMatter" --since "2020-05-10" --until "2020-05-15" --toptweets --maxtweets 10000 --lang en --output "BLM10M.csv"

#26- 31 May 2020
# GetOldTweets3 --querysearch "#BlackLivesMatter" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "BLM26M.csv"
# GetOldTweets3 --querysearch "#kitten" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "kitten26M.csv"
# GetOldTweets3 --querysearch "#pet" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "pet26M.csv" 
# GetOldTweets3 --querysearch "#COVID19" --since "2020-05-26" --until "2020-05-31" --toptweets --maxtweets 10000 --lang en --output "COVID26M.csv" 

#%% LOAD DATA FROM CSV FILES

# Replace the elements in namefiles with the name of the CSV files to load.
# (All files must be located in the same folder as the python script) 

namefiles = ['BLM10M','BLM26M','COVID10M','COVID26M','kitten10M','kitten26M','pet10M','pet26M']
all_data={}

for dataset in namefiles: #load all csvs

    with open(dataset + '.csv', newline='') as f:
        reader = csv.reader(f)
        list_reader =list(reader)
        all_data[dataset]= list_reader #all files are stored as a list in the all_data dictionary. 


#%% PREPROCESSING : 
# Remove errors in samples loaded. It checks that the date of the tweet is 
# May 2020, the presence of the # searched in the tweet text and  checks for repeated tweets. 
# It is necessary to introduce the hashtag associated to each list in the all_data dictionary. 
# NOTE: This step will always remove at least 1 sample as it corresponds to the first row 
# from the list of each dataset, which contains the labels of the columns. 

all_data_removed= {}
filtered_data = {}
hashtag= ['#BlackLivesMatter','#BlackLivesMatter','#COVID19','#COVID19','#kitten','#kitten','#pet','#pet'] # List of hashtags used in each dataset
length_datasets =[]

for i, datas in enumerate(all_data):
    # Select data to filter
    prep_data= pd.DataFrame(all_data[datas])

    # Check data is in may of 2020
    may_check= prep_data[0].str.contains('2020-05') #check data in May 2020 
    filtered_data[datas] = prep_data[may_check] # store filtered dataset
    removed_data = prep_data[~may_check] # store samples removed

    # Check data contains the # of interest
    hashtag_check = filtered_data[datas][6].str.contains(hashtag[i], flags = re.IGNORECASE) #check that data contains # CASE INSENSITIVE
    filtered_data [datas] = filtered_data[datas][hashtag_check]
    removed_data.append(filtered_data[datas][~hashtag_check])

    #Check repeated tweets (using URL): 
    equal_check = filtered_data[datas][11].duplicated()
    filtered_data[datas] = filtered_data[datas][~equal_check]
    removed_data.append (filtered_data[datas][equal_check])
    all_data_removed [datas] = removed_data 

    #Display info on samples removed 
    count_removed = len(removed_data) 
    print('Original size of data ' + datas +': ' + str(len(prep_data)))
    print('New size of data ' + datas + ': ' + str(len(filtered_data[datas]))) 



#%% SENTIMENT ANALYSIS: Identify how positive, negative or neutral is a tweet. 
analyzer = SentimentIntensityAnalyzer()

scores_all = {}
for i, datas in enumerate (filtered_data): #For each dataset
    selected = filtered_data[datas]
    scores = []
    row_names = selected.index.values
    for ind, sentence in enumerate(selected[6]): #Check each sentence 
        vs = analyzer.polarity_scores(sentence)
        scores.append([sentence, selected[3][row_names[ind]], selected[4][row_names[ind]], selected[5][row_names[ind]], vs['neg'], vs['neu'], vs['pos'], vs['compound']])

    #Save scores, text and maintain label
    scores_all [datas] = pd.DataFrame(scores, columns= ['Text','reply','rts','fav','neg','neu','pos','compound'])
    scores_all[datas].index=list(row_names)
    if (datas == 'kitten10M') or (datas == 'kitten26M'): #change label of kittens
        scores_all[datas].index= list('k'+ scores_all[datas].index.astype(str))


print('Sentiment analysis completed.')

#%% CHECK DATA TO MERGE
control_keys = ['kitten10M','pet10M','kitten26M','pet26M']
compound_control=[]
for val in control_keys:
    compound_control.append(scores_all[val]['compound'])

# Check if controls are equal. 
fig, ax= plt.subplots()
plt.boxplot(compound_control,labels=control_keys)
plt.title('Compound scores CONTROL data')
plt.show()

#%% STATISTICAL TEST?? 

#%% MERGE DATA 
scores_all['control10M'] =  pd.concat([scores_all['kitten10M'], scores_all['pet10M']])
scores_all['control26M'] =  pd.concat([scores_all['kitten26M'], scores_all['pet26M']])

# Delete old keys (now merged in control 10M and control26M)
del scores_all['kitten10M'], scores_all['kitten26M'], scores_all['pet10M'], scores_all['pet26M']

# Plot 
fig, ax= plt.subplots()
plt.boxplot([scores_all['control10M']['compound'], scores_all['control10M']['compound']],labels=['control10M','control26M'])
plt.title('Compound scores CONTROL data after merging')
plt.show()


#%% DATA SELECTION: Select a subset of data to work with. The subset must be as big as the smallest dataset. 
# NOTE: REMOVE RANDOM STATE BEFORE RUNNING!!!!! Sets a constant seed. 
# It also changes the type of the variables reply,rts and fav to numeric
length_datasets =[]
for datas in scores_all:
    length_datasets.append(len(scores_all[datas])) # Store sample size of each dataset 

min_sample = min(length_datasets)
sampled_data={}

for data in scores_all:  
    sampled_data[data] =  scores_all[data].sample(min_sample,random_state=1) #scores_all[data].sample(min_sample)
    sampled_data[data]['reply'] = pd.to_numeric(sampled_data[data]['reply'])
    sampled_data[data]['rts'] = pd.to_numeric(sampled_data[data]['rts'])
    sampled_data[data]['fav'] = pd.to_numeric(sampled_data[data]['fav'])
print('Sampling completed.')

#%% DATA INSPECTION

save_comp = []
save_rts = []
#Plot each sampled datset:neg, neu, pos
for data in sampled_data:   
    plt.figure()
    figs = sampled_data[data].boxplot(column= ['neg','neu','pos'])
    plt.title(data + ' all variables boxplot')
    plt.show()
    #save individual variables for plotting
    save_comp.append(sampled_data[data]['compound'])
    save_rts.append(sampled_data[data]['rts'])

#Plot compound 
fig, ax= plt.subplots()
ax.axhline(y=0)
plt.boxplot(save_comp ,labels= sampled_data.keys())
plt.title('Compound scores')
plt.show()

#Plot RTs  
fig, ax= plt.subplots()
ax.axhline(y=0)
plt.boxplot(save_rts ,labels= sampled_data.keys())
plt.title('RT values')
plt.show()


#%% TEST RESAMPLING 
num_resamp= 100

sampled_data={}
av_repeated = {}
for resamp in range (num_resamp):
    save_comp_mean=[]
    for data in scores_all:   
        sampled_data[data] =  scores_all[data].sample(min_sample) #random_state=1
        sampled_data[data]['reply'] = pd.to_numeric(sampled_data[data]['reply'])
        sampled_data[data]['rts'] = pd.to_numeric(sampled_data[data]['rts'])
        sampled_data[data]['fav'] = pd.to_numeric(sampled_data[data]['fav'])
        save_comp_mean.append(np.mean(sampled_data[data]['compound']))
    av_repeated[resamp]= save_comp_mean

av_repeated2=pd.DataFrame(av_repeated).transpose()
av_repeated2.columns = list(sampled_data.keys())
fig, ax= plt.subplots()
ax.axhline(y=0)
av_repeated2.boxplot()
plt.title('Compound scores ' + str(num_resamp) + ' resampling')
plt.show()

#%% 3D plot 

sampled_data_all = pd.DataFrame()
for data in sampled_data:
    sampled_data[data]['data']= [data] * len(sampled_data[data])
sampled_data_all = pd.concat ([sampled_data['BLM10M'],sampled_data['BLM26M'],sampled_data['COVID10M'],sampled_data['COVID26M'],sampled_data['control10M'],sampled_data['control26M']])

fig = px.scatter_3d(sampled_data_all, x='neu', y='pos', z='neg',color= 'data',opacity=0.7)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig['layout']['xaxis']['autorange'] = "reversed"
fig.show()

a = 1


# %% clustering, heatmaps, PCA of compound also? 
# One class modelling, 
# SVM, Model one class and to see how others project in that space. (linear kernel) 
#pca for one class modeling -> Orthogonal distance and scores... 
#pca for 1 class... get matrix distances and then project other data on this space. 

# %% One class modelling 

# Define SVM parameters 
svm = OneClassSVM(kernel='linear', nu=0.03)
print(svm)
# define training and test data 
sampled_data_numeric= sampled_data['BLM10M'][['neg','neu','pos','compound']]
sampled_data_numeric_test=sampled_data['BLM26M'][['neg','neu','pos','compound']]

# Train algorithm
BLM_svm= svm.fit(sampled_data_numeric)
# Test algorithm
pred = svm.predict(sampled_data_numeric_test)
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

#prepare data for 3D plot 
sampled_data_numeric_test ['out']= [0]*len(sampled_data_numeric_test) #class label
sampled_data_numeric_test.loc[ids,'out']= 1 # outlier label 

fig = px.scatter_3d(sampled_data_numeric_test, x='neu', y='pos', z='neg',color='out',opacity=0.7)
fig.show()
c=1
# %%
