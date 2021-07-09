#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on _
@author: Frix
"""

## ?? conda upgrade python==3.7
### !pip install sweetviz keras matplotlib pandas numpy scikit-learn tensorflow biopython tqdm seaborn imbalanced-learn xgboost

#import sweetviz as sv
import os
import pickle
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import sklearn
import tensorflow as tf
from tensorflow import keras

#for importing utility class files from coding_templates dir. so that 
#we can use the same ones across all different projects:
#https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
try:
    sys.path.insert(1, r'C:\Users\Frixos\OneDrive - University of Southampton\coding_templates')
except FileNotFoundError:
    print('Template code files not found. Assuming we are on IRIDIS and moving on...')

from Frix_utils import init_run,prototype_data,load_trigram_vecs,relabel_data, store_res \
    ,filter_out_rare_AAs,plot_hist_ranges,rm_sh_prots,contains_amb_AAs,calc_freq_plot_hist
#plot_scatter_chart, save_df_in_fasta, from_fasta, rm_sh_prots, filter_out_rare_AAs, plot_line_chart, build_hist_gram_counts, comp_sse
from utils_data import data_cleaning, data_split, resampl #,vis_model, t_sne,q_eval_classif_model



#%%
'''#FOR ALL OPERATIONS:'''
rand_state=10

os.chdir(r'C:\Users\Frixos\OneDrive - University of Southampton\PycharmProjects\covid_git')
exp_id=1
exp_path = str(exp_id)
curr_path = os.path.join(exp_path,'EDA')
if os.path.exists(curr_path)==False:
    os.mkdir(curr_path)

user='Frixos'

#%%
'''setting up for EVALUATIons,loading data '''

n_ngrams=8000

#TODO: put in fn for loading BPAD200 from csv or fasta? + handle case that we are on IRIDIS
#TODO: normally datasets should be found in dir. of current script
f_path = r'C:\Users\\'+str(user)+'\\OneDrive - University of Southampton\PycharmProjects\Classifier-1\BPAD200.csv'
data =pd.read_csv(f_path) #,index_col=[0])
if 'Accession' in data.columns.tolist():
    data.rename({'Accession':'Protein'},axis=1,inplace=True)
assert set(['Protein', 'Sequence', 'Label']).issubset(data.columns.tolist())

data = data[['Protein','Sequence','Label']]
n_classes = 2
    
#relabel neg. class to 0
class_labels = ['Non_antigens', 'Antigens']
assert n_classes == len(class_labels)
#print('Converting datasets to ' + str(n_classes) + '-classes: enzyme and non-enzyme(class 0)')
assert data['Label'].dtype == np.int64
data = relabel_data(data, curr_labels=[-1], new_label=0)
assert len(set(data['Label'])) == n_classes
   

####TODO: prototyping?
#data=data.iloc[:50,:]


#%%
'''**from this point on: code/variables should executed and looked at cell-by-cell i.e. dynamically'''

#%%

print('\n Number of seq in data: ',data.shape[0])


#%%
#we need to do INIT. filtering to our data first

print('checking for duplicate seq.')
assert len(set(data['Sequence']))==len(data['Sequence']),'duplicate AA seq. found!'

print('filtering out sequences with len << 50 AA..')
#but first let's vis. seq lengths
seq_len = [len(row[2]) for row in data.itertuples(index=True)]

#TODO: fix plot
plot_hist_ranges(seq_len,bar_denom=len(data)
                 ,title='Distribution of #proteins by length',x_label='Protein Length'
                   ,curr_path=curr_path,f_name='hist_distr_prot_len')


#now keep seqs that have len>50AAs
data = rm_sh_prots(data,50,f_path=curr_path+'\\BPAD200_prot_less_than_50AA.csv')


print('removing sequences that have ambig. AAs')
#TODO: put this into fn similarly to fn for short proteins
mask = [contains_amb_AAs(seq) for seq in data.loc[:,'Sequence']]
data[mask].to_csv(curr_path+'\\BPAD200_prot_amb_AAs.csv')
data = data[[not m for m in mask]]


#%%
'''basic stats on full data'''

print('\n Number of seq in data: ',data.shape[0])


print('plot #seq under each of the classes in our full train data (in all species ofc)')
###assumption: all classes have >0 seq. belonging to them in data
seq_cl_counts = pd.DataFrame(Counter(data['Label']), index=['Count']).T
print('Class distribution: ',seq_cl_counts)

#TODO: set below
#seq_cl_counts.columns = ['0: Non-enzymes','1: Oxidoreductases', '2: Transferases'
#        , '3: Hydrolases', '4: Lyases', '5: Isomerases', '6: Ligases', '7: Translocases']

#TODO: set xlabel, title, f_name
#calc_freq_plot_hist(seq_cl_counts,freq_denom=None,n_to_plot='all'
#                    ,xlabel='Main EC Class'
#        ,title='Freq. of main EC classes (incl. non-enzymes) in training data'
#        ,curr_path=curr_path,f_name='data_distr_classes')

#%%
print('Sort data based on protein-ID')
#TODO: encapsulate in fn as we use this a lot and we need to ensure it's the same!
data = data.sort_values('Protein',inplace=False,ascending=True).reset_index(
            inplace=False,drop=True)
    

print('Store filtered dataset')
data.to_csv(curr_path+'\\BPAD200_filtered.csv')

#%%

print('\ncompute stats for #sequences in which each ngram is found in in our data')

print('setting ngrams corpus and table')
#### loading ngrams used for Hist-8000 representations for consistency

#set below main params -> code checked only for aa_alphab ='standard_20' , n_gram_len=3
aa_alphab ='standard_20' 
n_gram_len=3

###set path below
#overlap == True and aa_alphab=='standard_20' and n_gram_len==3:
ngrams = pd.read_csv(r'C:\Users\Frixos\OneDrive - University of Southampton\masters\year1\9Mreport\prot_fun_pred\EC_num_pred\9\data'
                     +r'\\stand_20_alphab_3gram_AAs'+'.csv'
                       ,index_col=[0])
ngrams = ngrams.iloc[:,0].values
assert len(ngrams)==n_ngrams,'do we have more ngrams than what we thought?'
overlap=True

### for documenting counts of ngrams here -> across all sequences
n_gram_counts = []
for n_gram in ngrams:
    n_gram_counts.append((n_gram,0))
n_gram_counts=dict(n_gram_counts)

print('loop over sequences, split each into n_grams, count #seq in which each ngram is found at least once ')
for ind,seq in enumerate(tqdm(data['Sequence'])):
    assert contains_amb_AAs(seq)==False,'Sequence contains ambig. AAs..redo pre-processing'
    
    #to keep track of ngrams already observed in curr.seq.-> 
    #and do not have them count >1once for each seq
    curr_n_grams_found = []
    
    #break current sequence into n_grams -> shifted overlapping
    diff = 1
    #TODO: encapsulate this code line in fn! we use it in many places and we want to esnure it's the same exactly
    curr_n_grams = [seq[i:i + n_gram_len] for i in range(0, len(seq) - (n_gram_len-1), diff)]

    #mark n-gram (from defined n_gramS set) that have occurred at least once in the current sequence (as splitted above)
    for n_gram in curr_n_grams:
        if n_gram in curr_n_grams_found:
            continue
        else:
            curr_n_grams_found.append(n_gram)
            n_gram_counts[n_gram]+=1
    
    assert all([(count<=ind+1) for count in n_gram_counts.values()]),'you cant \
        have a ngram observed in >N sequences, while you are currently looking \
            the Nth sequence in the data'
    
    #TODO: calc similar stats for defined AAs, in curr. seq. data
    
assert all([elem < len(data) for elem in n_gram_counts.values()]),'we cant have a ngram '

print('Calc. frequencies for ngrams')

###ngrams
calc_freq_plot_hist(n_gram_counts,freq_denom=len(data),n_to_plot=[15,-25]
        ,xlabel='Tri'+'gram'
        ,title='by percentage of sequences in which '+'tri'+'gram occurs in (training data)'
        ,curr_path=curr_path
        ,f_name='data_hist_'+'tri'+'grams_n_seq_in')



#%%
print('Computing more stats on the ngrams distribution')

#TODO: load distr. stats


#%%
#print('sweetviz for EDA')
