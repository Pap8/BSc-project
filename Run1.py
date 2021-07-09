#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on _
@author: Frix
"""

## ?? conda upgrade python==3.9
### !pip install sweetviz keras matplotlib pandas numpy scikit-learn tensorflow biopython tqdm seaborn imbalanced-learn xgboost

import sweetviz as sv
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
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras

#for importing utility class files from coding_templates dir. so that 
#we can use the same ones across all different projects:
#https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(1, r'C:\Users\fp1n17\OneDrive - University of Southampton\coding_templates')

from Frix_p_vecs import create_pca_prot_vecs
from Frix_utils import init_run,prototype_data,load_trigram_vecs,relabel_data, store_res \
    ,build_hist_gram_counts# , \#plot_scatter_chart, save_df_in_fasta, from_fasta, rm_sh_prots, filter_out_rare_AAs, plot_line_chart, build_hist_gram_counts, comp_sse
from utils_data import data_cleaning, data_split, resampl #,vis_model, t_sne,q_eval_classif_model
from Hierarchical_clustering import Hierarchical_clustering
## from Classifier_model import Classifier_model


# %%
# TODO: #imporve on using template py code file in every new proj  and removing todos etc-> create abstract class/interface to import from coding_templates folder to each experiment-RUN class and code from there
# ** code in this file is written by having in mind that representation modelling and classification steps are separate!
# TODO: * see sw_eng_tips_notes.txt notes file for more ideas
# TODO: see experiment.xlsx file for curr. proj. for other data preprocess todo
#%%
#%%

if __name__ == '__main__':
    ''''''
    ''''''
    '''#FOR ALL OPERATIONS:'''
    rand_state=10

    '''experiment-run settings'''
    '''set curr. experiment-run ID'''
    run_id = 1

#### prototyping?
    prototype = False
    
    #parallelise creation of BoT/HoT vectors?
    parallel=False

    print('Creating experiment-run and results file locations')
    pc_path = r'C:\Users\Frixos\OneDrive - University of Southampton\PycharmProjects\covid_git'
    uni_path = r'C:\Users\fp1n17\OneDrive - University of Southampton\PycharmProjects\covid_git'

    run_path, curr_path,user = init_run(run_id, pc_path, uni_path,create_curr_path=True, prototype=prototype)

    ''''#pipelines results'''
    res = []

    ''''''
    '''starting defining pipelines'''
    ###first pipe_id val. will be added to the res variable when we add the first results of the 1st pipeline
    pipe_id=0

    '''pipeline settings'''
    reduce_resampl_ = [False] #['undersample',None]


    ''''''
    '''load basic data //OR ready datasets for modelling'''
    print('Loading initial datasets..')
    data =pd.read_csv(r'1\EDA\BPAD200_filtered.csv',index_col=[0]) #,index_col=[0])
    
    assert set(['Protein', 'Sequence', 'Label']).issubset(data.columns.tolist())
    data = data[['Protein','Sequence','Label']]
    n_classes = 2
    
    data = data.sort_values('Protein',inplace=False,ascending=True).reset_index(
            inplace=False,drop=True)
    
    
    print('Loading pre-trained ProtVec trigram vectors.')
    f_path_2 = r'C:\Users\\'+str(user)+'\\OneDrive - University of Southampton\masters\year1\9Mreport\prot_fun_pred\EC_num_pred\protVec_100d_3grams.csv'
    trigram_data = load_trigram_vecs(f_path=f_path_2
                    ,rm_ambig_3gram=True,use_avg_amb_3gram_vec=False)
    
    
###### prototyping?
    if prototype == True:
        print('*Prototyping: selecting subset of data-points for rapid prototyping')
        data = prototype_data(data,n_classes)
        # assert
    
    
    ''''''
    '''build protein representations ---> see also exp2, exp3, exp4(for further vector scaling) -> EC_num_pred proj'''
    print('Build protein representations for protein data, for pipeline ',pipe_id)
    
    #assert set(['Protein', 'Sequence', 'Label']).issubset(data.columns.tolist())

    ##########ProtVec, PCA
    print('build ProtVec-stand-PCA protein representations')
    pca_prot_data, res, curr_fname = create_pca_prot_vecs(data, trigram_data
                   ,steps_sum=1, scale_method='standardise'
        , curr_res=res,curr_path=curr_path, pipe_id=pipe_id, rand_state=rand_state)

    
    #####HCs derived representations
    
    ## for re-using clustered trigrams from prev.runs
    data_dir_1 = r'C:\Users\\'+str(user)+r'\\OneDrive - University of Southampton\masters\year1\9Mreport\prot_fun_pred\EC_num_pred\6\data'
    data_dir_2 = r'C:\Users\\'+str(user)+r'\\OneDrive - University of Southampton\masters\year1\9Mreport\prot_fun_pred\EC_num_pred\9\data'
    data_dir_3 = r'C:\Users\\'+str(user)+r'\\OneDrive - University of Southampton\masters\year1\9Mreport\prot_fun_pred\EC_num_pred\16\from_iridis4-2\EC_num_16\16\16\24-05-2021_13-26-56'
    data_dir_4 = r'C:\Users\\'+str(user)+r'\\OneDrive - University of Southampton\masters\year1\9Mreport\prot_fun_pred\EC_num_pred\16\from_iridis4-2\EC_num_16\16\16\24-05-2021_18-18-18'
    #to get the dir with clustered trigram vectors
    data_dir_3 = data_dir_3+r'\\_'
    data_dir_4 = data_dir_4+r'\\_'
    
    
    hc_scores= []
    print('HC on trigram vectors to built BoW-like protein vectors')
    for n_clusters in list(np.arange(20,320,20))+[400,500,1000,2000,4000]:
        print('----------------------------------------------------------')
        
        ### we want to use the cluster-IDs assigned to trigram_vecs from prev experiment
        if n_clusters in np.arange(200,320,20):
            trigram_data = data_dir_2+r'\hc' + \
                str(n_clusters) + 'cl_'+'trigram_100d_vecs_clusters'+'.csv'
        elif n_clusters in np.arange(20,200,20):
            trigram_data = data_dir_1+r'\hc' + \
                str(n_clusters) + 'cl_'+'trigram_100d_vecs_clusters'+'.csv'
        elif n_clusters in (400,500,1000):
            trigram_data = data_dir_3+r'\hc' + \
                str(n_clusters) + 'cl_'+'trigram_100d_vecs_clusters'+'.csv'
        elif n_clusters in (2000,4000):
            trigram_data = data_dir_4+r'\hc' + \
                str(n_clusters) + 'cl_'+'trigram_100d_vecs_clusters'+'.csv'
        
        
        hc1 = Hierarchical_clustering(trigram_data=trigram_data
            ,prot_data=data, num_clusters=n_clusters,normalis=False, tfidf=False
                                      ,prototype=prototype,parallel=False
                                      ,pipe_id=pipe_id,d_path = curr_path)

        hc_scores.extend(hc1.get_hc_scores())

    print('HC results and primary analysis already done from prev. experiment..')
    #res = pd.DataFrame(hc_scores)
    #res.to_csv(curr_path+r'\results.csv')
    
    del trigram_data
    
    ######and now we build the Hist-8000 representations for proteins'''
    print('Build (histogram of the counts of the 8000 3-gram) representations for protein sequences')
    n_gram_len=3 
    prot_data = build_hist_gram_counts(data,aa_alphab='standard_20',n_gram_len=n_gram_len
                ,overlap=True,curr_path=curr_path
        ,f_name='hist_'+str(n_gram_len)+'gram_counts_vecs')#+'_part_'+str(part))
