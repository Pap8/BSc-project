#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on _
@author: Frix
"""

## !conda update python (check==3.9)
### !pip install sweetviz keras matplotlib pandas numpy scikit-learn tensorflow biopython tqdm seaborn imbalanced-learn xgboost

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
    #set path
    sys.path.insert(1, r'C:\Users\fp1n17\OneDrive - University of Southampton\coding_templates')
except FileNotFoundError:
    print('Template code files not found. Assuming we are on IRIDIS and moving on...')

#from Frix_p_vecs import create_pca_prot_vecs #, data_scaling, pca
from Frix_utils import init_run,prototype_data,load_trigram_vecs,relabel_data, store_res,calc_ovrl_cm#, \
#plot_scatter_chart, save_df_in_fasta, from_fasta, rm_sh_prots, filter_out_rare_AAs, plot_line_chart, build_hist_gram_counts, comp_sse
from utils_data import data_cleaning, data_split, resampl #,vis_model, t_sne,q_eval_classif_model
#from Hierarchical_clustering import Hierarchical_clustering
from Classifier_model import Classifier_model

# %%
# TODO: #imporve on using template py code file in every new proj  and removing todos etc-> create abstract class/interface to import from coding_templates folder to each experiment-RUN class and code from there
# ** code in this file is written by having in mind that representation modelling and classification steps are separate!
# TODO: * see sw_eng_tips_notes.txt notes file for more ideas
# TODO: see experiment.xlsx file for curr. proj. for other processES todo
# %%


#%%%
''''''
''''''
'''#FOR ALL OPERATIONS:'''
rand_state=10

'''experiment-run settings'''
'''set curr. experiment-run ID'''
#set run-id
run_id = 2


####TODO: prototyping?
prototype = False

#parallelise creation of BoT/HoT vectors?
parallel=False

print('Creating experiment-run and results file locations')
#set paths below
pc_path = r'C:\Users\Frixos\OneDrive - University of Southampton\PycharmProjects\covid_git'
uni_path = r'C:\Users\fp1n17\OneDrive - University of Southampton\PycharmProjects\covid_git'

run_path, curr_path,user = init_run(run_id, pc_path, uni_path,create_curr_path=True, prototype=prototype)

#%%
''''#pipelines results'''
res = []

''''''
'''starting defining pipelines'''
###first pipe_id val. will be added to the res variable when we add the first results of the 1st pipeline
pipe_id=0

'''pipeline settings'''
# UNDERSAMPLE = sample randomly #seq for each class = #seq for smallest sized class (for building HC-based vectors and classification part)
reduce_resampl_ = [False] #['undersample',None]



#%%

''''''
''''''
'''load basic data //OR ready datasets for modelling'''
print('Loading initial datasets..')

n_classes = 2
class_labels = ['Non-BPA','BPA']



#%%

# prototyping=not here

''''''
'''initial cleaning/pre-processing//prep of data dependent on task'''
print('..')
#DATA CLEAN=not now

## ###potential prototyping values/code here

#assert


''''''

#EDA -> to be done in separate EDA_Run<XYZ> class (EDA on input classifier data)





#%%
'''FOR CLASSIFICATION EXPERIMENT-RUN ->'''
    
''''''
''''''
''' load res+ protein representations  for next modelling steps below'''
print('\nSpecifying datasets for classification')
data_dir = r'1\10-06-2021_13-22-24'

###### loop over dataset files in data_dir asnd save info in `datasets_info`
datasets_info = []
###### set n_clusters ranges and loops as required
for n_clusters in list(np.arange(20,320,20))+[400,500,1000,2000,4000]:
    datasets_info.append({'method': 'hot_vecs', 'dims': n_clusters
        , 'f_path': data_dir + r'\\hc' + str(n_clusters) + 'cl_HoT_vecs.csv'})
    
datasets_info.append({'method': 'protvec', 'dims': 100
        ,'f_path':data_dir+r'\\p_vecs_sum_tri.csv'})

datasets_info.append({'method': 'hist-8000', 'dims': 8000
        ,'f_path':data_dir+r'\\hist_3gram_counts_vecs.csv'})
    

datasets_info = pd.DataFrame(datasets_info)


#%%
print('\nSpecified datasets for classification, now specifying hyperparameters')

''''''
'''define models-hparams outside of pipelines loop!'''

print('Defining some hyper-parameters')
#### some hparams defined here as we might want to search in their space for tuning
#### metric is the same for all models -> but metric is defined below in hparams 
##-> only because different packages have different names for the same metric
models_hparam = { 'SVM':{'C':1, 'kernel':'linear' } }##'LinearSVC': {'C': [1]} }    
split_method = 'normal'
outer_splits=10
inner_splits=1
lobov_data_dir = 'bpad200_lobov_splits_indic'
main_metric= 'roc_auc_score'#'f1_score'


''''''
'''PIPELINES'''
''''''
'''loop over datasets for creating pipelines'''

print('Started pipelines loop.')
for row in datasets_info.itertuples(index=True):
    #  additional loop for resampling strategies
    for reduce_resampl in reduce_resampl_:
        method1 = row[1]
        n_inp_feats = row[2]
        
        print('Loading data...')
        print('Loading data...')
        verbose=False
        if n_inp_feats>999:
            verbose=True
        #memoery_map and engine options for making loading more efficient
        try:
            data = pd.read_csv(row[3],index_col=[0],verbose=verbose
                           ,memory_map=True,engine='c')
        except FileNotFoundError:
            ####### handle dataset f_path below when we are on IRIDIS!
            data = pd.read_csv('hc'+row[3].split('hc')[1]
                             ,index_col=[0],verbose=verbose
                           ,memory_map=True,engine='c')
        #reduce comp.complexity avoid--> assert data.columns[-1] == 'Label' and all(data.columns[:2] == ['Protein', 'Sequence'])
        #reduce comp.complexity avoid-->assert all([data.iloc[:, col].dtype == object for col in (0, 1)]), 'check again dtypes: ' + data.dtypes
        #reduce comp.complexity avoid-->assert data.Label.dtype == 'int64', 'Label col dtype /=int64...fix!'
        #reduce comp.complexity avoid-->assert len(data.columns[2:-1])==n_inp_feats

        ###sort data-points here to be consistent for all datasets to be used as classifier input
        data = data.sort_values('Protein',inplace=False,ascending=True).reset_index(
            inplace=False,drop=True)

        #data cleaning
        print('Data cleaning..')
        #reduce comp.complexity avoid-->data = data_cleaning(data)
        #reduce comp.complexity avoid-->assert set(['Protein', 'Sequence', 'Label']).issubset(data.columns.tolist())
        
        #data resampling?
        if reduce_resampl!=False:
            data = resampl(data, n_classes, method=reduce_resampl
                           , size=len(data[data['Label'] == 1]), uniform=True
                           , replace=False, curr_path=curr_path
                           , f_name=method1 + str(n_inp_feats))
        
            ###sort data-points again after resampling 
            data = data.sort_values('Protein',inplace=False,ascending=True).reset_index(
                inplace=False,drop=True)
            
        # select 1st opt_n_pcs PCs (features of data df) in case of PCA
        if method1 == 'protvec_pca':
            print('Selecting optimal #PCs from data')
            data = pd.concat([data.iloc[:, :2 + opt_n_pcs], data['Label']], axis=1)
        
### code for prototyping -> select subset of data in each classif. model for each dataset
        if prototype:
            print('*Prototyping: selecting subset of data-points for rapid prototyping')
            data = prototype_data(data,n_classes)
            assert len(set([val for val in Counter(data['Label']).values()])) ==1

        #########loop over different pipeline configurations
        for model_hp in models_hparam.items():
            print('Creating pipeline folder (pipeline_ID: ', pipe_id, ' )')
            pipe_path = os.path.join(curr_path, str(pipe_id))
            os.mkdir(pipe_path)

            classif = model_hp[0]
            hparams_space = model_hp[1]
            
            #########define models-hparams
            print('No hyperparameters that depend on dataset')
            ###we don't have an MLP model here
            #next 2 var.assignments are just to be consistent with `Classifier_model` class
            #hparams_space['input_dim']=input_dim
            #hparams_space['n_hidd_nodes']=n_hidd_nodes
            

            Classifier_model_ = Classifier_model(dataset=data
                  ,n_inp_feats=n_inp_feats,n_classes=n_classes
                  ,class_labels=class_labels,method1=method1
                  ,classif=classif,hparams_space=hparams_space
                    ,split_method=split_method,
                 lobov_data_dir=lobov_data_dir, outer_splits=outer_splits
                , inner_splits=inner_splits , main_metric=main_metric
                ,store_model = True
                , run_path=run_path, pipe_path=pipe_path, pipe_id=pipe_id
                             , rand_state=10 )
                             
            Classifier_model_.outer_cv()
            best_run_res = Classifier_model_.get_run_details()

            print('Adding results for pipeline ID ', pipe_id)
            res.append({'pipe_id': pipe_id, 'method1': method1, 'dims': n_inp_feats
                        ,'n_classes':n_classes,'class_labels':class_labels
                        ,'reduce_resampl':reduce_resampl,**best_run_res})
            
            ######store results after each pipeline finishes to have running status of experiment --> and store in the end ofc
            print('Storing current results..')
            #store_res(res, curr_path)-
            res_df = pd.DataFrame(res,index=np.arange(len(res))).set_index('pipe_id', inplace=False, drop=True)
            res_df.to_csv(curr_path + r'\\results.csv')

            pipe_id += 1
            

print('Calc+viz ovrL conf matr. for best pipeline OF RUN')
#sort df so that best peforming pipleine is first
best_pipe_id = res_df.sort_values('mean_test_score',ascending=False,inplace=False).iloc[0,:].name
best_pipe_path = curr_path+r'\\'+str(best_pipe_id)

calc_ovrl_cm(pipe_path=best_pipe_path, iridis_paths=False
             ,class_labels = class_labels
    ,title='Overall confusion matrix: mean +- std.dev.'
    ,d_path=curr_path
    ,f_name='ovrl_cm_best_pipe_ID='+str(best_pipe_id))



''''''
print('END')
