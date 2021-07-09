# -*- coding: utf-8 -*-
#!/usr/bin/env python3

## ?? conda upgrade python==3.9
### !pip install sweetviz keras matplotlib pandas numpy scikit-learn tensorflow biopython tqdm seaborn imbalanced-learn xgboost

import os
import pickle
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
import matplotlib
from collections import Counter

#for importing utility class files from coding_templates dir. so that 
#we can use the same ones across all different projects:
#https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
try:
    #set path -> are we or on PC or Uni-ECS-GPU-comp?
    sys.path.insert(1, r'C:\Users\Frixos\OneDrive - University of Southampton\coding_templates')
except FileNotFoundError:
    print('Template code files not found. Assuming we are on IRIDIS and moving on...')

from Frix_utils import intensity_plot,plot_roc_curve_nfoldCV_grouped\
        ,calc_ovrl_cm,plot_group_box_plot
#import init_run,prototype_data,load_trigram_vecs,relabel_data, store_res#, \
#plot_scatter_chart, save_df_in_fasta, from_fasta, rm_sh_prots, filter_out_rare_AAs, plot_line_chart, build_hist_gram_counts, comp_sse
from utils_data import data_cleaning, data_split, resampl #,vis_model, t_sne,q_eval_classif_model


#%%
'''loading'''
'''#FOR ALL OPERATIONS:'''
rand_state=10
#set curr.working dir.
os.chdir(r'C:\Users\Frixos\OneDrive - University of Southampton\PycharmProjects\covid_git')
#set run_id
exp_id=2
exp_path = str(exp_id)
curr_path = os.path.join(exp_path,'Eval')
if os.path.exists(curr_path)==False:
    os.mkdir(curr_path)

#%%
'''EVALUATING '''
#set below
n_classes=2
class_labels=['Non-BPA','BPA']
#set path for results.csv file of whole exp-run
res_path = r'C:\Users\Frixos\OneDrive - University of Southampton\PycharmProjects\covid_git\2\10-06-2021_21-09-59'
#\results.csv

#%%

print('plot overall ROC-CURVEs from k-fold C.V. (mean+-std.dev), \
      for each of the specified models (in terms of metric scores)')

#PASS path for accessing .csv file with predictions-probabilities predicted\
#for all folds for all pipelines we want to plot ROCs for 
#adapt values below in `det` as needed
g_path = r'2\10-06-2021_21-09-59'
det = pd.DataFrame([
    {'method':'Sum-of-trigrams','f_path': g_path+r'\\20'}
    ,{'method':'Hist-8000','f_path': g_path+r'\\21'}
        ,{'method':'HoT-4000','f_path': g_path+r'\\19'}
        #,{'method':'HoT-2000','f_path': g_path+r'\\18'}
        #,{'method':'HoT-1000','f_path': g_path+r'\\17'}
        #,{'method':'HoT-40','f_path': g_path+r'\\1'}
        ,{'method':'HoT-20','f_path': g_path+r'\\0'}])


plot_roc_curve_nfoldCV_grouped(det,max_fpr=0.05
                               ,curr_path=curr_path,f_name='roc_auc_curve_nfoldCV_grouped')


#%%
print('Calc. conf.matr. and viz-store, for pipelines that intrest us')

#adapt values below in `det` as needed
det=pd.DataFrame([{
    'method':'Sum-of-trigrams'
        ,'pipe_id': 20}
    ,{'method':'Hist-8000'
          ,'pipe_id': 21}
    ,{'method':'HoT-4000'
          ,'pipe_id': 19}
    ,{'method':'HoT-2000'
          ,'pipe_id': 18}
    ,{'method':'HoT-1000'
          ,'pipe_id': 17}
    ,{'method':'HoT-40'
          ,'pipe_id': 1}
    ,{'method':'HoT-20'
          ,'pipe_id': 0}])

for row in det.itertuples(index=True):
    method=row[1]
    pipe_id= row[2]
    pipe_path = res_path+r'\\'+str(pipe_id)
    
    #TODO: convert vals shown on conf.matrix to % ??
    calc_ovrl_cm(pipe_path=pipe_path, iridis_paths=False
                 ,class_labels = class_labels
        ,title=method+' Overall confusion matrix: mean +- std.dev.'
        ,d_path=curr_path
        ,f_name='ovrl_cm_pipe_ID='+str(pipe_id))





#%%
print('grouped box plot scores k-foldCV plot')

###set below
metric = 'AUC'
res_df = pd.read_csv(res_path+r'\\results.csv',index_col=[0])
col_id = res_df.columns.tolist().index('outer_runs_details')

#adapt values below in `det` as needed
det2=pd.DataFrame([{
    'method':'Sum-of-trigrams'
        ,'scores': eval(res_df.iloc[20,col_id])['test_score']}
    ,{'method':'Hist-8000'
          ,'scores': eval(res_df.iloc[21,col_id])['test_score']}
    ,{'method':'HoT-4000'
          ,'scores': eval(res_df.iloc[19,col_id])['test_score']}
    ,{'method':'HoT-2000'
          ,'scores': eval(res_df.iloc[18,col_id])['test_score']}
    ,{'method':'HoT-1000'
          ,'scores': eval(res_df.iloc[17,col_id])['test_score'] }
    ,{'method':'HoT-40'
          ,'scores': eval(res_df.iloc[1,col_id])['test_score'] }
    ,{'method':'HoT-20'
          ,'scores': eval(res_df.iloc[0,col_id])['test_score'] }])

plot_group_box_plot(det2,title=metric+' - 10foldCV'# (mu+-std.)'
                    ,y_label=metric
            ,curr_path=curr_path,f_name='grouped_box_plot_scores_nfoldCV')



#%%
