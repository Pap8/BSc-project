# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 02:33:01 2020

@author: Hi
"""


from Load_data_demo import Load_data_demo
from Pipeline import Pipeline
from Load_pipeline import Load_pipeline
from Store_results import Store_results 
import sys

#TODO: sort out code - OOP sw.eng-check notes file
    
    
#%%
if __name__ == '__main__':
    
    #------------user input--------------------------------------------------------
    #
    # r'C:\Users\pfrix\OneDrive - University of Southampton\PycharmProjects\Classifier-1\bpad200.fa'
    input_path = sys.argv[1]
    # r'results_demo'
    output_path= sys.argv[2]
    
    #-----------------load data-----------------------------------------------
    
    trigrams_path=r'C:\Users\pfrix\OneDrive - University of Southampton\PycharmProjects\Classifier-1\protVec_100d_3grams.tsv' 
    #optimal
    pre_processing = 'sum_trigrams'
    scaler_path = r'C:\Users\pfrix\OneDrive - University of Southampton\PycharmProjects\Classifier-1\feature_selection\(demo)17-05-2020_19-35-25\standardise\Standard_Scaler.pkl'
    data_transformer_path= r'C:\Users\pfrix\OneDrive - University of Southampton\PycharmProjects\Classifier-1\feature_selection\(demo)17-05-2020_19-35-25\standardise\PCA\PCA_11.pkl'
    classifier_path = r'C:\Users\pfrix\OneDrive - University of Southampton\PycharmProjects\Classifier-1\evaluation\standardise\PCA\PCs_of_pca_with_11PCs\random\LR/best_model.pkl'
   
    
    Loader = Load_data_demo(input_path)
    input_prots = Loader.load_input_prots()
    
    Loader2 = Load_pipeline(trigrams_path,pre_processing,scaler_path,data_transformer_path,classifier_path)
    trigram_vecs = Loader2.load_trigram_vecs('infer',[0])
    #pre_processing = Loader2.load_pre_processing()
    scaler, data_transformer = Loader2.load_data_transformer()
    classifier= Loader2.load_classifier()

    
    #full pipeline needed!
    print('\nData, Model loaded, starting main pipeline... \n\n')
    
    #-------------- show input dataset for demo --------------------
    print(input_prots.head(10))
    input()
    
    #------------start pipeline--------------------------------------------------------
    Pipeline_ = Pipeline(input_prots,trigram_vecs,pre_processing,scaler,data_transformer,classifier)
    Pipeline_.run()
    results = Pipeline_.get_results()
    
    #------------store results-------------------------------------------------
    print('Storing results..')
    Store_ = Store_results(input_prots,results,output_path)
    Store_.store_results()
    #output_path