from outlier_detection_sd import detect_outlier_sd_lexSemanticSim_auto
import numpy as np
from groupTxt_ByClass import groupItemsBySingleKeyIndex
from word_vec_extractor import extractAllWordVecs
from read_pred_true_text import ReadPredTrueText
from sent_vecgenerator import generate_sent_vecs_toktextdata
from general_util import print_by_group
from sklearn.cluster import SpectralClustering
from txt_process_util import RemoveHighClusterEntropyWordsIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from cluster_file_connected_component import clusterByConnectedComponentIndex
from cluster_file_leadNonOverlapWords import clusterByLeadingOnOverlappingWords
#from general_util import Print_list_pred_true_text

gloveFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/glove.42B.300d/glove.42B.300d.txt"



listtuple_pred_true_text=ReadPredTrueText("result/batchId_PredTrueText1")
newList=[]
i=-1
for pred_true_text in listtuple_pred_true_text:
  i=i+1
  newList.append(pred_true_text+[i,i])
listtuple_pred_true_text=newList  


listtuple_pred_true_text=RemoveHighClusterEntropyWordsIndex(listtuple_pred_true_text)

dic_tupple_class=groupItemsBySingleKeyIndex(listtuple_pred_true_text,0)

#wordVectorsDic = extractAllWordVecs(gloveFile, 300)
 
for label, cluster_pred_true_txt_inds in dic_tupple_class.items():
  _components,newPred_OldPred_true_text_inds=clusterByConnectedComponentIndex(cluster_pred_true_txt_inds)
  #print(_components, newPred_OldPred_true_text_inds)  
  dic_new_tupple_class=groupItemsBySingleKeyIndex(newPred_OldPred_true_text_inds,0)
  
  for newLabel, cluster_newPred_OldPred_true_text_inds in dic_new_tupple_class.items():
    print("newLabel",newLabel)
    #print_by_group(cluster_newPred_OldPred_true_text_inds)
    	
    nparr=np.array(cluster_newPred_OldPred_true_text_inds) 
    new_preds=list(nparr[:,0])
    old_preds=list(nparr[:,1])
    trues=list(nparr[:,2])	
    texts=list(nparr[:,3])
    #print_by_group(cluster_newPred_OldPred_true_text_inds)	
    if len(texts)<2:
      print_by_group(cluster_newPred_OldPred_true_text_inds)	
      continue
    clusterLabels, totalDocsClustered=clusterByLeadingOnOverlappingWords(texts)	  
    '''X=generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
    #vectorizer = TfidfVectorizer( max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
    #X = vectorizer.fit_transform(texts)  
    #outlierLabels=detect_outlier_sd_lexSemanticSim_auto(texts, X)
    clustering = SpectralClustering(n_clusters=3,assign_labels="discretize", random_state=0).fit(X)''' 
    list_sp_pred_true_text_ind_prevPred=np.column_stack((new_preds, old_preds, trues, texts, clusterLabels)).tolist()

    print_by_group(list_sp_pred_true_text_ind_prevPred)
  print("---Each pred group---\n")    
  
	