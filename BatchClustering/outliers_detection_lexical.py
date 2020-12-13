from groupTxt_ByClass import groupTxtByClass_Txtindex
from txt_process_util import getDocFreq
from txt_process_util import getDicTxtToClass_WordToTxt
from txt_process_util import getDicWordToClusterEntropy
from nltk.tokenize import word_tokenize
from print_cluster_evaluation import printClusterEvaluation_list
import collections
from general_util import Print_list_pred_true_text
from general_util import print_by_group

def outlierConditionMoreCommonWordWithOtherCluster1(text,dic_word_to_clusterEntropy, dic_word_cluster_indecies):
  words = word_tokenize(text)
  
  #find how many words have entropy more than avg entropy
  entSum=0
  for word in words:
    entVal = dic_word_to_clusterEntropy[word]
    entSum=entSum+entVal
  avgEnt=entSum/len(words)
  
  entGT_Avg_count=0
  for word in words:
    entVal = dic_word_to_clusterEntropy[word]    
    if entVal>avgEnt:
      entGT_Avg_count=entGT_Avg_count+1

  #if entGT_Avg_count>len(words)/2.0:
  if entGT_Avg_count>1:  
    return True
  
  return False
'''def outlierConditionMoreCommonWordWithOtherCluster(dic_word_cluster_indecies):
  for word, cluster_indecies in dic_word_cluster_indecies.items():
    word_cluster_appearance_counters=collections.Counter(cluster_indecies)  
    count_values=list(word_cluster_appearance_counters.values())
    if max(count_values)==min(count_values) and len(count_values)>1:
      return True	
  
  return False
'''
def outliersMoreCommonWordWithOtherCluster1(listtuple_pred_true_text):
  outlier_signs_more_commonWords_in_more_clusters=[]
  
  dic_txt_to_cluster, dic_word_to_txt=getDicTxtToClass_WordToTxt(listtuple_pred_true_text)
  
  dic_word_to_clusterEntropy, dic_word_to_cluster_indecies=getDicWordToClusterEntropy(dic_txt_to_cluster,dic_word_to_txt)

  #group text by label, and find outlier in each group
  
  '''for i in range(len(listtuple_pred_true_text)):
    pred_true_text= listtuple_pred_true_text[i]   
    text = pred_true_text[2]
    words = word_tokenize(text)
    #clusterIndecies=[]
    print("###"+text)
    dic_word_cluster_indecies={}	
    for word in words:
      txtIndecies = dic_word_to_txt[word]
      word_cluster_indecies=[]	 	  
      for txtIndex in txtIndecies:
        #clusterIndecies.append(dic_txt_to_cluster[txtIndex])
        word_cluster_indecies.append(dic_txt_to_cluster[txtIndex])		
      dic_word_cluster_indecies[word]=word_cluster_indecies    
    
    #print(dic_word_cluster_indecies)
    #distinctClusters=set(clusterIndecies)
    if outlierConditionMoreCommonWordWithOtherCluster(dic_word_cluster_indecies): #need to use word_entropy
      outlier_signs_more_commonWords_in_more_clusters.append(-1)
    else:
      outlier_signs_more_commonWords_in_more_clusters.append(1)	
    #if len(distinctClusters)>1:
    #  outlier_signs_more_commonWords_in_more_clusters.append(-1)
    #else:
    #  outlier_signs_more_commonWords_in_more_clusters.append(1)	
	'''
  for i in range(len(listtuple_pred_true_text)):
    pred_true_text= listtuple_pred_true_text[i]   
    text = pred_true_text[2]
    flag = outlierConditionMoreCommonWordWithOtherCluster1(text, dic_word_to_clusterEntropy, dic_word_to_cluster_indecies)
    if flag==True:
      outlier_signs_more_commonWords_in_more_clusters.append(-1)
    else:
      outlier_signs_more_commonWords_in_more_clusters.append(1)	
  	
  return outlier_signs_more_commonWords_in_more_clusters  


def outliersNoCommonWordWithOthers(listtuple_pred_true_text):
  outlier_signs_no_commonWords=[1]*len(listtuple_pred_true_text)
  
  dic_tupple_class_Txtindex=groupTxtByClass_Txtindex(listtuple_pred_true_text, False)

  for label, pred_true_txts_index in dic_tupple_class_Txtindex.items():
    texts_indecies=[]
    texts=[]	
    for pred_true_txt_index in pred_true_txts_index:
      texts_indecies.append([pred_true_txt_index[2], pred_true_txt_index[3]])
      texts.append(pred_true_txt_index[2])	  
    
    dicDocFreq_In_a_cluster=getDocFreq(texts)  
    for txt_index in texts_indecies:
      dfCount=0
      txt = txt_index[0]
      indexInOriginalList=txt_index[1]	  
      words = word_tokenize(txt)
      for word in words:
        dfCount=dfCount+dicDocFreq_In_a_cluster[word]
      if dfCount==len(words):
        outlier_signs_no_commonWords[indexInOriginalList]=-1
  
  avgItemsInCluster=len(listtuple_pred_true_text)/len(dic_tupple_class_Txtindex)
  
  return outlier_signs_no_commonWords, avgItemsInCluster  

def DetectNonOutliersLexical(listtuple_pred_true_text):
  non_outlier_pred_true_txts=[] 
  outlier_pred_true_txts=[]
  outlier_pred_true_txts_no_common=[]
  outlier_pred_true_txts_more_common=[]    

  outlier_signs_no_commonWords, avgItemsInCluster=outliersNoCommonWordWithOthers(listtuple_pred_true_text) 
  
  outlier_signs_more_commonWords=outliersMoreCommonWordWithOtherCluster1(listtuple_pred_true_text)
      	
  for i in range(len(outlier_signs_no_commonWords)):
    if outlier_signs_no_commonWords[i]==-1:
      outlier_pred_true_txts_no_common.append(listtuple_pred_true_text[i])
    elif outlier_signs_more_commonWords[i]==-1: #need to use later	  
      outlier_pred_true_txts_more_common.append(listtuple_pred_true_text[i])	
    else:
      non_outlier_pred_true_txts.append(listtuple_pred_true_text[i])	  

  outlier_pred_true_txts=outlier_pred_true_txts_no_common+outlier_pred_true_txts_more_common	  
	    
  print("non_outlier_pred_true_txts="+str(len(non_outlier_pred_true_txts)))
  print_by_group(non_outlier_pred_true_txts)
  
  print("outlier_pred_true_txts_more_common="+str(len(outlier_pred_true_txts_more_common)))
  print_by_group(outlier_pred_true_txts_more_common)

  print("outlier_pred_true_txts_no_common="+str(len(outlier_pred_true_txts_no_common)))
  print_by_group(outlier_pred_true_txts_no_common)
  
  print("DetectOutliersLexical: total #texts="+str(len(listtuple_pred_true_text)))
  printClusterEvaluation_list(listtuple_pred_true_text)
  
  print("DetectOutliersLexical: total #non outliers="+str(len(non_outlier_pred_true_txts)))
  printClusterEvaluation_list(non_outlier_pred_true_txts)

  print("DetectOutliersLexical: total #outliers="+str(len(outlier_pred_true_txts)))
  printClusterEvaluation_list(outlier_pred_true_txts)  
  
  return [non_outlier_pred_true_txts, outlier_pred_true_txts, avgItemsInCluster]  