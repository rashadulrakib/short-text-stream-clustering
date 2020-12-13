from cluster_file_connected_component import clusterByConnectedComponent
from cluster_file_connected_component import clusterByConnectedComponentIndex
from groupTxt_ByClass import groupTxtByClass
from groupTxt_ByClass import groupTxtByClass_Txtindex
import numpy as np
from collections import Counter

'''def outlierBySmallestGroupsIndex(newPred_OldPred_true_text_inds):
  outliersInCluster_Index=[]
  non_outliersInCluster_Index=[]
   
  rows=len(newPred_OldPred_true_text_inds)
  np_arr=np.array(newPred_OldPred_true_text_inds)
  new_preds=np_arr[:,0].tolist()

  new_pred_dict=Counter(new_preds)
  #maxKey = max(new_pred_dict, key=new_pred_dict.get)
  minKey = min(new_pred_dict, key=new_pred_dict.get)
  #newPredMaxLabel= maxKey
  newPredMinLabel= minKey
  
  for newPred_OldPred_true_text_ind in newPred_OldPred_true_text_inds:
    newPredLabel=newPred_OldPred_true_text_ind[0]
    oldPredLabel=newPred_OldPred_true_text_ind[1]
    trueLabel=newPred_OldPred_true_text_ind[2]
    text=newPred_OldPred_true_text_ind[3]
    ind=newPred_OldPred_true_text_ind[4]	
    OldPred_true_text_ind=[oldPredLabel, trueLabel, text, ind]	
    #if str(newPredLabel)==str(newPredMaxLabel):
    if len(new_pred_dict)>1:
      if str(newPredLabel)!=str(newPredMinLabel): 	
        non_outliersInCluster_Index.append(OldPred_true_text_ind)
      else:
        outliersInCluster_Index.append(OldPred_true_text_ind)
    else:
      outliersInCluster_Index.append(OldPred_true_text_ind)	
    #if str(newPredLabel)!=str(newPredMinLabel): 	
    #  non_outliersInCluster_Index.append(OldPred_true_text_ind)
    #else:
    #  outliersInCluster_Index.append(OldPred_true_text_ind) 

  #print(outliersInCluster_Index)	  
  
  return [outliersInCluster_Index, non_outliersInCluster_Index]  
'''

def outlierBySmallestGroupsIndex(newPred_OldPred_true_text_inds):
  outliersInCluster_Index=[]
  non_outliersInCluster_Index=[]
   
  rows=len(newPred_OldPred_true_text_inds)
  np_arr=np.array(newPred_OldPred_true_text_inds)
  new_preds=np_arr[:,0].tolist()

  new_pred_dict=Counter(new_preds)
  maxKey = max(new_pred_dict, key=new_pred_dict.get)
  newPredMaxLabel= maxKey
  
  for newPred_OldPred_true_text_ind in newPred_OldPred_true_text_inds:
    newPredLabel=newPred_OldPred_true_text_ind[0]
    oldPredLabel=newPred_OldPred_true_text_ind[1]
    trueLabel=newPred_OldPred_true_text_ind[2]
    text=newPred_OldPred_true_text_ind[3]
    ind=newPred_OldPred_true_text_ind[4]	
    OldPred_true_text_ind=[oldPredLabel, trueLabel, text, ind]	
    if str(newPredLabel)==str(newPredMaxLabel):
      non_outliersInCluster_Index.append(OldPred_true_text_ind)
    else:
      outliersInCluster_Index.append(OldPred_true_text_ind) 

  #print(outliersInCluster_Index)	  
  
  return [outliersInCluster_Index, non_outliersInCluster_Index]  
  
def outlierBySmallestGroups(newPred_OldPred_true_texts):
  outliersInCluster=[]
  non_outliersInCluster=[]
  rows=len(newPred_OldPred_true_texts)  
  
  #np.concatenate((A[:,0].reshape(2,1), A[:,2:4]),axis=1)  
  np_arr=np.array(newPred_OldPred_true_texts)
  newPred_true_texts=np.concatenate((np_arr[:,0].reshape(rows,1), np_arr[:,2:4]),axis=1).tolist()  
  
  dic_tupple_class=groupTxtByClass(newPred_true_texts, False)

  maxGroupSize=-10
  newPredMaxLabel=""  
  totalGroups = len(dic_tupple_class)
  #print("totalGroups by connComp="+str(totalGroups))  
  
  for label, pred_true_txts in dic_tupple_class.items():
    groupSize=len(pred_true_txts)
    if maxGroupSize<groupSize:
      newPredMaxLabel=label
      maxGroupSize=groupSize	  

  for newPred_OldPred_true_text in newPred_OldPred_true_texts:
    newPredLabel=newPred_OldPred_true_text[0]
    oldPredLabel=newPred_OldPred_true_text[1]
    trueLabel=newPred_OldPred_true_text[2]
    text=newPred_OldPred_true_text[3]
    OldPred_true_text=[oldPredLabel, trueLabel, text]	
    if str(newPredLabel)==str(newPredMaxLabel):
      non_outliersInCluster.append(OldPred_true_text)
    else:
      outliersInCluster.append(OldPred_true_text) 	
  '''if totalGroups>1:
    #split the group into outlier/non-outliers"
    non_outliersInCluster=dic_tupple_class[newPredMaxLabel]
    for label, pred_true_txts in dic_tupple_class.items():
      if label!=newPredMaxLabel:
        outliersInCluster.extend(pred_true_txts)	  
  else:
    non_outliersInCluster=newPred_OldPred_true_texts'''

  	
      
  return [outliersInCluster, non_outliersInCluster]
  
'''def removeOutlierConnectedComponentLexicalIndex(listtuple_pred_true_text):
  avgItemsInCluster=0
  outlier_pred_true_text_inds=[]
  non_outlier_pred_true_text_inds=[]
  
  dic_tupple_class_Txtindex=groupTxtByClass_Txtindex(listtuple_pred_true_text, False) 
  #index is the original index in listtuple_pred_true_text

  for label, pred_true_txt_inds in dic_tupple_class_Txtindex.items():
    _components,newPred_OldPred_true_text_inds=clusterByConnectedComponentIndex(pred_true_txt_inds)
    #find the smallest components and use it as outliers	
    outliersInCluster_Index,non_outliersInCluster_Index=outlierBySmallestGroupsIndex(newPred_OldPred_true_text_inds)
	
    outlier_pred_true_text_inds.extend(outliersInCluster_Index)
    non_outlier_pred_true_text_inds.extend(non_outliersInCluster_Index) 	
	
  avgItemsInCluster=len(listtuple_pred_true_text)/len(dic_tupple_class_Txtindex) 
  

  return [outlier_pred_true_text_inds, non_outlier_pred_true_text_inds, avgItemsInCluster] ''' 
  
'''def removeOutlierConnectedComponentLexical(listtuple_pred_true_text):
  outlier_pred_true_texts=[] 
  non_outlier_pred_true_txts=[] 
  avgItemsInCluster=0

  dic_tupple_class=groupTxtByClass(listtuple_pred_true_text, False)

  for label, pred_true_txts in dic_tupple_class.items():
    _components,newPred_OldPred_true_texts=clusterByConnectedComponent(pred_true_txts)
    #find the smallest components and use it as outliers	
    outliersInCluster,non_outliersInCluster=outlierBySmallestGroups(newPred_OldPred_true_texts)
	
    outlier_pred_true_texts.extend(outliersInCluster)
    non_outlier_pred_true_txts.extend(non_outliersInCluster) 	
	
  avgItemsInCluster=len(listtuple_pred_true_text)/len(dic_tupple_class)
    
  return [outlier_pred_true_texts, non_outlier_pred_true_txts, avgItemsInCluster]'''
  
def removeOutlierConnectedComponentLexicalByItem(listtuple, batchDocs, maxPredLabel): 
   
  outliers=[] 
  non_outliers=[] 
  avgItemsInCluster=0

  dic_tupple_class=groupTxtByClass(listtuple, False)

  for label, items in dic_tupple_class.items():
    #_components,newPred_items=clusterByConnectedComponentByItem(items)
    print("to do")	
  
  
  return [outliers, non_outliers, avgItemsInCluster, maxPredLabel]