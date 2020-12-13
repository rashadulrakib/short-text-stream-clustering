import statistics
from collections import Counter
import numpy as np
from compute_util import findCloseCluster_GramKey_lexical
from compute_util import findCloseCluster_GramKey_Semantic
from compute_util import computeSimBtnList
from txt_process_util import commonWordSims_clusterGroup
from txt_process_util import semanticSims
from sent_vecgenerator import generate_sent_vecs_toktextdata

def removeCommonTxtInds(dic_bitri_keys_selectedClusters_seenBatch):
  dic_txtId_to_grams={}
  new_dic_bitri_keys_selectedClusters_seenBatch={}
  #not_clustered_inds_seen_batch=[]
  #------sort dic_bitri_keys_selectedClusters_seenBatch by no of values in key.
  #------remove duplicate txtInd from large clusters
  temp_totalTexts=[]
  for key in dic_bitri_keys_selectedClusters_seenBatch.keys():
    txtInds= dic_bitri_keys_selectedClusters_seenBatch[key]
    txtInds=set(txtInds)	
    temp_totalTexts	+=txtInds
    for txtInd in txtInds:
      dic_txtId_to_grams.setdefault(txtInd, []).append(key)

  temp_totalTexts=list(set(temp_totalTexts))	  
  print("temp_totalTexts=", len(temp_totalTexts))	  
	  
  sortedTxtIndsByGrams=sorted(dic_txtId_to_grams, key = lambda key: len(dic_txtId_to_grams[key]))
  
  list_distSize=[]  
  for txtInd in sortedTxtIndsByGrams:
    clsDistributions=dic_txtId_to_grams[txtInd]	
    size=len(clsDistributions)
    #if size==1:
    #  continue	
    list_distSize.append(size)

  mean_distSize=0
  if len(list_distSize)>=1:
    mean_distSize=statistics.mean(list_distSize)
  std_distSize=mean_distSize
  if len(list_distSize)>=2:  
    std_distSize=statistics.stdev(list_distSize)

  print("mean_distSize", mean_distSize, std_distSize)  
	
  
  prev_grams_key={}
  for txtInd in sortedTxtIndsByGrams: 
    clsDistributions=dic_txtId_to_grams[txtInd]
    new_clsDistributions=[]
    if len(clsDistributions)>1:
      #clsDistributions=['brady peyton tom', 'bronco patriot rally'], 
	  #remove 'bronco patriot rally' from array if this appaer before. #not perfect yet
      
      for gram in clsDistributions:
        if gram not in prev_grams_key:
          new_clsDistributions.append(gram)          		

    if len(new_clsDistributions)>=1:
      clsDistributions=new_clsDistributions
	
    '''if len(clsDistributions)==8:	  
      prev_grams_key[clsDistributions[0]]=True
      prev_grams_key[clsDistributions[1]]=True
      prev_grams_key[clsDistributions[2]]=True	  
      prev_grams_key[clsDistributions[3]]=True	  	  
      prev_grams_key[clsDistributions[4]]=True	  	  	  
      prev_grams_key[clsDistributions[5]]=True
      prev_grams_key[clsDistributions[6]]=True 	  
      prev_grams_key[clsDistributions[7]]=True 	  
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)
    elif len(clsDistributions)==7:	  
      prev_grams_key[clsDistributions[0]]=True
      prev_grams_key[clsDistributions[1]]=True
      prev_grams_key[clsDistributions[2]]=True	  
      prev_grams_key[clsDistributions[3]]=True	  	  
      prev_grams_key[clsDistributions[4]]=True	  	  	  
      prev_grams_key[clsDistributions[5]]=True	  	  	  	  
      prev_grams_key[clsDistributions[6]]=True	  
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)
    if len(clsDistributions)==6:	  
      prev_grams_key[clsDistributions[0]]=True
      prev_grams_key[clsDistributions[1]]=True
      prev_grams_key[clsDistributions[2]]=True	  
      prev_grams_key[clsDistributions[3]]=True	  	  
      prev_grams_key[clsDistributions[4]]=True	  	  	  
      prev_grams_key[clsDistributions[5]]=True	  	  	  	  
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)	
    elif len(clsDistributions)==5:	  
      prev_grams_key[clsDistributions[0]]=True
      prev_grams_key[clsDistributions[1]]=True
      prev_grams_key[clsDistributions[2]]=True	  
      prev_grams_key[clsDistributions[3]]=True	  	  
      prev_grams_key[clsDistributions[4]]=True	  	  	  
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd) 	
    elif len(clsDistributions)==4:	  
      prev_grams_key[clsDistributions[0]]=True
      prev_grams_key[clsDistributions[1]]=True
      prev_grams_key[clsDistributions[2]]=True	  
      prev_grams_key[clsDistributions[3]]=True	  	  
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)	  	  
    elif len(clsDistributions)==3:	  
      prev_grams_key[clsDistributions[0]]=True
      prev_grams_key[clsDistributions[1]]=True
      prev_grams_key[clsDistributions[2]]=True	  
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)  	  
    elif len(clsDistributions)==2:	  
      prev_grams_key[clsDistributions[0]]=True
      prev_grams_key[clsDistributions[1]]=True	  
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)	
    el'''

    #if len(clsDistributions)>=1 and len(clsDistributions)<=mean_distSize+2.5*std_distSize: #stable
    if len(clsDistributions)>=1:	
      for clsDistribution in clsDistributions:
        prev_grams_key[clsDistribution]=True
      new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)        		
        	  
	
    #if len(clsDistributions)==1:
    #  prev_grams_key[clsDistributions[0]]=True
    #  new_dic_bitri_keys_selectedClusters_seenBatch.setdefault(clsDistributions[0], []).append(txtInd)
    

  '''std_size=0
  min_size=0
  list_sizes=[]
  
  for key, items in new_dic_bitri_keys_selectedClusters_seenBatch.items():
    size=len(items)
    list_sizes.append(size)
   
  min_size=0 
  if len(list_sizes)>=1:	
    min_size=statistics.mean(list_sizes)
  std_size=min_size
  if len(list_sizes)>=2:  
    std_size=statistics.stdev(list_sizes)	
  
  print("removeCommonTxtInds#####", min_size, std_size)

  	
    	
   

  #temp_dic={} 
  #items_threshold= int(abs(min_size-std_size)/1)
  #if items_threshold<10:
  #  items_threshold=10
  #items_threshold=max(10, items_threshold)	
  temp_dic={} 
  items_threshold= int(abs(min_size-std_size)/2)
  #if items_threshold<3:
  #  items_threshold=3
  print("items_threshold=", items_threshold)     
  for key, items in new_dic_bitri_keys_selectedClusters_seenBatch.items():
    #if len(items)<=int(abs(min_size-std_size)/2): #stable
    if len(items)<=items_threshold:	
      continue
    temp_dic[key]=items
  	
  
  new_dic_bitri_keys_selectedClusters_seenBatch=temp_dic'''  
     
  return new_dic_bitri_keys_selectedClusters_seenBatch#, not_clustered_inds_seen_batch] #temporary	'''  

def mergeByCommonTextInds(dic_bitri_keys_selectedClusters_seenBatch, simThreshold=0.8):
  new_dic_bitri_keys_selectedClusters_seenBatch={}
  keys_list=list(dic_bitri_keys_selectedClusters_seenBatch.keys())
  
  dic_usedKey_to_maxSim={}
  
  for i in range(0, len(keys_list)):
       
    max_sim=0
    max_j=-1	
    if keys_list[i] in dic_usedKey_to_maxSim:
      continue	
    for j in range(0, len(keys_list)):
 	
      if i==j or keys_list[j] in dic_usedKey_to_maxSim:
        continue	
      txtIndsi=	dic_bitri_keys_selectedClusters_seenBatch[keys_list[i]]
      txtIndsj=	dic_bitri_keys_selectedClusters_seenBatch[keys_list[j]]
	 
      #--------if txtIndsi is the subset of txtIndsj
      z = set(txtIndsi).intersection(set(txtIndsj))
      if len(z)==len(set(txtIndsi)) or len(z)==len(set(txtIndsj)):
        dic_usedKey_to_maxSim[keys_list[i]]=max_sim
        dic_usedKey_to_maxSim[keys_list[j]]=max_sim
        target_key = keys_list[i] # if len(set(txtIndsi))>len(set(txtIndsj)) else keys_list[j]		
        new_dic_bitri_keys_selectedClusters_seenBatch[target_key]=list(set(txtIndsi+txtIndsj))
        continue			  
      #--------end if txtIndsi is the subset of txtIndsj	  
	  
	  
      sim=computeSimBtnList(txtIndsi, txtIndsj)
      if sim >= simThreshold and sim> max_sim:
        max_sim=sim
        max_j=j
		
    if max_j>-1 and keys_list[max_j] not in dic_usedKey_to_maxSim:	
      dic_usedKey_to_maxSim[keys_list[i]]=max_sim
      dic_usedKey_to_maxSim[keys_list[max_j]]=max_sim
      txtIndsi=	dic_bitri_keys_selectedClusters_seenBatch[keys_list[i]]
      txtIndsj=	dic_bitri_keys_selectedClusters_seenBatch[keys_list[max_j]]	  
      #select key (i,j) based on the size of (txtinds)
      #print("mergeByCommonTextInds", keys_list[i], ",", keys_list[max_j])
      target_key = keys_list[i] #if len(set(txtIndsi))>len(set(txtIndsj)) else keys_list[max_j]	  
      new_dic_bitri_keys_selectedClusters_seenBatch[target_key]=list(set(txtIndsi+txtIndsj))	  
		
      	  
  return new_dic_bitri_keys_selectedClusters_seenBatch
  
def mergeByCommonWords(dic_biGram_to_textInds, dic_triGram_to_textInds, dic_bitri_keys_selectedClusters_seenBatch, minCommomGram, t_minSize, t_maxSize, b_minSize, b_maxSize):
  new_dic_bitri_keys_selectedClusters_seenBatch={}
  
  keys_list=dic_bitri_keys_selectedClusters_seenBatch.keys()
  for key, txtInds in dic_triGram_to_textInds.items():
    #try the key to merge with big dic  
	#if can not merge with big dic, then add the key to the big dic
    txtInds=list(set(txtInds))
     	
    gram_clusterSize=len(txtInds)
    if gram_clusterSize<=t_minSize:
      continue	
    #keys_list=dic_bitri_keys_selectedClusters_seenBatch.keys()
    word_arr=key.split(' ')	
    closeKey_Lexical=findCloseCluster_GramKey_lexical(keys_list,word_arr,2)
    if closeKey_Lexical==None:
      closeKey_Lexical=key	
      dic_bitri_keys_selectedClusters_seenBatch[key]=list(set(txtInds))
    else:
      dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical]=list(set(dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical]+txtInds))
	  
    if len(dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical])<=t_minSize:
      del dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical]
      continue	  
	  
    #print(closeKey_Lexical, dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical])	  
  
  
  keys_list=dic_bitri_keys_selectedClusters_seenBatch.keys()
  for key, txtInds in dic_biGram_to_textInds.items():
    #try the key to merge with big dic  
	#if can not merge with big dic, then add the key to the big dic
    txtInds=list(set(txtInds))

    gram_clusterSize=len(txtInds)
    if gram_clusterSize<=b_minSize:
      continue	
    #keys_list=dic_bitri_keys_selectedClusters_seenBatch.keys()
    word_arr=key.split(' ')	
    closeKey_Lexical=findCloseCluster_GramKey_lexical(keys_list,word_arr,2)
    if closeKey_Lexical==None:	
      closeKey_Lexical=key	
      dic_bitri_keys_selectedClusters_seenBatch[key]=list(set(txtInds))
    else:
      dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical]=list(set(dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical]+txtInds))
	  
    if len(dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical])<=b_minSize:
      del dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical]
      continue

    #print(closeKey_Lexical, dic_bitri_keys_selectedClusters_seenBatch[closeKey_Lexical])	  
	  
  new_dic_bitri_keys_selectedClusters_seenBatch=dic_bitri_keys_selectedClusters_seenBatch	  
  
  return new_dic_bitri_keys_selectedClusters_seenBatch

def assignToClusterSimDistribution(not_clustered_inds_batch, dic_bitri_keys_selectedClusters_seenBatch, seen_list_pred_true_words_index, wordVectorsDic):
  
  new_not_clustered_inds_batch=[]
  
  ##follow Mstream
  dic_ClusterGroupsDetail={}  
  dic_ClusterWords={}
  dic_ClusterTextWords={}
  dic_ClusterVecs={}
  
  for key, txtInds in dic_bitri_keys_selectedClusters_seenBatch.items():
    list_pred_true_words_index=[]
    cluster_words=[]
    txtWords=[]
    vec=np.zeros(shape=[300])
    for txtInd in txtInds:
      pred= seen_list_pred_true_words_index[txtInd][0]
      true= seen_list_pred_true_words_index[txtInd][1]
      words= seen_list_pred_true_words_index[txtInd][2]
      index= seen_list_pred_true_words_index[txtInd][3]	
      list_pred_true_words_index.append([pred, true, words, index])
      cluster_words.extend(words)
      txtWords.append(words)
      sent_vec=generate_sent_vecs_toktextdata([words], wordVectorsDic, 300)[0]
      sent_vec=np.asarray(sent_vec)
      vec=np.add(vec, sent_vec)	  
	  
    dic_ClusterGroupsDetail[key]=list_pred_true_words_index
    dic_ClusterWords[key]=[Counter(cluster_words), len(cluster_words)]	
    dic_ClusterTextWords[key]=txtWords	
    dic_ClusterVecs[key]=vec # np.true_divide(vec, len(txtInds)+1)
    #print("dic_ClusterVecs[key]", dic_ClusterVecs[key])	
  
  
  ##end follow Mstream
    
  
  ####our logic starts
  keys_list=list(dic_bitri_keys_selectedClusters_seenBatch.keys())
  
  #new_clusters={}

  for item in not_clustered_inds_batch:
    word_arr=item[2]
    global_index=item[3]
    true=item[1]  

    dic_lex_Sim_CommonWords, maxPredLabel_lex, maxSim_lex, maxCommon_lex, minSim_lex=commonWordSims_clusterGroup(word_arr, dic_ClusterWords)
		
    text_Vec=generate_sent_vecs_toktextdata([word_arr], wordVectorsDic, 300)[0]		
    dic_semanticSims, maxPredLabel_Semantic, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, dic_ClusterVecs)		
        
    if maxCommon_lex>0 and str(maxPredLabel_lex)==str(maxPredLabel_Semantic):
      new_pred=str(maxPredLabel_lex)
      new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])
      '''if len(new_pred.split(' '))==1 and new_pred.isnumeric()==True:
        #print("new_pred.isnumeric=", new_pred)
        dic_ClusterVecs[new_pred]= np.add(dic_ClusterVecs[new_pred], np.asarray(text_Vec))
        count_dic=dic_ClusterWords[new_pred][0]
        totalwords_dic=dic_ClusterWords[new_pred][1] 
        dic_ClusterWords[new_pred]=[count_dic+Counter(word_arr), totalwords_dic+len(word_arr)]'''	  
    '''else:
      new_key=str(len(dic_ClusterVecs)+10)
      new_pred=new_key
      new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])
      dic_ClusterVecs[new_pred]=np.asarray(text_Vec)
      dic_ClusterWords[new_pred]=[Counter(word_arr), len(word_arr)]	  
      #print("new_pred=", new_pred)'''	  
	  
 	
    '''closeKey_Lexical=findCloseCluster_GramKey_lexical(keys_list,word_arr,1)
    closeKey_Semantic, max_Semantic_sim_gram=findCloseCluster_GramKey_Semantic(keys_list,word_arr,0, wordVectorsDic, False)
    if closeKey_Lexical==closeKey_Semantic:
      new_pred=str(closeKey_Lexical)
      new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])
    else:
      closeKey_Lexical=findCloseCluster_GramKey_lexical(keys_list,word_arr,2)
      if closeKey_Lexical != None:
        new_pred=str(closeKey_Lexical)
        new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])
      #elif max_Semantic_sim_gram>=0.8:
      #  new_pred=str(closeKey_Lexical)
      #  new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])	  
      else:
        dic_lex_Sim_CommonWords, maxPredLabel_lex, maxSim_lex, maxCommon_lex, minSim_lex=commonWordSims_clusterGroup(word_arr, dic_ClusterWords)
		
        text_Vec=generate_sent_vecs_toktextdata([word_arr], wordVectorsDic, 300)[0]		
        dic_semanticSims, maxPredLabel_Semantic, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, dic_ClusterVecs)		
        
        if maxCommon_lex>0 and str(maxPredLabel_lex)==str(maxPredLabel_Semantic):
          new_pred=str(maxPredLabel_lex)
          new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])
        #else: #assign to a new cluster
        #  new_key=str(len(new_clusters)	+ len(keys_list)+10)
        #  new_pred=new_key
        #  new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])		  
        #  new_clusters.setdefault(new_key,[]).append([new_pred, true, word_arr, global_index])		
          		
          		
        #elif maxCommon_lex>=6:
        #  new_pred=str(maxPredLabel_lex)
        #  new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])
        #elif maxSim_Semantic>=0.5:
        #  new_pred=str(maxPredLabel_Semantic)
        #  new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])		  
        #  maxPredLabel=int(str(maxPredLabel))+1	
        #  pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
        #  new_outs.append(pred_true_text_ind_prevPred)	
    #elif closeKey_Lexical != None:
    #  new_pred=str(closeKey_Lexical)
    #  new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])'''
    '''else:
      if closeKey_Semantic !=None:
        new_pred=str(closeKey_Semantic)
        new_not_clustered_inds_batch.append([new_pred, true, word_arr, global_index])'''  	
	
	

  return new_not_clustered_inds_batch

def assignToClusterBySimilarity(not_clustered_inds_seen_batch, seen_list_pred_true_words_index, dic_combined_keys_selectedClusters, wordVectorsDic):
  new_not_clustered_inds_seen_batch=[]
  dic_preds={}
  count=0
  keys_list=list(dic_combined_keys_selectedClusters.keys())
  for txtInd in not_clustered_inds_seen_batch:
    pred_true_words_index= seen_list_pred_true_words_index[txtInd]
    word_arr=pred_true_words_index[2]
    closeKey_Lexical=findCloseCluster_GramKey_lexical(keys_list,word_arr,2)
    closeKey_Semantic=findCloseCluster_GramKey_Semantic(keys_list,word_arr,1, wordVectorsDic)
    seen_item=seen_list_pred_true_words_index[txtInd]	
    if closeKey_Lexical != None:
      dic_preds.setdefault(closeKey_Lexical,[]).append(txtInd)
      count+=1	
	  
      new_not_clustered_inds_seen_batch.append([closeKey_Lexical ,seen_item[1], seen_item[2], seen_item[3]])  	  
    else:
      if closeKey_Semantic !=None:
        dic_preds.setdefault(closeKey_Semantic,[]).append(txtInd)
        count+=1
		
        new_not_clustered_inds_seen_batch.append([closeKey_Semantic ,seen_item[1], seen_item[2], seen_item[3]])		
        	  
	
        	
  total_dic_items=sum([len(set(dic_combined_keys_selectedClusters[x])) for x in dic_combined_keys_selectedClusters if isinstance(dic_combined_keys_selectedClusters[x], list)])
  print("batch-eval: asign count "+ str(count)+"," +str(len(not_clustered_inds_seen_batch))+", total_dic_items,"+str(total_dic_items))
  #print("batch-eval: total_dic_items", total_dic_items)
  return [dic_preds, new_not_clustered_inds_seen_batch]
  

def filterClusters(dictri_keys_selectedClusters_currentBatch, dicbi_keys_selectedClusters_currentBatch, sub_list_pred_true_words_index, seen_list_pred_true_words_index):
  new_dictri_keys_selectedClusters_currentBatch={}
  new_dicbi_keys_selectedClusters_currentBatch={}
  new_dic_combined_keys_selectedClusters={}
  new_not_clustered_inds_currentBatch=[]
  dic_txtIds={}
  new_sub_list_pred_true_words_index=[]
  
  for key, txtInds in dictri_keys_selectedClusters_currentBatch.items():
    txtInds=list(set(txtInds))  
    if len(txtInds)==0:
      continue	
    if len(txtInds)==1:
      new_not_clustered_inds_currentBatch.extend(txtInds)
      continue	  
    new_dictri_keys_selectedClusters_currentBatch[key]=txtInds
    new_dic_combined_keys_selectedClusters[key]=txtInds	
    for txtInd in txtInds:
      dic_txtIds[txtInd]=key
	  
	  
      #assign  label to text	  

  for key, txtInds in dicbi_keys_selectedClusters_currentBatch.items():
    txtInds=list(set(txtInds))  
    if len(txtInds)==0:
      continue	
    if len(txtInds)==1:
      new_not_clustered_inds_currentBatch.extend(txtInds)
      continue	  
    new_dicbi_keys_selectedClusters_currentBatch[key]=txtInds
    new_dic_combined_keys_selectedClusters[key]=txtInds	
    for txtInd in txtInds:
      dic_txtIds[txtInd]=key

  for pred_true_words_index	in sub_list_pred_true_words_index: 
    index=pred_true_words_index[3]
    if index in dic_txtIds:
      cluster_key=dic_txtIds[index]	
      seen_item=seen_list_pred_true_words_index[index]
      new_sub_list_pred_true_words_index.append([cluster_key, seen_item[1], seen_item[2], seen_item[3]]) 
	  
      continue 	
    new_not_clustered_inds_currentBatch.append(index)

  new_not_clustered_inds_currentBatch=list(set(new_not_clustered_inds_currentBatch))
  	
  print("filter", "batch-eval:not clustered", len(new_not_clustered_inds_currentBatch), "total texts clustered", len(dic_txtIds), "clusters",len(new_dic_combined_keys_selectedClusters))
    	   
  return [new_dictri_keys_selectedClusters_currentBatch, new_dicbi_keys_selectedClusters_currentBatch, new_not_clustered_inds_currentBatch, new_dic_combined_keys_selectedClusters, new_sub_list_pred_true_words_index]

def mergeWithPrevBatch(dic_keys_selectedClusters, dic_keys_selectedClusters_prevBatch):
  if len(dic_keys_selectedClusters_prevBatch)==0:
    return dic_keys_selectedClusters
  new_dic_keys_selectedClusters={} 	
  #print("mergeWithPrevBatch", len(dic_keys_selectedClusters_prevBatch))  

  for key, txtInds in dic_keys_selectedClusters.items():
    new_dic_keys_selectedClusters[key]=txtInds
	
  for key, prev_txtInds in dic_keys_selectedClusters_prevBatch.items():
    if key in new_dic_keys_selectedClusters:
      new_txtInds = new_dic_keys_selectedClusters[key]
      combined_txtInds=list(set(prev_txtInds+new_txtInds))
      new_dic_keys_selectedClusters[key]=combined_txtInds
    else:
      new_dic_keys_selectedClusters[key]=prev_txtInds	
      	  
  	
  return new_dic_keys_selectedClusters	



def assignToMergedClusters(list_pred_true_words_index, not_clustered_inds,dicMerged_keys_selectedClusters, minMatch, seen_list_pred_true_words_index):
  #new_dicMerged_keys_selectedClusters={} #key =[txtInds w.r.t to sublist list_pred_true_words_index]
  new_not_clustered_inds=[]
  keys_list=list(dicMerged_keys_selectedClusters.keys())
  for txtInd in not_clustered_inds:
    '''item=list_pred_true_words_index[txtInd]'''
    item=seen_list_pred_true_words_index[txtInd]	
    word_arr=item[2]
    closeKey_Lexical=findCloseCluster_GramKey_lexical(keys_list, word_arr, minMatch)
    if closeKey_Lexical==None:
      new_not_clustered_inds.append(txtInd)
    else:
      #print("list before close key", dicMerged_keys_selectedClusters[closeKey_Lexical])  
      '''print("closeKey_Lexical", closeKey_Lexical+",",
	  list_pred_true_words_index[txtInd])'''	  
      print("closeKey_Lexical", closeKey_Lexical+",", seen_list_pred_true_words_index[txtInd])	  
      new_list=dicMerged_keys_selectedClusters[closeKey_Lexical]
      new_list.append(txtInd)      	  
      dicMerged_keys_selectedClusters[closeKey_Lexical]=new_list
      for lid in new_list:
        '''print("new_list,", list_pred_true_words_index[lid])'''
        print("new_list,", seen_list_pred_true_words_index[lid])	  		
      #print("list after close key", dicMerged_keys_selectedClusters[closeKey_Lexical])

  texts_clustered_sum=0
  max_group_sum=0
  for mergedKey, txtInds in dicMerged_keys_selectedClusters.items():
    #txtInds=list(set(txtInds))	
    #print("mergedKey->", mergedKey, txtInds)	
    texts_clustered_sum+=len(txtInds)
    true_label_list=[]
    for txtInd in txtInds:
      '''true_label_list.append(list_pred_true_words_index[txtInd][1])'''
      true_label_list.append(seen_list_pred_true_words_index[txtInd][1])	  
      	
    max_group_sum+=max(Counter(true_label_list).values())

  	
	
  print("\nnew_not_clustered_inds", len(new_not_clustered_inds), max_group_sum, texts_clustered_sum, max_group_sum/texts_clustered_sum, "old_not_clustered_inds", len(not_clustered_inds)) 	
  return [dicMerged_keys_selectedClusters, new_not_clustered_inds]

def extractNotClusteredItems(list_pred_true_words_index, list_dic_keys_selectedClusters, seen_list_pred_true_words_index):
  not_clustered_inds=[]
  
  list_clustered_inds=[]
  for dic_keys_selectedClusters in list_dic_keys_selectedClusters:
    for gramkey, txtInds in dic_keys_selectedClusters.items():
      list_clustered_inds.extend(txtInds)
  list_clustered_inds=set(list_clustered_inds)
  
  '''for i in range(len(list_pred_true_words_index)):
    if i in list_clustered_inds:
      continue
    not_clustered_inds.append(i)
    print("not clustered", list_pred_true_words_index[i])'''
  for i in range(len(list_pred_true_words_index)):
    org_i=list_pred_true_words_index[i][3]  
    if org_i in list_clustered_inds:
      continue
    not_clustered_inds.append(org_i)
    print("not clustered", seen_list_pred_true_words_index[org_i])	

  print("len(not_clustered_inds)", len(not_clustered_inds))	
  	  
  return not_clustered_inds

def mergeGroups(dicgram_keys_selectedClusters, matchCount, list_pred_true_words_index, seen_list_pred_true_words_index):
  max_group_sum=0
  texts_clustered_sum=0
  new_dic_keys_selectedClusters={}
  dic_used_key={}  
  keys=list(dicgram_keys_selectedClusters.keys())
  #print("keys", keys) 
  
  for i in range(len(keys)):
    #print("list(dic_used_key.keys())", list(dic_used_key.keys()))
    if keys[i] in list(dic_used_key.keys()):	
      continue	
    list_used_keys=[] 	  
    dic_used_key[keys[i]]=1	
    for j in range(i+1, len(keys)):
      if keys[j] in list(dic_used_key.keys()):	
        continue	
      seti=set(keys[i].split(' '))
      setj=set(keys[j].split(' '))
      commonWords=seti.intersection(setj)
      if len(commonWords)==matchCount:
        dic_used_key[keys[j]]=1
        list_used_keys.append(keys[j])			
    		
    if len(list_used_keys)>0:
      
      list_used_keys.append(keys[i])
      #list_key_words=[] 	  
      list_key_values=[]	  
      for key in list_used_keys:
        #list_key_words.extend(key.split(' '))
        list_key_values.extend(dicgram_keys_selectedClusters[key])
      #list_key_words.sort()
      #list_key_words=set(list_key_words)	  
      #mergedKey=' '.join(list_key_words)
      #new_dic_keys_selectedClusters[mergedKey]=list_key_values	  
      new_dic_keys_selectedClusters[keys[i]]=list_key_values	  	  
      #print(mergedKey)	  
    else:
      new_dic_keys_selectedClusters[keys[i]]=dicgram_keys_selectedClusters[keys[i]]  
    
  for mergedKey, txtInds in new_dic_keys_selectedClusters.items():
    #txtInds=list(set(txtInds))	
    print("mergedKey->", mergedKey, txtInds)	
    texts_clustered_sum+=len(txtInds)
    true_label_list=[]
    for txtInd in txtInds:
      '''true_label_list.append(list_pred_true_words_index[txtInd][1])
      print(list_pred_true_words_index[txtInd])'''		
      true_label_list.append(seen_list_pred_true_words_index[txtInd][1])
      print(seen_list_pred_true_words_index[txtInd])	  
    max_group_sum+=max(Counter(true_label_list).values())	        		
      	  
  return [max_group_sum, texts_clustered_sum, new_dic_keys_selectedClusters]

'''def clusterByNgram(dic_gram_to_textInds, minSize, maxSize, dic_used_textIds, list_pred_true_words_index, seen_list_pred_true_words_index):
  print("-----gram calculation---------")  
  #find clusters based on gram
  dicgram_keys_selectedClusters={}
  dicgram_clusterSizes={}
  dicgram_keys_selectedNonEmptyClusters={}
  for gram, txtInds in dic_gram_to_textInds.items():
    size=len(dic_gram_to_textInds[gram])
    if size not in dicgram_clusterSizes: dicgram_clusterSizes[size]=0	
    dicgram_clusterSizes[size]+=1   
    if size>=minSize and size<=maxSize:
      dicgram_keys_selectedClusters[gram]=dic_gram_to_textInds[gram]
      #print(gram, dicgram_keys_selectedClusters[gram])
  #for key, size in dicgram_clusterSizes.items():
  #  print(key, size)
	
  dicgram_keys_selectedClusters={k: v for k, v in sorted(dicgram_keys_selectedClusters.items(), key=lambda item: item[1])}
  selectedClustersKeysList=list(dicgram_keys_selectedClusters.keys())
  texts_clustered_by_gram=0 
  max_group_sum_gram=0  
  for i in range(len(selectedClustersKeysList)):
    #remove previously used textIds
    i_txtIds=dicgram_keys_selectedClusters[selectedClustersKeysList[i]]	
    new_i_txtIds=[]
    for txtId in i_txtIds:
      if txtId not in dic_used_textIds:
        new_i_txtIds.append(txtId)	  
    dicgram_keys_selectedClusters[selectedClustersKeysList[i]]=new_i_txtIds	
    	
    common_txtIds_with_Others=[]	
    for j in range(i+1, len(selectedClustersKeysList)):  
      seti=set(dicgram_keys_selectedClusters[selectedClustersKeysList[i]])
      setj=set(dicgram_keys_selectedClusters[selectedClustersKeysList[j]])
      commonIds=list(seti.intersection(setj))
      common_txtIds_with_Others.extend(commonIds)
    	  
    filtered_txt_ids_i= []	
    for txt_id in dicgram_keys_selectedClusters[selectedClustersKeysList[i]]:
      if txt_id not in common_txtIds_with_Others:
        filtered_txt_ids_i.append(txt_id)
        dic_used_textIds[txt_id]=1	
    #print("\nfiltered_txt_ids_i", len(filtered_txt_ids_i), len(dicgram_keys_selectedClusters[selectedClustersKeysList[i]]), filtered_txt_ids_i, dicgram_keys_selectedClusters[selectedClustersKeysList[i]])
    dicgram_keys_selectedClusters[selectedClustersKeysList[i]]=filtered_txt_ids_i
    texts_clustered_by_gram+=len(filtered_txt_ids_i)	
 
    true_label_list=[] 
    print("selectedClustersKeysList[i]", selectedClustersKeysList[i])    	
    for txtInd in dicgram_keys_selectedClusters[selectedClustersKeysList[i]]:	
  
      print(seen_list_pred_true_words_index[txtInd])
      true_label_list.append(seen_list_pred_true_words_index[txtInd][1])	  	  
    if len(true_label_list)>0: 
      max_group_sum_gram+=max(Counter(true_label_list).values())
      dicgram_keys_selectedNonEmptyClusters[selectedClustersKeysList[i]]=dicgram_keys_selectedClusters[selectedClustersKeysList[i]]	  
	
  return [dic_used_textIds, max_group_sum_gram, texts_clustered_by_gram, dicgram_keys_selectedNonEmptyClusters]'''	
	
	
def clusterByNgram(dic_gram_to_textInds, minSize, maxSize, dic_used_textIds, list_pred_true_words_index, seen_list_pred_true_words_index):
  print("-----gram calculation---------")
  #merge two grams: using similarity between grams and sim= txtInds1 and txtInds2  
  #find clusters based on gram
  dicgram_keys_selectedClusters={}
  #dicgram_clusterSizes={}
  dicgram_keys_selectedNonEmptyClusters={}
  for gram, txtInds in dic_gram_to_textInds.items():
    size=len(dic_gram_to_textInds[gram])
    #if size not in dicgram_clusterSizes: dicgram_clusterSizes[size]=0	
    #dicgram_clusterSizes[size]+=1   
    if size>=minSize and size<=maxSize:
      dicgram_keys_selectedClusters[gram]=dic_gram_to_textInds[gram]
      #print(gram, dicgram_keys_selectedClusters[gram])
  #for key, size in dicgram_clusterSizes.items():
  #  print(key, size)
	
  dicgram_keys_selectedClusters={k: v for k, v in sorted(dicgram_keys_selectedClusters.items(), key=lambda item: item[1])}
  selectedClustersKeysList=list(dicgram_keys_selectedClusters.keys())
  texts_clustered_by_gram=0 
  max_group_sum_gram=0  
  for i in range(len(selectedClustersKeysList)):
    #remove previously used textIds
    i_txtIds=dicgram_keys_selectedClusters[selectedClustersKeysList[i]]	
    new_i_txtIds=[]
    for txtId in i_txtIds:
      if txtId not in dic_used_textIds:
        new_i_txtIds.append(txtId)	  
    dicgram_keys_selectedClusters[selectedClustersKeysList[i]]=new_i_txtIds	
    	
    common_txtIds_with_Others=[]	
    for j in range(i+1, len(selectedClustersKeysList)):  
      seti=set(dicgram_keys_selectedClusters[selectedClustersKeysList[i]])
      setj=set(dicgram_keys_selectedClusters[selectedClustersKeysList[j]])
      commonIds=list(seti.intersection(setj))
      common_txtIds_with_Others.extend(commonIds)
    	  
    filtered_txt_ids_i= []	
    for txt_id in dicgram_keys_selectedClusters[selectedClustersKeysList[i]]:
      if txt_id not in common_txtIds_with_Others:
        filtered_txt_ids_i.append(txt_id)
        dic_used_textIds[txt_id]=1	
    #print("\nfiltered_txt_ids_i", len(filtered_txt_ids_i), len(dicgram_keys_selectedClusters[selectedClustersKeysList[i]]), filtered_txt_ids_i, dicgram_keys_selectedClusters[selectedClustersKeysList[i]])
    dicgram_keys_selectedClusters[selectedClustersKeysList[i]]=filtered_txt_ids_i
    texts_clustered_by_gram+=len(filtered_txt_ids_i)	
 
    true_label_list=[] 
    print("selectedClustersKeysList[i]", selectedClustersKeysList[i])    	
    for txtInd in dicgram_keys_selectedClusters[selectedClustersKeysList[i]]:	
  
      print(seen_list_pred_true_words_index[txtInd])
      true_label_list.append(seen_list_pred_true_words_index[txtInd][1])	  	  
    if len(true_label_list)>0: 
      max_group_sum_gram+=max(Counter(true_label_list).values())
      dicgram_keys_selectedNonEmptyClusters[selectedClustersKeysList[i]]=dicgram_keys_selectedClusters[selectedClustersKeysList[i]]	  
	
  return [dic_used_textIds, max_group_sum_gram, texts_clustered_by_gram, dicgram_keys_selectedNonEmptyClusters]	

def populateNgramStatistics(dic_gram_to_textInds, minTxtIndsForNgram=1):
  ordered_keys_gram_to_textInds = sorted(dic_gram_to_textInds, key = lambda key: len(dic_gram_to_textInds[key]))
  txtIndsSize=[]
  for key in ordered_keys_gram_to_textInds:
    #print(key, dic_gram_to_textInds[key])
    if len(dic_gram_to_textInds[key])>minTxtIndsForNgram: txtIndsSize.append(len(dic_gram_to_textInds[key]))
  
  size_mean=0
  size_max=0
  size_min=0
  if len(txtIndsSize)>=1:
    size_mean=statistics.mean(txtIndsSize)
    size_max=max(txtIndsSize)  
    size_min=min(txtIndsSize)
  size_std=size_mean  
  if len(txtIndsSize)>=2:
    size_std=statistics.stdev(txtIndsSize)
   
  
  
  return [size_std, size_mean, size_max, size_min]