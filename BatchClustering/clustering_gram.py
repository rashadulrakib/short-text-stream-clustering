from txt_process_util import concatWordsSort
from txt_process_util import createTerm_Doc_matrix_dic

from collections import Counter
from clustering_gram_util import populateNgramStatistics
#from clustering_gram_util import clusterByNgram
#from clustering_gram_util import mergeGroups
#from clustering_gram_util import extractNotClusteredItems
#from clustering_gram_util import assignToMergedClusters
#from clustering_gram_util import mergeWithPrevBatch
from clustering_gram_util import mergeByCommonWords
from clustering_gram_util import mergeByCommonTextInds
from clustering_gram_util import removeCommonTxtInds

#from clustering_util import clusterByHdbscan
from evaluation import Evaluate
from evaluation_util import evaluateByGram

#import hdbscan


def transpose(l1, l2): 
  
    # iterate over list l1 to the length of an item  
    for i in range(len(l1[0])): 
        # print(i) 
        row =[] 
        for item in l1: 
            # appending to new list with values and index positions 
            # i contains index position and item contains values 
            row.append(item[i]) 
        l2.append(row) 
    return l2 

def cluster_gram_freq(list_pred_true_words_index, batchNo, dic_bitri_keys_selectedClusters_seenBatch={}, seen_list_pred_true_words_index=[]):
  dic_uniGram_to_textInds={}
  dic_biGram_to_textInds={}
  dic_triGram_to_textInds={}
  uni_std_csize_offset=50000
  bi_std_csize_offset=1000000
  tri_std_csize_offset=100000  
  
  for pred_true_words_index in list_pred_true_words_index:
    
    words=pred_true_words_index[2]
    org_ind_i=pred_true_words_index[3]	
    for j in range(len(words)):
      dic_uniGram_to_textInds.setdefault(words[j],[]).append(org_ind_i) 
      for k in range(j+1,len(words)):		  
        dic_biGram_to_textInds.setdefault(concatWordsSort([words[j], words[k]]),[]).append(org_ind_i)
        for l in range(k+1, len(words)):
          dic_triGram_to_textInds.setdefault(concatWordsSort([words[j], words[k], words[l]]),[]).append(org_ind_i)		

  uni_std,uni_mean,uni_max,uni_min=populateNgramStatistics(dic_uniGram_to_textInds, 1)
  bi_std,bi_mean,bi_max,bi_min=populateNgramStatistics(dic_biGram_to_textInds, 1)
  tri_std,tri_mean,tri_max,tri_min=populateNgramStatistics(dic_triGram_to_textInds, 1)
  
  #print(bi_std,bi_mean,bi_max,bi_min)
  #print(tri_std,tri_mean,tri_max,tri_min)
  
  dic_bitri_keys_selectedClusters_seenBatch=mergeByCommonWords(dic_biGram_to_textInds, dic_triGram_to_textInds, dic_bitri_keys_selectedClusters_seenBatch, 2, tri_mean+tri_std, tri_mean+tri_std+tri_std_csize_offset, bi_mean+bi_std, bi_mean+bi_std+bi_std_csize_offset)
  
  dic_bitri_keys_selectedClusters_seenBatch=mergeByCommonTextInds(dic_bitri_keys_selectedClusters_seenBatch, 0.8)
  
  dic_bitri_keys_selectedClusters_seenBatch=removeCommonTxtInds(dic_bitri_keys_selectedClusters_seenBatch)
  
  
  
  '''while True:
    currentClusters=len(dic_bitri_keys_selectedClusters_seenBatch)
    dic_bitri_keys_selectedClusters_seenBatch=mergeByCommonTextInds(dic_bitri_keys_selectedClusters_seenBatch, 0.8)
    prevClusters=len(dic_bitri_keys_selectedClusters_seenBatch)
    print("currentClusters", currentClusters, "prevClusters", prevClusters) 	
    predsSeen_list_pred_true_words_index=evaluateByGram(dic_bitri_keys_selectedClusters_seenBatch, seen_list_pred_true_words_index)
    Evaluate(predsSeen_list_pred_true_words_index)    
  	
    if abs(currentClusters-prevClusters)<=100: # currentClusters<=prevClusters:
      break
   '''	  
  
  
  '''#-----temp hdbscan---------------
  term_doc_matrix, dic_txt_index=createTerm_Doc_matrix_dic(dic_bitri_keys_selectedClusters_seenBatch)
  #l2 = [] 
  #l2=transpose(term_doc_matrix, l2) 
  clusterer = hdbscan.HDBSCAN()
  clusterer.fit(term_doc_matrix)
  print(clusterer.labels_)
  #print("before transpose", len(term_doc_matrix), "after", len(l2))
  print("hdbscan", len(clusterer.labels_), clusterer.labels_.max())
  
  #list_temp_eval=[]
  
  #for global_txtInd, matrixTxtIndex in dic_txt_index.items():
  #  matrixTxtIndex=matrixTxtIndex-len(l2)  
  #  item=seen_list_pred_true_words_index[matrixTxtIndex]
  #  predlabel=clusterer.labels_[matrixTxtIndex]	
  #  list_temp_eval.append([predlabel, item[1], item[2], item[3]]) 
  
  #Evaluate(list_temp_eval)  
  
  #---------end temp hdbscan---------------'''
  #####dic_bitri_keys_selectedClusters_seenBatch=clusterByHdbscan(dic_bitri_keys_selectedClusters_seenBatch, 10)
 
  #new_not_clustered_inds_seen_batch=not_clustered_inds_seen_batch
  new_dic_bitri_keys_selectedClusters_seenBatch=dic_bitri_keys_selectedClusters_seenBatch
  
  
  return new_dic_bitri_keys_selectedClusters_seenBatch
  '''dic_used_textIds={}
  dic_used_textIds, max_group_sum_tri, texts_clustered_by_tri, dictri_keys_selectedClusters=clusterByNgram(dic_triGram_to_textInds,tri_mean, tri_mean+tri_std+tri_std_csize_offset, dic_used_textIds, list_pred_true_words_index, seen_list_pred_true_words_index)
  
  dic_used_textIds, max_group_sum_bi, texts_clustered_by_bi, dicbi_keys_selectedClusters=clusterByNgram(dic_biGram_to_textInds, bi_mean+bi_std, bi_mean+bi_std+bi_std_csize_offset, dic_used_textIds, list_pred_true_words_index, seen_list_pred_true_words_index)
 
  print("###")
  print("tri", len(dic_triGram_to_textInds), "total cls#",len(dictri_keys_selectedClusters), tri_min, tri_max, tri_mean, tri_std, "texts_clustered_by_tri", texts_clustered_by_tri, "max_group_sum_tri", max_group_sum_tri, max_group_sum_tri/texts_clustered_by_tri)
  print("bi", len(dic_biGram_to_textInds), "total cls#",len(dicbi_keys_selectedClusters), bi_min, bi_max, bi_mean, bi_std, "texts_clustered_by_bi", texts_clustered_by_bi, "max_group_sum_bi", max_group_sum_bi, max_group_sum_bi/texts_clustered_by_bi) 
  
  
  print("mergekeys###")
  max_group_sum_tri, texts_clustered_by_tri, dictri_keys_selectedClusters=mergeGroups(dictri_keys_selectedClusters, 2, list_pred_true_words_index, seen_list_pred_true_words_index)
  max_group_sum_bi, texts_clustered_by_bi, dicbi_keys_selectedClusters=mergeGroups(dicbi_keys_selectedClusters, 1, list_pred_true_words_index, seen_list_pred_true_words_index)
  
  print("tri", len(dic_triGram_to_textInds), "merged total cls#",len(dictri_keys_selectedClusters), tri_min, tri_max, tri_mean, tri_std, "texts_clustered_by_tri", texts_clustered_by_tri, "max_group_sum_tri", max_group_sum_tri, max_group_sum_tri/texts_clustered_by_tri)
  print("bi", len(dic_biGram_to_textInds), "merged total cls#",len(dicbi_keys_selectedClusters), bi_min, bi_max, bi_mean, bi_std, "texts_clustered_by_bi", texts_clustered_by_bi, "max_group_sum_bi", max_group_sum_bi, max_group_sum_bi/texts_clustered_by_bi)
  
  #merge with prev batcches
  dictri_keys_selectedClusters=mergeWithPrevBatch(dictri_keys_selectedClusters, dictri_keys_selectedClusters_prevBatch)
  dicbi_keys_selectedClusters=mergeWithPrevBatch(dicbi_keys_selectedClusters, dicbi_keys_selectedClusters_prevBatch)  
  #end merge with prev batcches
  
  print("###txtIds not in merged clusters###")
  not_clustered_inds=extractNotClusteredItems(list_pred_true_words_index, [dictri_keys_selectedClusters, dicbi_keys_selectedClusters], seen_list_pred_true_words_index)
  
  print("###assign non clustered to merged clusters###")
  new_dicTriMerged_keys_selectedClusters, not_clustered_inds_tri=assignToMergedClusters(list_pred_true_words_index, not_clustered_inds, dictri_keys_selectedClusters, 2, seen_list_pred_true_words_index)
  #new_dicBiMerged_keys_selectedClusters, not_clustered_inds_bi=assignToMergedClusters(list_pred_true_words_index, not_clustered_inds_tri, dicbi_keys_selectedClusters, 2)
  
  return [new_dicTriMerged_keys_selectedClusters, dicbi_keys_selectedClusters, not_clustered_inds_tri]
  #return [dictri_keys_selectedClusters_batch, ]'''
    
  
    
  
    
  
  
  