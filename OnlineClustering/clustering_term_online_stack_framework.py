from CPost import CPost
from CPostProcessed import CPostProcessed
from CFVector import CFVector

from evaluation import Evaluate
from collections import Counter
import statistics
from random import randint
from datetime import datetime
from sent_vecgenerator import generate_sent_vecs_toktextdata
from txt_process_util import construct_biterms
from txt_process_util import generateGramsConsucetive
from txt_process_util import semanticSims
from compute_util import computeTextClusterSimilarity_framework
from operator import add
import math

# need to delete high entropy words in targetClusterIds when compute similarity between ti to the targetClusterIds

# minGSize=1
# maxGSize=2

embedDim = 50

max_cposts = 5000

ignoreMinusOne = False
isSemantic = False
microDivide = 1000000
DeleteInterval = 500


def removeHighEntropyFtrs(c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq,
                          dic_biterm__allClusterFreq):
    ####c_bitermsFreqs[clusterId][biterm]=0

    dic_biterm__entropy = {}

    for biterm, dic_clusterId__Freq in dic_biterm__clusterId_Freq.items():
        entropy = 0
        totalFreq = dic_biterm__allClusterFreq[biterm]
        dic_biterm__entropy[biterm] = entropy
        if totalFreq <= 0:
            continue
        for clusterId, Freq in dic_clusterId__Freq.items():
            entropy = entropy + -1 * (Freq / totalFreq) * math.log(Freq / totalFreq)

        dic_biterm__entropy[biterm] = entropy

    allentropies = list(dic_biterm__entropy.values())

    mean_entropy = 0
    std_entropy = 0

    if len(allentropies) > 2:
        mean_entropy = statistics.mean(allentropies)
        std_entropy = statistics.stdev(allentropies)

    listTargetClusters = []

    for biterm, entropy in dic_biterm__entropy.items():
        if entropy > mean_entropy + std_entropy:
            del dic_biterm__clusterId_Freq[biterm]
            del dic_biterm__allClusterFreq[biterm]
            # print(biterm, entropy)

            for clusterId, dic_biterms__freq in c_bitermsFreqs.items():
                if biterm in dic_biterms__freq:
                    clusterBitermFreq = c_bitermsFreqs[clusterId][biterm]
                    del c_bitermsFreqs[clusterId][biterm]
                    c_totalBiterms[clusterId] -= clusterBitermFreq

    listTargetClusters = list(c_bitermsFreqs.keys())
    for clusterId in listTargetClusters:
        if c_totalBiterms[clusterId] <= 0:
            del c_totalBiterms[clusterId]
            del c_bitermsFreqs[clusterId]
            del c_txtIds[clusterId]

    return [c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq]


def computeTextSimCommonWord_WordDic(words_i, words_j, txt_i_len, txt_j_len):
    text_sim = 0
    commonCount = 0

    len_i = len(words_i)
    len_j = len(words_j)

    if len_i > len_j:
        temp = words_i
        words_i = words_j
        words_j = temp

    for word_i, i_count in words_i.items():
        if word_i in words_j.keys():
            commonCount = commonCount + i_count + words_j[word_i]

    if txt_i_len > 0 and txt_j_len > 0:
        text_sim = commonCount / (txt_i_len + txt_j_len)

    return [text_sim, commonCount]


def computeTextSimCommonWord_WordArr(txt_i_wordArr, txt_j_wordArr):  # not used

    txt_i_len = len(txt_i_wordArr)
    txt_j_len = len(txt_j_wordArr)

    words_i = Counter(txt_i_wordArr)  # assume words_i small
    words_j = Counter(txt_j_wordArr)

    text_sim, commonCount = computeTextSimCommonWord_WordDic(words_i, words_j, txt_i_len, txt_j_len)

    return [text_sim, commonCount]


def removeTargetMultiClusterBiTerms(c_bitermsFreqs, c_totalBiterms, c_txtIds, targetClusterIds, txtBitermsFreqs,
                                    dic_biterm__clusterIds):
    dic_biterm__TargetClustersTotalFreq = {}
    dic_biterm__TargetClustersEntropy = {}
    for clusterId in targetClusterIds:
        if clusterId not in c_bitermsFreqs:
            continue
        dic_bitermsFreqs = c_bitermsFreqs[clusterId]

        for biterm, freq in dic_bitermsFreqs.items():
            if biterm not in dic_biterm__TargetClustersTotalFreq:
                dic_biterm__TargetClustersTotalFreq[biterm] = 0
            dic_biterm__TargetClustersTotalFreq[biterm] += freq

    for clusterId in targetClusterIds:
        if clusterId not in c_bitermsFreqs:
            continue
        dic_bitermsFreqs = c_bitermsFreqs[clusterId]

        for biterm, freq in dic_bitermsFreqs.items():
            if biterm not in dic_biterm__TargetClustersEntropy:
                dic_biterm__TargetClustersEntropy[biterm] = 0
            dic_biterm__TargetClustersEntropy[biterm] -= freq / dic_biterm__TargetClustersTotalFreq[biterm] * math.log(
                freq / dic_biterm__TargetClustersTotalFreq[biterm])

    allentropies = list(dic_biterm__TargetClustersEntropy.values())
    # print(dic_biterm__TargetClustersEntropy)

    mean_entropy = 0
    std_entropy = 0

    if len(allentropies) > 2:
        mean_entropy = statistics.mean(allentropies)
        std_entropy = statistics.stdev(allentropies)

    for biterm, entropy in dic_biterm__TargetClustersEntropy.items():
        if entropy > mean_entropy + std_entropy:
            for clusterId in targetClusterIds:
                if clusterId not in c_bitermsFreqs or biterm not in c_bitermsFreqs[clusterId]:
                    continue
                clusterBitermFreq = c_bitermsFreqs[clusterId][biterm]
                del c_bitermsFreqs[clusterId][biterm]
                c_totalBiterms[clusterId] -= clusterBitermFreq

    for clusterId in targetClusterIds:
        if clusterId not in c_totalBiterms:
            continue
        if c_totalBiterms[clusterId] <= 0:
            del c_totalBiterms[clusterId]
            if clusterId in c_bitermsFreqs:
                del c_bitermsFreqs[clusterId]
            if clusterId in c_txtIds:
                del c_txtIds[clusterId]

    return [c_bitermsFreqs, c_totalBiterms, c_txtIds, txtBitermsFreqs]


def findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds):
    targetClusterIds = []

    for biterm, freq in txtBitermsFreqs.items():
        if biterm not in dic_biterm__clusterIds:
            continue
        targetClusterIds.extend(dic_biterm__clusterIds[biterm])

    targetClusterIds = set(targetClusterIds)
    return list(targetClusterIds)


def findCloseClusterByTargetClusters_framework(c_CFVector, oCPostProcessed, targetClusterIds, max_c_id,
                                               oCSimilarityFlgas):
    clusterId = -1
    max_sim = 0

    all_sims = []

    for clusId in targetClusterIds:
        if clusId not in c_CFVector:
            continue
        oCFVector = c_CFVector[clusId]

        totalSim = computeTextClusterSimilarity_framework(oCPostProcessed, oCFVector, oCSimilarityFlgas)

        if totalSim > max_sim:
            max_sim = totalSim
            clusterId = clusId

        all_sims.append(totalSim)

    mean_sim = 0
    std_sim = 0

    if len(all_sims) > 3:
        mean_sim = statistics.mean(all_sims)
        std_sim = statistics.stdev(all_sims)

    if clusterId == -1 or max_sim < mean_sim + std_sim:
        clusterId = max_c_id + 1

    # print('clusterId, max_sim', clusterId, max_sim, 'oCSimilarityFlgas.isTagSim', oCSimilarityFlgas.isTagSim,
    # 'targetClusterIds', targetClusterIds)
    return clusterId


def findCloseClusterByTargetClusters(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords,
                                     c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id,
                                     text_Vec, dic_biterm__clusterIds, targetClusterIds):
    clusterId_lex = -1
    clusterId_sem = -1
    clusterId = -1
    max_sim = 0
    max_sim_lex = 0

    dic_lexicalSims = {}

    for clusId in targetClusterIds:
        if clusId not in c_bitermsFreqs:
            continue
        # print('####targetClusterIds', len(targetClusterIds))
        clusBitermsFreqs = c_bitermsFreqs[clusId]
        txt_j_len = c_totalBiterms[clusId]

        text_sim, commonCount = computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len,
                                                                 txt_j_len)
        if text_sim > max_sim:
            max_sim = text_sim
            clusterId = clusId

        if text_sim > max_sim_lex:
            max_sim_lex = text_sim
            clusterId_lex = clusId
        dic_lexicalSims[clusId] = text_sim

    lex_sim_values = list(dic_lexicalSims.values())

    mean_lex_sim = 0
    std_lex_sim = 0

    if len(lex_sim_values) > 2:
        mean_lex_sim = statistics.mean(lex_sim_values)
        std_lex_sim = statistics.stdev(lex_sim_values)

    if clusterId_lex == -1:  # or clusterId_sem==-1:
        # clusterId=len(c_bitermsFreqs)+1
        clusterId = max_c_id + 1
        if isSemantic == True:
            dic_semanticSims, clusterId_sem, maxSim_Semantic, minSim_semantic = semanticSims(text_Vec, c_clusterVecs,
                                                                                             c_txtIds)
            sem_sim_values = list(dic_semanticSims.values())
            mean_sem_sim = 0
            std_sem_sim = 0
            if len(sem_sim_values) > 2:
                mean_sem_sim = statistics.mean(sem_sim_values)
                std_sem_sim = statistics.stdev(sem_sim_values)
                if maxSim_Semantic >= mean_sem_sim + std_sem_sim:  # and randint(0,1)==1: work
                    clusterId = clusterId_sem
                    # elif clusterId_lex!=clusterId_sem:
    #  clusterId=max_c_id+1
    # elif clusterId_lex==clusterId_sem:
    #  clusterId=clusterId_lex
    elif max_sim_lex >= mean_lex_sim + std_lex_sim:  # and randint(0,1)==1: work
        clusterId = clusterId_lex
    else:
        clusterId = max_c_id + 1  # clusterId_lex

    # print(text_Vec, clusterId_lex, clusterId_sem, clusterId)
    return clusterId
    # return clusterId_lex


def findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs,
                     txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec,
                     dic_biterm__clusterIds):
    clusterId_lex = -1
    clusterId_sem = -1
    clusterId = -1
    max_sim = 0
    max_sim_lex = 0

    dic_lexicalSims = {}

    targetClusterIds = findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds)

    # remove multi-cluster biterms from c_bitermsFreqs   using targetClusterIds; before computing similarity

    for clusId in targetClusterIds:
        if clusId not in c_bitermsFreqs:
            continue
        # print('####targetClusterIds', len(targetClusterIds))
        clusBitermsFreqs = c_bitermsFreqs[clusId]
        txt_j_len = c_totalBiterms[clusId]

        text_sim, commonCount = computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len,
                                                                 txt_j_len)
        if text_sim > max_sim:
            max_sim = text_sim
            clusterId = clusId

        if text_sim > max_sim_lex:
            max_sim_lex = text_sim
            clusterId_lex = clusId
        dic_lexicalSims[clusId] = text_sim

    '''for clusId, clusBitermsFreqs in c_bitermsFreqs.items():
    txt_j_len= c_totalBiterms[clusId]	

    text_sim, commonCount=computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len, txt_j_len)
    if text_sim> max_sim:
      max_sim=text_sim
      clusterId=clusId	  

    if text_sim> max_sim_lex:
      max_sim_lex=text_sim
      clusterId_lex=clusId
    dic_lexicalSims[clusId]=text_sim'''

    # dic_semanticSims, clusterId_sem, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, c_clusterVecs, c_txtIds)

    lex_sim_values = list(dic_lexicalSims.values())
    # sem_sim_values= list(dic_semanticSims.values())

    mean_lex_sim = 0
    std_lex_sim = 0

    if len(lex_sim_values) > 2:
        mean_lex_sim = statistics.mean(lex_sim_values)
        std_lex_sim = statistics.stdev(lex_sim_values)

        # mean_sem_sim=statistics.mean(sem_sim_values)
    # std_sem_sim=statistics.stdev(sem_sim_values)

    if clusterId_lex == -1:  # or clusterId_sem==-1:
        # clusterId=len(c_bitermsFreqs)+1
        clusterId = max_c_id + 1
    # elif clusterId_lex!=clusterId_sem:
    #  clusterId=max_c_id+1
    # elif clusterId_lex==clusterId_sem:
    #  clusterId=clusterId_lex
    # elif max_sim_lex>=mean_lex_sim+std_lex_sim: # and randint(0,1)==1: work
    #  clusterId=clusterId_lex
    else:
        clusterId = clusterId_lex

    # print(text_Vec, clusterId_lex, clusterId_sem, clusterId)
    return clusterId
    # return clusterId_lex


'''def findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec):
  clusterId_lex=-1
  clusterId_sem=-1
  clusterId=-1
  max_sim_lex=0
  max_sim_sem=0
  
  #if randint(0,1)==1:
  #  clusterId=max_c_id+1
    #print('random cluster', clusterId)	
  #  return clusterId	
     
  dic_lexicalSims={}
  for clusId, clusBitermsFreqs in c_bitermsFreqs.items():
    txt_j_len= c_totalBiterms[clusId]	    	
    text_sim, commonCount=computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len, txt_j_len)
    if text_sim> max_sim_lex:
      max_sim_lex=text_sim
      clusterId_lex=clusId
    dic_lexicalSims[clusId]=text_sim 
  if clusterId_lex==-1:
    return max_c_id+1 	
  lex_sim_values= list(dic_lexicalSims.values())
  mean_lex_sim=0
  std_lex_sim=0
  if len(lex_sim_values)>2:
    mean_lex_sim=statistics.mean(lex_sim_values)
    std_lex_sim=statistics.stdev(lex_sim_values) 
  if max_sim_lex>=mean_lex_sim+std_lex_sim:
    return clusterId_lex	
   
	
	
	
	
  
  dic_semanticSims, clusterId_sem, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, c_clusterVecs, c_txtIds)
  if clusterId_sem==-1:
    return max_c_id+1
  sem_sim_values= list(dic_semanticSims.values())
  mean_sem_sim=0
  std_sem_sim=0
  if len(sem_sim_values)>2:  
    mean_sem_sim=statistics.mean(sem_sim_values)
    std_sem_sim=statistics.stdev(sem_sim_values)
  if mean_sem_sim>=mean_sem_sim+std_sem_sim:
    return clusterId_sem 		
  	

  max_SR_sim=0
  SR_clusId=-1 
  dic_SRSims={}
  for clusId, lex_sim in dic_lexicalSims.items(): 
    sem_sim=dic_semanticSims[clusId]
    if max(lex_sim,sem_sim)<=0:
      continue	
    sim=min(lex_sim,sem_sim)/max(lex_sim,sem_sim)*(lex_sim+sem_sim)	
    if sim>max_SR_sim:
      max_SR_sim=sim
      SR_clusId=clusId
    dic_SRSims[clusId]=sim 	  
  if SR_clusId==-1:  
    return max_c_id+1 
  SR_sim_values= list(dic_SRSims.values())
  mean_SR_sim=0
  std_SR_sim=0
  if len(SR_sim_values)>2:  
    mean_SR_sim=statistics.mean(SR_sim_values)
    std_SR_sim=statistics.stdev(SR_sim_values)
  
  
  
  
  if max_SR_sim>=mean_SR_sim+std_SR_sim: # and randint(0,1)==1:
    return SR_clusId
  #elif mean_sem_sim>=mean_sem_sim+std_sem_sim:
  #  return clusterId_sem 	
  

   	
	
    	
    	
      
  #print(text_Vec, clusterId_lex, clusterId_sem, clusterId)
  #return clusterId
  return max_c_id+1'''


def populateClusterFeature_framework(c_CFVector, oCPostProcessed, dic_bitermTag__clusterIds,
                                     dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds, clusterId, id,
                                     oCSimilarityFlgas):
    if clusterId not in c_CFVector:
        oCFVector = CFVector(oCPostProcessed.txtBitermsFreqs_Tag, oCPostProcessed.bi_terms_len_Tag,
                             oCPostProcessed.txtBitermsFreqs_Title, oCPostProcessed.bi_terms_len_Title,
                             oCPostProcessed.txtBitermsFreqs_Body, oCPostProcessed.bi_terms_len_Body,
                             oCPostProcessed.text_VecTag, oCPostProcessed.text_VecTitle,
                             oCPostProcessed.text_VecBody)

        oCFVector.appendTextId(id)
        c_CFVector[clusterId] = oCFVector

        if oCSimilarityFlgas.isTagSim:
            for biterm, bitermFreq in oCPostProcessed.txtBitermsFreqs_Tag.items():
                dic_bitermTag__clusterIds.setdefault(biterm, []).append(clusterId)

        if oCSimilarityFlgas.isTitleSim:
            for biterm, bitermFreq in oCPostProcessed.txtBitermsFreqs_Title.items():
                dic_bitermTitle__clusterIds.setdefault(biterm, []).append(clusterId)

        if oCSimilarityFlgas.isBodySim:
            for biterm, bitermFreq in oCPostProcessed.txtBitermsFreqs_Body.items():
                dic_bitermBody__clusterIds.setdefault(biterm, []).append(clusterId)

    else:
        oCFVector = c_CFVector[clusterId]
        oCFVector.appendTextId(id)

        if oCSimilarityFlgas.isTagSim:
            oCFVector.txtBitermsFreqs_Tag = oCFVector.txtBitermsFreqs_Tag + oCPostProcessed.txtBitermsFreqs_Tag
            oCFVector.bi_terms_len_Tag = oCFVector.bi_terms_len_Tag + oCPostProcessed.bi_terms_len_Tag

            for biterm, bitermFreq in oCPostProcessed.txtBitermsFreqs_Tag.items():
                dic_bitermTag__clusterIds.setdefault(biterm, []).append(clusterId)

            if isSemantic:
                oCFVector.text_VecTag = list(map(add, oCPostProcessed.text_VecTag, oCFVector.text_VecTag))

        if oCSimilarityFlgas.isTitleSim:
            oCFVector.txtBitermsFreqs_Title = oCFVector.txtBitermsFreqs_Title + oCPostProcessed.txtBitermsFreqs_Title
            oCFVector.bi_terms_len_Title = oCFVector.bi_terms_len_Title + oCPostProcessed.bi_terms_len_Title

            for biterm, bitermFreq in oCPostProcessed.txtBitermsFreqs_Title.items():
                dic_bitermTitle__clusterIds.setdefault(biterm, []).append(clusterId)

            if isSemantic:
                oCFVector.text_VecTitle = list(map(add, oCPostProcessed.text_VecTitle, oCFVector.text_VecTitle))

        if oCSimilarityFlgas.isBodySim:
            oCFVector.txtBitermsFreqs_Body = oCFVector.txtBitermsFreqs_Body + oCPostProcessed.txtBitermsFreqs_Body
            oCFVector.bi_terms_len_Body = oCFVector.bi_terms_len_Body + oCPostProcessed.bi_terms_len_Body

            for biterm, bitermFreq in oCPostProcessed.txtBitermsFreqs_Body.items():
                dic_bitermBody__clusterIds.setdefault(biterm, []).append(clusterId)

            if isSemantic:
                oCFVector.text_VecBody = list(map(add, oCPostProcessed.text_VecBody, oCFVector.text_VecBody))

        c_CFVector[clusterId] = oCFVector

    return [c_CFVector, dic_bitermTag__clusterIds, dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds]


def populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs,
                           txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id, text_Vec,
                           dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq, dic_biterm__clusterIds):
    c_txtIds.setdefault(clusterId, []).append(current_txt_id)

    if clusterId not in c_bitermsFreqs:
        c_bitermsFreqs[clusterId] = {}
    if clusterId not in c_totalBiterms:
        c_totalBiterms[clusterId] = 0
    c_totalBiterms[clusterId] += bi_terms_len

    if clusterId not in c_wordsFreqs:
        c_wordsFreqs[clusterId] = {}
    if clusterId not in c_totalWords:
        c_totalWords[clusterId] = 0
    c_totalWords[clusterId] += words_len

    for biterm, bitermFreq in txtBitermsFreqs.items():
        if biterm not in c_bitermsFreqs[clusterId]:
            c_bitermsFreqs[clusterId][biterm] = 0
        c_bitermsFreqs[clusterId][biterm] += bitermFreq

        dic_biterm__clusterIds.setdefault(biterm, []).append(clusterId)

        '''if biterm not in dic_biterm__clusterId_Freq:
      dic_biterm__clusterId_Freq[biterm]={}
    if clusterId not in dic_biterm__clusterId_Freq[biterm]:
      dic_biterm__clusterId_Freq[biterm][clusterId]=0 
    dic_biterm__clusterId_Freq[biterm][clusterId]+=bitermFreq 
    if biterm not in dic_biterm__allClusterFreq:
      dic_biterm__allClusterFreq[biterm]=0
    dic_biterm__allClusterFreq[biterm]+=bitermFreq'''

    for word, wordFreq in txtWordsFreqs.items():
        if word not in c_wordsFreqs[clusterId]:
            c_wordsFreqs[clusterId][word] = 0
        c_wordsFreqs[clusterId][word] += wordFreq

    if isSemantic == True:
        if clusterId not in c_clusterVecs:
            c_clusterVecs[clusterId] = text_Vec
        else:
            c_clusterVecs[clusterId] = list(map(add, text_Vec, c_clusterVecs[clusterId]))

    return [c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs,
            dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq, dic_biterm__clusterIds]


def deleteOldClusters_framework(c_CFVector, c_itemsCount, dic_clus__id):
    print('deleteOldClusters_framework', DeleteInterval)

    list_c_sizes = []
    list_c_ids = []

    for c_id, itemsCount in c_itemsCount.items():
        list_c_sizes.append(itemsCount)
        list_c_ids.append(dic_clus__id[c_id])

    mean_c_size = 0
    std_c_size = 0
    if len(list_c_sizes) > 2:
        mean_c_size = statistics.mean(list_c_sizes)
        std_c_size = statistics.stdev(list_c_sizes)

    mean_c_id = 0
    std_c_id = 0
    if len(list_c_ids) > 2:
        mean_c_id = statistics.mean(list_c_ids)
        std_c_id = statistics.stdev(list_c_ids)

    print('mean_c_size', mean_c_size, 'std_c_size', std_c_size, 'mean_c_id', mean_c_id, 'std_c_id', std_c_id)

    list_del_cids = []

    itemsCounts = list(c_itemsCount.values())
    itemsCounts.sort(reverse=True)
    # print('deleteOldClusters_framework', itemsCounts)
    max_c_size = max(itemsCounts)
    # top_itemsCounts=itemsCounts[0:]

    for c_id, itemsCount in c_itemsCount.items():
        if itemsCount >= mean_c_size + std_c_size:  # int(max_c_size):  # itemsCount > mean_c_size + std_c_size:
            list_del_cids.append(c_id)

    list_del_cids = set(list_del_cids)
    print('#list_del_cids', len(list_del_cids))

    for c_id in list_del_cids:
        if c_id in c_CFVector:
            del c_CFVector[c_id]

        if c_id in c_itemsCount:
            del c_itemsCount[c_id]

    return [c_CFVector, c_itemsCount]


# split large clusters to small clusters

def cluster_biterm_framework(f, list_CPost, c_CFVector, max_c_id, dic_txtId__CPost, wordVectorsDic, dic_clus__id,
                             dic_bitermTag__clusterIds, dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds,
                             dic_ngram__txtIds, min_gram, max_gram, oCSimilarityFlgas, c_itemsCount):
    eval_pred_true_txt = []

    line_count = 0


    t11 = datetime.now()

    for oCPost in list_CPost:

        trueLabel = oCPost.trueLabel
        tagWords = oCPost.tagWords
        titleWords = oCPost.titleWords
        bodyWords = oCPost.bodyWords
        id = oCPost.id
        soPostId = oCPost.soPostId
        createtime = oCPost.createtime

        print('id', id, 'tagWords', tagWords, 'titleWords', titleWords, 'bodyWords', bodyWords)

        txtBitermsFreqs_Tag = None
        bi_terms_len_Tag = 0
        grams_Tag = None

        txtBitermsFreqs_Title = None
        bi_terms_len_Title = 0
        grams_Title = None

        txtBitermsFreqs_Body = None
        bi_terms_len_Body = 0
        grams_Body = None

        text_VecTag = None
        text_VecTitle = None
        text_VecBody = None
        targetClusterIds = []

        dic_txtId__CPost[id] = oCPost

        if oCSimilarityFlgas.isTagSim:
            bi_termsTag = construct_biterms(tagWords)

            grams_Tag = generateGramsConsucetive(tagWords, min_gram, max_gram)
            for gram in grams_Tag:
                if gram in dic_ngram__txtIds and len(set(dic_ngram__txtIds[gram])) > max_cposts:
                    continue
                dic_ngram__txtIds.setdefault(gram, []).append(id)
            txtBitermsFreqs_Tag = Counter(bi_termsTag)
            bi_terms_len_Tag = len(bi_termsTag)
            tCIds = findTargetClusters(txtBitermsFreqs_Tag, dic_bitermTag__clusterIds)
            # print('dic_bitermTag__clusterIds', dic_bitermTag__clusterIds, 'txtBitermsFreqs_Tag', txtBitermsFreqs_Tag)
            targetClusterIds.extend(tCIds)

            if isSemantic:
                X = generate_sent_vecs_toktextdata([tagWords], wordVectorsDic, embedDim)
                text_VecTag = X[0]

        if oCSimilarityFlgas.isTitleSim:
            bi_termsTitle = construct_biterms(titleWords)
            grams_Title = generateGramsConsucetive(titleWords, min_gram, max_gram)
            for gram in grams_Title:
                if gram in dic_ngram__txtIds and len(set(dic_ngram__txtIds[gram])) > max_cposts:
                    continue
                dic_ngram__txtIds.setdefault(gram, []).append(id)
            txtBitermsFreqs_Title = Counter(bi_termsTitle)
            bi_terms_len_Title = len(bi_termsTitle)
            tCIds = findTargetClusters(txtBitermsFreqs_Title, dic_bitermTitle__clusterIds)
            targetClusterIds.extend(tCIds)

            if isSemantic:
                X = generate_sent_vecs_toktextdata([titleWords], wordVectorsDic, embedDim)
                text_VecTitle = X[0]

        if oCSimilarityFlgas.isBodySim:
            bi_termsBody = construct_biterms(bodyWords)
            grams_Body = generateGramsConsucetive(bodyWords, min_gram, max_gram)
            for gram in grams_Body:
                if gram in dic_ngram__txtIds and len(set(dic_ngram__txtIds[gram])) > max_cposts:
                    continue
                dic_ngram__txtIds.setdefault(gram, []).append(id)
            txtBitermsFreqs_Body = Counter(bi_termsBody)
            bi_terms_len_Body = len(bi_termsBody)
            tCIds = findTargetClusters(txtBitermsFreqs_Body, dic_bitermBody__clusterIds)
            targetClusterIds.extend(tCIds)

            if isSemantic:
                X = generate_sent_vecs_toktextdata([bodyWords], wordVectorsDic, embedDim)
                text_VecBody = X[0]

        oCPostProcessed = CPostProcessed(txtBitermsFreqs_Tag, bi_terms_len_Tag, txtBitermsFreqs_Title,
                                         bi_terms_len_Title, txtBitermsFreqs_Body, bi_terms_len_Body, text_VecTag,
                                         text_VecTitle, text_VecBody)

        targetClusterIds = set(targetClusterIds)

        clusterId = findCloseClusterByTargetClusters_framework(c_CFVector, oCPostProcessed, targetClusterIds, max_c_id,
                                                               oCSimilarityFlgas)

        if ignoreMinusOne:
            if str(trueLabel) != '-1':
                f.write(
                    str(clusterId) + "	" + str(trueLabel) + "	" +
                    ' '.join(tagWords) + "	" + str(soPostId) + "\n")
        else:
            f.write(
                str(clusterId) + "	" + str(trueLabel) + "	" + ' '.join(tagWords) + "	" + str(soPostId) + "\n")

        eval_pred_true_txt.append([clusterId, trueLabel, tagWords])

        if clusterId not in c_itemsCount:
            c_itemsCount[clusterId] = 0
        c_itemsCount[clusterId] += 1

        max_c_id = max([max_c_id, clusterId, len(c_CFVector)])

        dic_clus__id[clusterId] = max_c_id
        # print('max_c_id, len(c_CFVector)', max_c_id, len(c_CFVector))

        c_CFVector, dic_bitermTag__clusterIds, dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds = populateClusterFeature_framework(
            c_CFVector, oCPostProcessed, dic_bitermTag__clusterIds, dic_bitermTitle__clusterIds,
            dic_bitermBody__clusterIds, clusterId, id, oCSimilarityFlgas)

        del oCPostProcessed
        del oCPost

        line_count += 1

        if line_count % DeleteInterval == 0:
            c_CFVector, c_itemsCount = deleteOldClusters_framework(c_CFVector, c_itemsCount, dic_clus__id)

        if line_count % 1000 == 0:
            # print('c_itemsCount', c_itemsCount)
            Evaluate(eval_pred_true_txt, ignoreMinusOne)

    return [c_CFVector, max_c_id, dic_txtId__CPost, dic_clus__id, dic_bitermTag__clusterIds,
            dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds, dic_ngram__txtIds, c_itemsCount]


def cluster_biterm(f, list_pred_true_words_index_postid_createtime, c_bitermsFreqs={}, c_totalBiterms={},
                   c_wordsFreqs={}, c_totalWords={}, c_txtIds={}, c_clusterVecs={}, txtId_txt={}, last_txtId=0,
                   max_c_id=0, wordVectorsDic={}, dic_clus__id={}, dic_biterm__clusterId_Freq={},
                   dic_biterm__allClusterFreq={}, dic_biterm__clusterIds={}, c_textItems={}, dic_ngram__textItems={},
                   min_gram=1, max_gram=2, isTagSim=True, isTitleSim=False, isBodySim=False):
    print("cluster_bigram")

    # current_txt_id=last_txtId

    eval_pred_true_txt = []

    line_count = 0

    t11 = datetime.now()

    for item in list_pred_true_words_index_postid_createtime:

        words = item[2]
        current_txt_id = int(item[3])
        postId = item[4]

        bi_terms = construct_biterms(words)
        grams = generateGramsConsucetive(words, min_gram, max_gram)
        # bi_terms=generateGramsConsucetive(words,minGSize, maxGSize)
        # print(words, bi_terms)

        for gram in grams:
            dic_ngram__textItems.setdefault(gram, []).append(item)

        line_count += 1

        txtBitermsFreqs = Counter(bi_terms)
        bi_terms_len = len(bi_terms)

        txtWordsFreqs = Counter(words)
        words_len = len(words)

        text_Vec = [0] * embedDim
        if isSemantic == True:
            X = generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
            text_Vec = X[0]

        # clusterId=findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec, dic_biterm__clusterIds)

        targetClusterIds = findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds)

        clusterId = findCloseClusterByTargetClusters(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs,
                                                     c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len,
                                                     txtWordsFreqs, words_len, max_c_id, text_Vec,
                                                     dic_biterm__clusterIds, targetClusterIds)

        c_textItems.setdefault(clusterId, []).append(item)

        max_c_id = max([max_c_id, clusterId, len(c_bitermsFreqs)])

        dic_clus__id[clusterId] = max_c_id

        txtId_txt[current_txt_id] = words

        c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq, dic_biterm__clusterIds = populateClusterFeature(
            c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs,
            bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id, text_Vec, dic_biterm__clusterId_Freq,
            dic_biterm__allClusterFreq, dic_biterm__clusterIds)

        # c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq=removeHighEntropyFtrs(c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq)

        # print('clusterId', clusterId, 'current_txt_id', current_txt_id, len(c_textItems), len(c_txtIds), words, len(targetClusterIds), len(dic_ngram__textItems))

        eval_pred_true_txt.append([clusterId, item[1], item[2]])
        if ignoreMinusOne == True:
            if str(item[1]) != '-1':
                f.write(
                    str(clusterId) + "	" + str(item[1]) + "	" + str(' '.join(item[2])) + "	" + postId + "\n")
        else:
            f.write(str(clusterId) + "	" + str(item[1]) + "	" + str(' '.join(item[2])) + "	" + postId + "\n")

        if line_count % 500 == 0:

            # print(dic_clus__id)
            print(len(dic_clus__id))
            # delete old and small clusters, remove multi-cluster words from clusters
            list_c_sizes = []
            list_c_ids = []
            # list_size__cid={}

            for c_id, txtIds in c_txtIds.items():
                list_c_sizes.append(len(txtIds))
                list_c_ids.append(dic_clus__id[c_id])
                # list_size__cid[len(txtIds)]=c_id
            mean_c_size = 0
            std_c_size = 0
            if len(list_c_sizes) > 2:
                mean_c_size = statistics.mean(list_c_sizes)
                std_c_size = statistics.stdev(list_c_sizes)

            mean_c_id = 0
            std_c_id = 0
            if len(list_c_ids) > 2:
                mean_c_id = statistics.mean(list_c_ids)
                std_c_id = statistics.stdev(list_c_ids)

            print('preocess', line_count, 'texts', 'mean_c_size', mean_c_size, 'std_c_size', std_c_size)
            print('preocess', line_count, 'texts', 'mean_c_id', mean_c_id, 'std_c_id', std_c_id)

            list_del_cids = []
            del_count = 0

            for c_id, txtIds in c_txtIds.items():
                c_size = len(txtIds)
                if ((c_size <= 1 or float(c_size) <= float(abs(mean_c_size - std_c_size))) or (
                        float(c_size) >= mean_c_size + std_c_size)) or (
                        (float(c_id) <= float(abs(mean_c_id - std_c_id))) or (
                        float(c_id) >= float(abs(mean_c_id + std_c_id)))):
                    list_del_cids.append(c_id)

            list_del_cids = set(list_del_cids)
            print('#list_del_cids', len(list_del_cids), 'len(c_bitermsFreqs)', len(c_bitermsFreqs))

            listTargetBiterms = []  # need to uncomment

            for c_id in list_del_cids:

                if c_id in c_bitermsFreqs:
                    # print('del c_id', c_id, len(c_bitermsFreqs[c_id]))
                    del c_bitermsFreqs[c_id]

                if c_id in c_totalBiterms:
                    del c_totalBiterms[c_id]

                if c_id in c_txtIds:
                    del c_txtIds[c_id]

                if c_id in c_wordsFreqs:
                    del c_wordsFreqs[c_id]

                if c_id in c_totalWords:
                    del c_totalWords[c_id]

                if c_id in dic_clus__id:
                    del dic_clus__id[c_id]

                if isSemantic == True:
                    del c_clusterVecs[c_id]

            # c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq=removeHighEntropyFtrs(c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq)

        if line_count % 1000 == 0:
            print('#######-personal-eval_pred_true_txt', len(eval_pred_true_txt))
            Evaluate(eval_pred_true_txt, ignoreMinusOne)

            t12 = datetime.now()
            t_diff = t12 - t11
            print("total time diff secs=", t_diff.seconds)

    last_txtId = current_txt_id
    return [c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId,
            dic_clus__id, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq, dic_biterm__clusterIds, c_textItems,
            dic_ngram__textItems]
