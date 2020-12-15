import os
from datetime import datetime
import math
from collections import Counter
import statistics

from general_util import readStackOverflowDataSetTagTitleBody
from general_util import readStackOverflowDataSetBody
from general_util import readlistWholeJsonDataSet
from txt_process_util import generateGrams
# from txt_process_util import generateGramsConsucetive

# from clustering_gram_util import removeCommonTextIds
from clustering_gram_util import removeCommonTextIdsByCSize

from evaluation import Evaluate_old

import uuid

min_gram = 2
max_gram = 2

lang = 'python'
tagIgnore = ''
max_hitindex = 2500
maxClusterNos = 2500
textType = 'tag'  # tag, title, body
microDivide = 1000000
isStopWord = True

tagWeight = 0.4
titleWeight = 0.2
bodyWeight = 0.4

simtype = textType + 'Similarity'
hitranktype = textType + 'HitRank'

outputPath = "result/"

dataSetName = 'News'
inputfile = 'data/' + dataSetName
batchSize = 4000

fileOut = open(outputPath + dataSetName + '_result', 'w')


def buildNGramIndex(list_pred_true_words_index):
    dic_ngram__txtIds = {}
    dic_txtId__text = {}

    for item in list_pred_true_words_index:
        words = item[2]
        txtId = item[3]  # index

        dic_txtId__text[txtId] = item

        grams = generateGrams(words, min_gram, max_gram)  # len(words))

        for gram in grams:
            dic_ngram__txtIds.setdefault(gram, []).append(txtId)

    return [dic_ngram__txtIds, dic_txtId__text]


# dic_gram__txtIds : cluster to TextIds
def gramClusterToFeatures(dic_gram__txtIds, dic_txtId__text):
    dic_term_clusterGramIds = {}  # tag ftr->cluster id

    dic_cluster_ftrs = {}
    dic_cluster_size = {}

    for gramClusterID, txtIds in dic_gram__txtIds.items():  # gram is a cluster id
        cluster_ftrs = []  # tags

        for txtId in txtIds:
            item = dic_txtId__text[txtId]
            words = item[2]

            ftrs = generateGrams(words, min_gram, max_gram)  # len(words))

            cluster_ftrs.extend(ftrs)

            for ftr in ftrs:
                dic_term_clusterGramIds.setdefault(ftr, []).append(gramClusterID)

        ftr_dict = Counter(cluster_ftrs)
        dic_cluster_ftrs[gramClusterID] = ftr_dict
        dic_cluster_size[gramClusterID] = len(cluster_ftrs)

    return [dic_term_clusterGramIds, dic_cluster_ftrs, dic_cluster_size]


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
            # commonCount=commonCount+ min(i_count,words_j[word_i])

    if txt_i_len > 0 and txt_j_len > 0:
        text_sim = float(commonCount) / (txt_i_len + txt_j_len)

    # print(text_sim, commonCount, words_i, words_j, txt_i_len, txt_j_len)

    return [text_sim, commonCount]


def aggregateTextIds(sortedGrams, dic_ngram__txtIds):
    txtIds = []
    for sortGram in sortedGrams:
        if sortGram not in dic_ngram__txtIds:
            continue
        txtIds.extend(dic_ngram__txtIds[sortGram])

    txtIds = set(txtIds)

    return list(txtIds)


def getTagetGramClusterIds(grams, dic_term_clusterGramIds):
    tagetGramClusterIds = []

    for gram in grams:
        if gram not in dic_term_clusterGramIds:
            continue
        tagetGramClusterIds.extend(dic_term_clusterGramIds[gram])

    return set(tagetGramClusterIds)


def getTextIdsFromClusters(dic_gramCluster__txtIds, grams):
    textIds = []

    for gram in grams:
        if gram not in dic_gramCluster__txtIds:
            continue

        textIds.extend(dic_gramCluster__txtIds[gram])

    return list(set(textIds))
    # return textIds


def findCloseClusterByTargetClusters(dic_cluster_ftrs, dic_cluster_size, txt_ftr_dic, txt_len,
                                     tagetGramClusterIds):
    clusterId = ''
    dict_cluster_sims = {}
    for gramClusterId in tagetGramClusterIds:
        sim, commCount = computeTextSimCommonWord_WordDic(txt_ftr_dic, dic_cluster_ftrs[gramClusterId],
                                                          txt_len, dic_cluster_size[gramClusterId])

        dict_cluster_sims[gramClusterId] = sim

    sortList_tups__keyVal = sorted(dict_cluster_sims.items(), key=lambda x: x[1], reverse=True)
    sortClusterGrams = [tup[0] for tup in sortList_tups__keyVal]
    # print('sortClusterGrams', sortClusterGrams, 'sortList_tups__keyVal', sortList_tups__keyVal)
    if len(sortClusterGrams) == 0:
        clusterId = str(uuid.uuid1())
    else:
        clusterId = sortClusterGrams[0]
    return clusterId


def clusterBatchByGram(sub_list_pred_true_words_index):
    dic_ngram__txtIds, dic_txtId__text = buildNGramIndex(sub_list_pred_true_words_index)

    ########remove overlap texs from clusters, remove big clusters
    #####no text appear in two clusters
    # clusters based on grams: dic_nonCommon__txtIds
    dic_nonCommon__txtIds = removeCommonTextIdsByCSize(dic_ngram__txtIds)

    listtuple_pred_true_text = []  # no use
    list_textId_notClust = []
    dic_nonCommon__txtIds_Clust = {}
    li = []
    for gram, txtIds in dic_nonCommon__txtIds.items():
        # print(gram, 'len(txtIds)', len(txtIds))
        li.append(len(txtIds))

    print('1st:min', min(li), 'max', max(li), 'median', statistics.median(li), 'avg', statistics.mean(li), 'sum of li',
          sum(li), 'totalclusters#', len(dic_nonCommon__txtIds))

    mean_li = statistics.mean(li)
    std_li = statistics.stdev(li)
    print('mean_li', mean_li, 'std_li', std_li)
    for gram, txtIds in dic_nonCommon__txtIds.items():
        # print(gram, 'len(txtIds)', len(txtIds))
        if len(txtIds) < mean_li + 1.0 * std_li:
            list_textId_notClust.extend(txtIds)
            continue
        dic_nonCommon__txtIds_Clust[gram] = txtIds

        # no use
        for txtId in txtIds:
            item = dic_txtId__text[txtId]
            # list_pred_true_words_index
            listtuple_pred_true_text.append([gram, str(item[1]), item[2]])
        # no use

    # no use
    print('dic_nonCommon__txtIds', len(dic_nonCommon__txtIds), 'dic_ngram__txtIds', len(dic_ngram__txtIds))

    return [dic_nonCommon__txtIds_Clust, list_textId_notClust, dic_txtId__text, listtuple_pred_true_text]


def assignTextToClusters(list_textId_notClust, dic_txtId__text, dic_term_clusterGramIds, dic_cluster_ftrs,
                         dic_cluster_size):
    list_pred_true_text_clust = []

    for textId_notClust in list_textId_notClust:
        item = dic_txtId__text[textId_notClust]
        # pred_true_words_index
        trueLabel = item[1]
        words = item[2]

        txt_ftrs = generateGrams(words, min_gram, max_gram)  # len(words))

        tagetGramClusterIds = getTagetGramClusterIds(txt_ftrs, dic_term_clusterGramIds)
        # print('tagetGramClusterIds:', tagetGramClusterIds)
        txt_len = len(txt_ftrs)
        txt_ftr_dic = Counter(txt_ftrs)
        clusterId = findCloseClusterByTargetClusters(dic_cluster_ftrs, dic_cluster_size, txt_ftr_dic, txt_len,
                                                     tagetGramClusterIds)

        if len(clusterId) < 20:
            list_pred_true_text_clust.append([clusterId, trueLabel, words])

    return list_pred_true_text_clust


# absFilePath = os.path.abspath(__file__)
# print(absFilePath)
# fileDir = os.path.dirname(os.path.abspath(__file__))
# print(fileDir)
# parentDir = os.path.dirname(fileDir)
# print(parentDir)
# parentDir = os.path.dirname(parentDir)
# print(parentDir)


list_pred_true_words_index = readlistWholeJsonDataSet(inputfile, isStopWord)

print('finish list_pred_true_words_index_postid_createtime', len(list_pred_true_words_index))

t11 = datetime.now()

allTexts = len(list_pred_true_words_index)
batchNo = 0

for start in range(0, allTexts, batchSize):
    batchNo += 1
    print('\n\nBatch', batchNo)
    end = start + batchSize if start + batchSize < allTexts else allTexts
    print('start, end', start, end)
    sub_list_pred_true_words_index = list_pred_true_words_index[start:end]
    print('total texts#', len(sub_list_pred_true_words_index))
    dic_nonCommon__txtIds_Clust, list_textId_notClust, dic_txtId__text, batch___listtuple_pred_true_text = clusterBatchByGram(
        sub_list_pred_true_words_index)
    # create word/ftr of a text (dic_nonCommon__txtIds_Clust) to-> clusterGramID
    dic_term_clusterGramIds, dic_cluster_ftrs, dic_cluster_size = gramClusterToFeatures(dic_nonCommon__txtIds_Clust,
                                                                                        dic_txtId__text)
    list_pred_true_text_clust = assignTextToClusters(list_textId_notClust, dic_txtId__text, dic_term_clusterGramIds,
                                                     dic_cluster_ftrs, dic_cluster_size)

    Evaluate_old(batch___listtuple_pred_true_text)
    print()
    Evaluate_old(list_pred_true_text_clust)
    print()
    Evaluate_old(list_pred_true_text_clust + batch___listtuple_pred_true_text)

t12 = datetime.now()
print('total time diff secs=', (t12 - t11).seconds)

fileOut.close()

'''
t11 = datetime.now()
# gram is a cluster: txtId->text->word/bigram->cluster
# gram is a cluster: c->Counter(word/bigram), c->#ftrs , c->#texts
##to be used: dic_term_clusterGramIds
##to be used: dic_cluster_ftrs
##to be used: dic_cluster_size


'''
