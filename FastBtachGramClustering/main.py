import os
from datetime import datetime
import math
from collections import Counter
import statistics
from statistics import mean
from scipy import spatial

from general_util import readStackOverflowDataSetTagTitleBody
from general_util import readStackOverflowDataSetBody
from general_util import readlistWholeJsonDataSet
from txt_process_util import generateGrams
# from txt_process_util import generateGramsConsucetive
from word_vec_extractor import extractAllWordVecsPartialStemming
from sent_vecgenerator import generate_sent_vecs_toktextdata

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

dataSetName = 'News'  # 'Tweets-T' #'News-T' #stackoverflow_large
inputfile = 'data/' + dataSetName
batchSize = 4000
embedDim = 50
embeddingfile = 'D:/glove.42B.300d/glove.6B/glove.6B.50d.txt'

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

    dic_word_clusterGramIds = {}  # tag ftr->cluster id
    dic_cluster_words = {}
    dic_cluster_wordSize = {}

    for gramClusterID, txtIds in dic_gram__txtIds.items():  # gram is a cluster id
        cluster_ftrs = []  # tags
        cluster_words = []  # tags

        for txtId in txtIds:
            item = dic_txtId__text[txtId]
            words = item[2]
            ftrs = generateGrams(words, min_gram, max_gram)  # len(words))

            cluster_ftrs.extend(ftrs)
            cluster_words.extend(words)

            for ftr in ftrs:
                dic_term_clusterGramIds.setdefault(ftr, []).append(gramClusterID)

            for word in words:
                dic_word_clusterGramIds.setdefault(word, []).append(gramClusterID)

        dic_cluster_ftrs[gramClusterID] = Counter(cluster_ftrs)
        dic_cluster_size[gramClusterID] = len(cluster_ftrs)

        dic_cluster_words[gramClusterID] = Counter(cluster_words)
        dic_cluster_wordSize[gramClusterID] = len(cluster_words)

    # print(dic_word_clusterGramIds, dic_cluster_words, dic_cluster_wordSize)

    return [dic_term_clusterGramIds, dic_cluster_ftrs, dic_cluster_size, dic_word_clusterGramIds, dic_cluster_words,
            dic_cluster_wordSize]


def populateClusterVecs(dic_nonCommon__txtIds_Clust, dic_txtId__text):
    dic_clusteVecs = {}

    for gramKey, txtIds in dic_nonCommon__txtIds_Clust.items():
        data = []
        for txtId in txtIds:
            item = dic_txtId__text[txtId]
            words = item[2]
            X = generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
            text_Vec = X[0]
            data.append(text_Vec)
        avg = [sum(col) / float(len(col)) for col in zip(*data)]
        dic_clusteVecs[gramKey] = avg

    return dic_clusteVecs


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
                                     tagetGramClusterIds, dic_cluster_words, dic_cluster_wordSize, txt_word_dic,
                                     txt_word_len,
                                     tageWordClusterIds, dic_clusteVecs={}, words={}):
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

        dict_cluster_sims = {}
        for gramClusterId in tageWordClusterIds:
            sim, commCount = computeTextSimCommonWord_WordDic(txt_word_dic, dic_cluster_words[gramClusterId],
                                                              txt_word_len, dic_cluster_wordSize[gramClusterId])

            dict_cluster_sims[gramClusterId] = sim

        sortList_tups__keyVal = sorted(dict_cluster_sims.items(), key=lambda x: x[1], reverse=True)
        sortClusterGrams = [tup[0] for tup in sortList_tups__keyVal]

        if len(sortClusterGrams) >= 2:
            allSims = list(dict_cluster_sims.values())
            mean_sim = statistics.mean(allSims)
            std_sim = statistics.stdev(allSims)
            max_sim = max(allSims)
            if max_sim > mean_sim + std_sim:
                clusterId = sortClusterGrams[0]
            # print('findCloseClusterByTargetClusters:dict_cluster_sims', dict_cluster_sims.values())

        '''if len(dic_clusteVecs) > 2:
            X = generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
            text_Vec = X[0]
            # print('Semantic computations:use !!tagetGramClusterIds', text_Vec, len(dic_clusteVecs))
            max_sim = 0
            max_clusId = str(uuid.uuid1())
            for clusterVecId, clusterVec in dic_clusteVecs.items():
                # print('clusterVecId', clusterVecId, clusterVec)
                similarity = 1 - spatial.distance.cosine(text_Vec, clusterVec)
                # print('clusterVecId: ', clusterVecId, similarity)
                if max_sim < similarity:
                    max_sim = similarity
                    max_clusId = clusterVecId

            clusterId = max_clusId
        

        else:
            clusterId = str(uuid.uuid1())'''
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
        if len(txtIds) < mean_li + 0.1 * std_li:
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
                         dic_cluster_size, dic_word_clusterGramIds, dic_cluster_words, dic_cluster_wordSize,
                         dic_clusteVecs={}):
    list_pred_true_text_clust = []

    for textId_notClust in list_textId_notClust:
        item = dic_txtId__text[textId_notClust]
        # pred_true_words_index
        trueLabel = item[1]
        words = item[2]

        txt_ftrs = generateGrams(words, min_gram, max_gram)  # len(words))

        tagetGramClusterIds = getTagetGramClusterIds(txt_ftrs, dic_term_clusterGramIds)
        tageWordClusterIds = getTagetGramClusterIds(words, dic_word_clusterGramIds)
        # print('tageWordClusterIds:', tageWordClusterIds, 'tagetGramClusterIds', tagetGramClusterIds)

        clusterId = findCloseClusterByTargetClusters(dic_cluster_ftrs, dic_cluster_size, Counter(txt_ftrs),
                                                     len(txt_ftrs),
                                                     tagetGramClusterIds, dic_cluster_words, dic_cluster_wordSize,
                                                     Counter(words), len(words),
                                                     tageWordClusterIds, dic_clusteVecs, words)

        # if len(clusterId) < 20:
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

all_words = []
for item in list_pred_true_words_index:
    all_words.extend(item[2])
all_words = list(set(all_words))

wordVectorsDic = {}
# wordVectorsDic = extractAllWordVecsPartialStemming(embeddingfile, embedDim, all_words)

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
    dic_clusteVecs = populateClusterVecs(dic_nonCommon__txtIds_Clust, dic_txtId__text)
    # print(dic_clusteVecs)

    # create word/ftr of a text (dic_nonCommon__txtIds_Clust) to-> clusterGramID
    dic_term_clusterGramIds, dic_cluster_ftrs, dic_cluster_size, dic_word_clusterGramIds, dic_cluster_words, dic_cluster_wordSize = gramClusterToFeatures(
        dic_nonCommon__txtIds_Clust, dic_txtId__text)

    list_pred_true_text_clust = assignTextToClusters(list_textId_notClust, dic_txtId__text, dic_term_clusterGramIds,
                                                     dic_cluster_ftrs, dic_cluster_size, dic_word_clusterGramIds,
                                                     dic_cluster_words, dic_cluster_wordSize, dic_clusteVecs)

    Evaluate_old(batch___listtuple_pred_true_text)
    print()
    Evaluate_old(list_pred_true_text_clust)
    print()
    Evaluate_old(list_pred_true_text_clust + batch___listtuple_pred_true_text)
    batch_result = list_pred_true_text_clust + batch___listtuple_pred_true_text
    for item in batch_result:
        fileOut.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')

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
