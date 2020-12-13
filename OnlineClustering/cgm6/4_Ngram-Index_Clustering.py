import os
from datetime import datetime
import math
from collections import Counter
import statistics

from general_util import readStackOverflowDataSetBody
from txt_process_util import generateGrams
from txt_process_util import generateGramsConsucetive
from clustering_gram_util import removeCommonTextIds

min_gram = 1
max_gram = 2

lang = 'r'
tagIgnore = ''
max_hitindex = 5000
textType = 'tag'  # tag, title, body
microDivide = 1000000
isStopWord = True

simtype = textType + 'Similarity'
hitranktype = textType + 'HitRank'


def buildNGramIndex(list_pred_true_words_index_postid_createtime):
    dic_ngram__txtIds = {}
    dic_txtId__text = {}

    for item in list_pred_true_words_index_postid_createtime:
        words = item[2]
        txtId = item[3]  # index

        dic_txtId__text[txtId] = item

        grams = generateGramsConsucetive(words, min_gram, max_gram)  # len(words))

        for gram in grams:
            dic_ngram__txtIds.setdefault(gram, []).append(txtId)

    return [dic_ngram__txtIds, dic_txtId__text]


def gramClusterToFeatures(dic_gram__txtIds, dic_txtId__text):
    dic_term_clusterGramIds = {}
    dic_cluster_ftrs = {}
    dic_cluster_size = {}

    for gramClusterID, txtIds in dic_gram__txtIds.items():  # gram is a cluster id
        cluster_ftrs = []
        for txtId in txtIds:
            item = dic_txtId__text[txtId]
            words = item[2]
            ftrs = generateGramsConsucetive(words, min_gram, max_gram)  # len(words))
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


# absFilePath = os.path.abspath(__file__)
# print(absFilePath)
# fileDir = os.path.dirname(os.path.abspath(__file__))
# print(fileDir)
# parentDir = os.path.dirname(fileDir)
# print(parentDir)
# parentDir = os.path.dirname(parentDir)
# print(parentDir)


outputPath = "result/"

inputfile = 'train_stackoverflow_' + lang + '_true_id_title_tags_body_createtime'

list_pred_true_words_index_postid_createtime = readStackOverflowDataSetBody(inputfile, isStopWord, 6, textType,
                                                                            tagIgnore)

t11 = datetime.now()
dic_ngram__txtIds, dic_txtId__text = buildNGramIndex(list_pred_true_words_index_postid_createtime)

########remove overlap texs from clusters, remove big clusters
#####no text appear in two clusters
# clusters based on grams: dic_nonCommon__txtIds
dic_nonCommon__txtIds, commonTxtIds = removeCommonTextIds(dic_ngram__txtIds)
print('dic_nonCommon__txtIds, commonTxtIds', len(dic_nonCommon__txtIds), len(commonTxtIds))

li = []
for gram, txtIds in dic_nonCommon__txtIds.items():
    #print(gram, 'len(txtIds)', len(txtIds))
    li.append(len(txtIds))
print('1st:min', min(li), 'max', max(li), 'median', statistics.median(li), 'avg', statistics.mean(li), 'sum of li', sum(li), 'totalclusters#', len(dic_nonCommon__txtIds))
# gram is a cluster: txtId->text->word/bigram->cluster
# gram is a cluster: c->Counter(word/bigram), c->#ftrs , c->#texts
##to be used: dic_term_clusterGramIds
##to be used: dic_cluster_ftrs
##to be used: dic_cluster_size
dic_term_clusterGramIds, dic_cluster_ftrs, dic_cluster_size = gramClusterToFeatures(dic_nonCommon__txtIds,
                                                                                    dic_txtId__text)

common_pred_true_words_index_postid_createtime = []
for commonTxtId in commonTxtIds:
    common_pred_true_words_index_postid_createtime.append(dic_txtId__text[commonTxtId])

##to be used: dic_common_ngram__txtIds
###not be used: dic_common_txtId__text
dic_common_ngram__txtIds, dic_common_txtId__text = buildNGramIndex(common_pred_true_words_index_postid_createtime)
print('dic_common_ngram__txtIds', len(dic_common_ngram__txtIds))

####


t12 = datetime.now()
t_diff = t12 - t11

print('train time for', lang, textType, str(t_diff.microseconds / float(microDivide)))

keysByLength = sorted(dic_ngram__txtIds, key=lambda key: len(dic_ngram__txtIds[key]), reverse=True)
li = []
for key in keysByLength:
    # print('clusterid=', key, '#items', len(dic_ngram__txtIds[key]))
    li.append(len(dic_ngram__txtIds[key]))
print('2nd:min', min(li), 'max', max(li), 'median', statistics.median(li), 'avg', statistics.mean(li), 'std',
      statistics.stdev(li), 'sum of li', sum(li), 'totalclusters#', len(keysByLength))

#############test section

# testPostId	trainPostId	TitleSim	BodySim	TagSim	LuceneHitRank	ProposedHitRank	lucene_hit_duration	Proposed_hit_duration_micro  LuceneTestTrueLabel


print(
    "testpostId" + "\t" + "trainPostId" + "\t" + simtype + "\t" + hitranktype + "\t" + "Proposed_hit_duration_micro" + "\t" + "Proposed_TestTrueLabel" + "\t" + "testText" + "\t" + "trainText" + "\t" + "testCreateTime" + "\t" + "TrainCreateTime" + "\t" + "DaysDiff" + "\t" + "OriginalRank")

testfile = 'test_stackoverflow_' + lang + '_true_id_title_tags_body_createtime'
testList_pred_true_words_index_postid_createtime = readStackOverflowDataSetBody(testfile, isStopWord, 6, textType,
                                                                                tagIgnore)

for item in testList_pred_true_words_index_postid_createtime:
    testTruelabel = item[1]
    words = item[2]
    testpostId = item[4]
    testCreateTime = item[5]

    testDateTime = datetime.strptime(str(item[5]).split("t")[0], "%Y-%m-%d")

    t11 = datetime.now()

    test_grams = generateGramsConsucetive(words, min_gram, max_gram)  # len(words))
    test_term_dict = Counter(test_grams)
    test_term_size = len(test_grams)

    tagetGramClusterIds = getTagetGramClusterIds(test_grams, dic_term_clusterGramIds)

    dict_cluster_sims = {}
    for gramClusterId in tagetGramClusterIds:
        # print('clusterId', clusterId, 'len(dic_tupple_class[clusterId])', len(dic_tupple_class[clusterId]))
        sim, commCount = computeTextSimCommonWord_WordDic(test_term_dict, dic_cluster_ftrs[gramClusterId],
                                                          test_term_size, dic_cluster_size[gramClusterId])

        dict_cluster_sims[gramClusterId] = sim

    # print(dict_cluster_sims.values())
    # dict_cluster_sims = {k: v for k, v in sorted(dict_cluster_sims.items(), key=lambda item: item[1], reverse=True)}
    sortList_tups__keyVal = sorted(dict_cluster_sims.items(), key=lambda x: x[1], reverse=True)
    clusterGrams = [tup[0] for tup in sortList_tups__keyVal]
    # select top similar gramClusters,
    # clusters based on grams: dic_nonCommon__txtIds
    # get txtIds from :dic_nonCommon__txtIds)
    clusterTextIds = getTextIdsFromClusters(dic_nonCommon__txtIds, clusterGrams)
    # get text ids from dic_common_ngram__txtIds
    invertIndexTextIds = aggregateTextIds(test_grams, dic_common_ngram__txtIds)
    allTextIds = set(clusterTextIds + invertIndexTextIds)
    # print('words', words, 'tagetGramClusterIds', tagetGramClusterIds, sortList_tups__keyVal, 'clusterGrams',
    # clusterGrams, 'clusterTextIds', clusterTextIds, 'invertIndexTextIds', invertIndexTextIds, 'allTextIds',
    # allTextIds)

    flag = False
    largestGram = ''
    ProposedHitRank = 0

    for txtId in allTextIds:
        ProposedHitRank += 1

        train_item = dic_txtId__text[txtId]

        trainTruelabel = train_item[1]
        train_words = train_item[2]
        trainPostId = train_item[4]
        trainCreateTime = train_item[5]

        if str(trainTruelabel) == str(testTruelabel):
            t12 = datetime.now()
            t_diff = t12 - t11

            text_sim, commonCount = computeTextSimCommonWord_WordDic(Counter(words), Counter(train_words), len(words),
                                                                     len(train_words))
            ProposedHitRank_val = int(max(1, math.floor(ProposedHitRank / len(test_grams))))

            trainDateTime = datetime.strptime(train_item[5].split("t")[0], "%Y-%m-%d")
            date_diff = trainDateTime - testDateTime
            date_diff = date_diff.days

            print(str(testpostId) + "\t" + str(trainPostId) + "\t" + str(text_sim) + "\t" + str(
                ProposedHitRank_val) + "\t" + str(t_diff.microseconds / float(microDivide)) + "\t" + str(
                testTruelabel) + "\t" + ' '.join(words) + "\t" + ' '.join(
                train_words) + "\t" + testCreateTime + "\t" + trainCreateTime + "\t" + str(date_diff) + "\t" + str(
                ProposedHitRank))
            flag = True
            break

        if ProposedHitRank > max_hitindex:
            break

    if not flag:
        t12 = datetime.now()
        t_diff = t12 - t11
        print(str(testpostId) + "\t" + "-100" + "\t0\t" + "" + "\t" + str(
            t_diff.microseconds / float(microDivide)) + "\t" + str(testTruelabel) + "\t" + ' '.join(
            words) + "\t" + "" + "\t" + "" + "\t" + "" + "\t" + "" + "\t" + "")

'''for item in testList_pred_true_words_index_postid_createtime:
    testTruelabel = item[1]
    words = item[2]
    testpostId = item[4]
    testCreateTime = item[5]

    testDateTime = datetime.strptime(str(item[5]).split("t")[0], "%Y-%m-%d")

    t11 = datetime.now()

    grams = generateGramsConsucetive(words, min_gram, max_gram)  # len(words))
    sortedGrams = list(sorted(grams, key=len, reverse=True))

    flag = False
    largestGram = ''
    ProposedHitRank = 0

    txtIds = aggregateTextIds(sortedGrams, dic_ngram__txtIds)

    for txtId in txtIds:
        ProposedHitRank += 1

        train_item = dic_txtId__text[txtId]

        trainTruelabel = train_item[1]
        train_words = train_item[2]
        trainPostId = train_item[4]
        trainCreateTime = train_item[5]

        if str(trainTruelabel) == str(testTruelabel):
            t12 = datetime.now()
            t_diff = t12 - t11

            text_sim, commonCount = computeTextSimCommonWord_WordDic(Counter(words), Counter(train_words), len(words),
                                                                     len(train_words))
            ProposedHitRank_val = int(max(1, math.floor(ProposedHitRank / len(sortedGrams))))

            trainDateTime = datetime.strptime(train_item[5].split("t")[0], "%Y-%m-%d")
            date_diff = trainDateTime - testDateTime
            date_diff = date_diff.days

            print(str(testpostId) + "\t" + str(trainPostId) + "\t" + str(text_sim) + "\t" + str(
                ProposedHitRank_val) + "\t" + str(t_diff.microseconds / float(microDivide)) + "\t" + str(
                testTruelabel) + "\t" + ' '.join(words) + "\t" + ' '.join(
                train_words) + "\t" + testCreateTime + "\t" + trainCreateTime + "\t" + str(date_diff) + "\t" + str(
                ProposedHitRank))
            flag = True
            break

        if ProposedHitRank > max_hitindex:
            break

    if not flag:
        t12 = datetime.now()
        t_diff = t12 - t11
        print(str(testpostId) + "\t" + "-100" + "\t0\t" + str(-100) + "\t" + str(
            t_diff.microseconds / float(microDivide)) + "\t" + str(testTruelabel) + "\t" + ' '.join(
            words) + "\t" + "" + "\t" + "" + "\t" + "" + "\t" + "" + "\t" + "")
'''
