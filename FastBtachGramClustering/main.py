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

dataSetName = 'News-T'
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


def gramClusterToFeatures(dic_gram__txtIds, dic_txtId__text):
    dic_term_clusterGramIds = {}  # tag ftr->cluster id

    dic_cluster_ftrs = {}
    dic_cluster_size = {}

    dic_cluster_titles = {}
    dic_cluster_titleSize = {}

    dic_cluster_bodys = {}
    dic_cluster_bodySize = {}

    for gramClusterID, txtIds in dic_gram__txtIds.items():  # gram is a cluster id
        cluster_ftrs = []  # tags
        title_ftrs = []
        body_ftrs = []
        for txtId in txtIds:
            item = dic_txtId__text[txtId]
            words = item[2]
            title_words = item[7]
            body_words = item[8]
            ftrs = generateGramsConsucetive(words, min_gram, max_gram)  # len(words))
            t_ftrs = generateGramsConsucetive(title_words, min_gram, max_gram)  # len(words))
            b_ftrs = generateGramsConsucetive(body_words, min_gram, max_gram)  # len(words))
            cluster_ftrs.extend(ftrs)
            title_ftrs.extend(t_ftrs)
            body_ftrs.extend(b_ftrs)
            for ftr in ftrs:
                dic_term_clusterGramIds.setdefault(ftr, []).append(gramClusterID)

        ftr_dict = Counter(cluster_ftrs)
        dic_cluster_ftrs[gramClusterID] = ftr_dict
        dic_cluster_size[gramClusterID] = len(cluster_ftrs)

        ftr_dict_title = Counter(title_ftrs)
        dic_cluster_titles[gramClusterID] = ftr_dict_title
        dic_cluster_titleSize[gramClusterID] = len(title_ftrs)

        ftr_dict_body = Counter(body_ftrs)
        dic_cluster_bodys[gramClusterID] = ftr_dict_body
        dic_cluster_bodySize[gramClusterID] = len(body_ftrs)

    return [dic_term_clusterGramIds, dic_cluster_ftrs, dic_cluster_size, dic_cluster_titles, dic_cluster_titleSize,
            dic_cluster_bodys, dic_cluster_bodySize]


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

    dic_dup = {}

    for gram in grams:
        if gram not in dic_gramCluster__txtIds:
            continue
        # for tid in dic_gramCluster__txtIds[gram]:
        #    if tid in dic_dup:
        #        continue
        #    dic_dup[tid] = tid
        #    textIds.append(tid)

        textIds.extend(dic_gramCluster__txtIds[gram])

    return list(set(textIds))
    # return textIds


def clusterBatchByGram(sub_list_pred_true_words_index):
    dic_ngram__txtIds, dic_txtId__text = buildNGramIndex(sub_list_pred_true_words_index)

    ########remove overlap texs from clusters, remove big clusters
    #####no text appear in two clusters
    # clusters based on grams: dic_nonCommon__txtIds
    dic_nonCommon__txtIds = removeCommonTextIdsByCSize(dic_ngram__txtIds)

    listtuple_pred_true_text = []
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
            continue
        for txtId in txtIds:
            item = dic_txtId__text[txtId]
            # list_pred_true_words_index
            listtuple_pred_true_text.append([gram, str(item[1]), item[2]])

    print('dic_nonCommon__txtIds', len(dic_nonCommon__txtIds), 'dic_ngram__txtIds', len(dic_ngram__txtIds))
    Evaluate_old(listtuple_pred_true_text)


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

allTexts = len(list_pred_true_words_index)
batchNo = 0

for start in range(0, allTexts, batchSize):
    batchNo += 1
    end = start + batchSize if start + batchSize < allTexts else allTexts
    print(start, end)
    sub_list_pred_true_words_index = list_pred_true_words_index[start:end]
    print(len(sub_list_pred_true_words_index))
    clusterBatchByGram(sub_list_pred_true_words_index)

fileOut.close()

'''
t11 = datetime.now()
# gram is a cluster: txtId->text->word/bigram->cluster
# gram is a cluster: c->Counter(word/bigram), c->#ftrs , c->#texts
##to be used: dic_term_clusterGramIds
##to be used: dic_cluster_ftrs
##to be used: dic_cluster_size

# populate title and body here
dic_term_clusterGramIds, dic_cluster_ftrs, dic_cluster_size, dic_cluster_titles, dic_cluster_titleSize, dic_cluster_bodys, dic_cluster_bodySize = gramClusterToFeatures(
    dic_nonCommon__txtIds,
    dic_txtId__text)

#############test section

# testPostId	trainPostId	TitleSim	BodySim	TagSim	LuceneHitRank	ProposedHitRank	lucene_hit_duration	Proposed_hit_duration_micro  LuceneTestTrueLabel


# print( "testpostId" + "\t" + "trainPostId" + "\t" + simtype + "\t" + hitranktype + "\t" +
# "Proposed_hit_duration_micro" + "\t" + "Proposed_TestTrueLabel" + "\t" + "testText" + "\t" + "trainText" + "\t" +
# "testCreateTime" + "\t" + "TrainCreateTime" + "\t" + "DaysDiff" + "\t" + "OriginalRank")

fileOut.write(
    "testpostId" + "\t" + "trainPostId" + "\t" + simtype + "\t" + hitranktype + "\t" + "Proposed_hit_duration_micro" + "\t" + "Proposed_TestTrueLabel" + "\t" + "testText" + "\t" + "trainText" + "\t" + "testCreateTime" + "\t" + "TrainCreateTime" + "\t" + "DaysDiff" + "\t" + "OriginalRank\n")

testfile = 'test_stackoverflow_' + lang + '_true_id_title_tags_body_createtime'
testList_pred_true_words_index_postid_createtime = readStackOverflowDataSetTagTitleBody(testfile, isStopWord, 6,
                                                                                        textType,
                                                                                        tagIgnore)

count = 0
for item in testList_pred_true_words_index_postid_createtime:
    testTruelabel = item[1]
    words = item[2]
    testpostId = item[4]
    testCreateTime = item[5]
    title_words = item[7]
    body_words = item[8]

    testDateTime = datetime.strptime(str(item[5]).split("t")[0], "%Y-%m-%d")

    t11 = datetime.now()

    test_grams = generateGramsConsucetive(words, min_gram, max_gram)  # len(words))
    test_term_dict = Counter(test_grams)
    test_term_size = len(test_grams)

    title_grams = generateGramsConsucetive(title_words, min_gram, max_gram)  # len(words))
    test_term_dict_title = Counter(title_grams)
    test_term_size_title = len(title_grams)

    body_grams = generateGramsConsucetive(body_words, min_gram, max_gram)  # len(words))
    test_term_dict_body = Counter(body_grams)
    test_term_size_body = len(body_grams)

    tagetGramClusterIds = getTagetGramClusterIds(test_grams, dic_term_clusterGramIds)
    count += 1
    # print('count', count, 'len(tagetGramClusterIds)', len(tagetGramClusterIds))

    dict_cluster_sims = {}
    for gramClusterId in tagetGramClusterIds:
        # print('clusterId', clusterId, 'len(dic_tupple_class[clusterId])', len(dic_tupple_class[clusterId]))
        tag_sim, tag_commCount = computeTextSimCommonWord_WordDic(test_term_dict, dic_cluster_ftrs[gramClusterId],
                                                                  test_term_size, dic_cluster_size[gramClusterId])

        title_sim, title_commCount = computeTextSimCommonWord_WordDic(test_term_dict_title,
                                                                      dic_cluster_titles[gramClusterId],
                                                                      test_term_size_title,
                                                                      dic_cluster_titleSize[gramClusterId])

        body_sim, body_commCount = computeTextSimCommonWord_WordDic(test_term_dict_body,
                                                                    dic_cluster_bodys[gramClusterId],
                                                                    test_term_size_body,
                                                                    dic_cluster_bodySize[gramClusterId])

        dict_cluster_sims[gramClusterId] = tag_sim * tagWeight + title_sim * titleWeight + body_sim * bodyWeight

    # print(dict_cluster_sims.values())
    # dict_cluster_sims = {k: v for k, v in sorted(dict_cluster_sims.items(), key=lambda item: item[1], reverse=True)}
    sortList_tups__keyVal = sorted(dict_cluster_sims.items(), key=lambda x: x[1], reverse=True)
    # print('sortList_tups__keyVal', sortList_tups__keyVal)
    clusterGrams = [tup[0] for tup in sortList_tups__keyVal]
    top_clusterGrams = clusterGrams[0:min(maxClusterNos, len(clusterGrams) - 1)]
    # select top similar gramClusters,
    # clusters based on grams: dic_nonCommon__txtIds
    # get txtIds from :dic_nonCommon__txtIds)
    # clusterTextIds = getTextIdsFromClusters(dic_nonCommon__txtIds, clusterGrams)
    clusterTextIds = getTextIdsFromClusters(dic_nonCommon__txtIds, top_clusterGrams)
    # print('top_clusterGrams', len(top_clusterGrams), 'clusterTextIds', len(clusterTextIds))
    # get text ids from dic_common_ngram__txtIds
    invertIndexTextIds = []  # aggregateTextIds(test_grams, dic_common_ngram__txtIds)
    # allTextIds = set(clusterTextIds + invertIndexTextIds)
    allTextIds = clusterTextIds + invertIndexTextIds
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

            fileOut.write(str(testpostId) + "\t" + str(trainPostId) + "\t" + str(text_sim) + "\t" + str(
                ProposedHitRank_val) + "\t" + str(t_diff.microseconds / float(microDivide)) + "\t" + str(
                testTruelabel) + "\t" + ' '.join(words) + "\t" + ' '.join(
                train_words) + "\t" + testCreateTime + "\t" + trainCreateTime + "\t" + str(date_diff) + "\t" + str(
                ProposedHitRank) + '\n')
            flag = True
            break

        if ProposedHitRank > max_hitindex:
            break

    if not flag:
        t12 = datetime.now()
        t_diff = t12 - t11

        fileOut.write(str(testpostId) + "\t" + "-100" + "\t0\t" + "" + "\t" + str(
            t_diff.microseconds / float(microDivide)) + "\t" + str(testTruelabel) + "\t" + ' '.join(
            words) + "\t" + "" + "\t" + "" + "\t" + "" + "\t" + "" + "\t" + "" + "\n")

fileOut.close()
'''
