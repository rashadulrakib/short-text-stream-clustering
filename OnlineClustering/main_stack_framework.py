from CSimilarityFlgas import CSimilarityFlgas

import os
from datetime import datetime

from general_util import readStackOverflowDataSetTagTitleBody
from evaluation import Evaluate
from read_pred_true_text import ReadPredTrueTextPostid
from clustering_term_online_stack_framework import cluster_biterm_framework
from word_vec_extractor import extractAllWordVecsPartialStemming

from clustering_term_online_stack_test_data_framework import test_cluster_bitermMapping_buffer_framework

ignoreMinusOne = False
isSemantic = False

absFilePath = os.path.abspath(__file__)
print(absFilePath)
fileDir = os.path.dirname(os.path.abspath(__file__))
print(fileDir)
parentDir = os.path.dirname(fileDir)
print(parentDir)
parentDir = os.path.dirname(parentDir)
print(parentDir)

dataDir = "data/"
outputPath = "result/"

min_gram = 1
max_gram = 2

lang = 'r'
tagIgnore = ''
max_hitindex = 5000
textType = 'tag'  # tag, title, body
microDivide = 1000000
isStopWord = True
columnsInFile = 6
randMax = 0
isTagSim = True
isTitleSim = True
isBodySim = True
tagWeight = 0.65  # 0.2
titleWeight = 0.05  # 0.1
bodyWeight = 0.30  # 0.7

oCSimilarityFlgas = CSimilarityFlgas(isTagSim, isTitleSim, isBodySim, tagWeight, titleWeight, bodyWeight)

simtype = textType + 'Similarity'
hitranktype = textType + 'HitRank'

# parentDir='/users/grad/rakib/stackoverflow'
# inputfile = parentDir+'/PyMigrationRecommendation/src/notebooks/train_stackoverflow_r_true_id_title_tags'
inputfile = parentDir + '/PyMigrationRecommendation/src/notebooks/train_stackoverflow_' + lang + '_true_id_title_tags_body_createtime'

resultFile = outputPath + 'train_biterm_' + lang + '_' + textType + '.txt'

list_CPost = readStackOverflowDataSetTagTitleBody(inputfile, isStopWord, columnsInFile, tagIgnore, randMax)

print(len(list_CPost))

gloveFile = "glove.6B.50d.txt"
embedDim = 50
wordVectorsDic = {}
if isSemantic == True:
    all_words = []
    for oCPost in list_CPost:
        all_words.extend(oCPost.tagWords)
        all_words.extend(oCPost.titleWords)
        all_words.extend(oCPost.bodyWords)
    all_words = list(set(all_words))
    wordVectorsDic = extractAllWordVecsPartialStemming(gloveFile, embedDim, all_words)

if os.path.exists(resultFile):
    os.remove(resultFile)

dic_ngram__txtIds = {}
c_itemsCount = {}

c_CFVector = {}

dic_txtId__CPost = {}
dic_bitermTag__clusterIds = {}
dic_bitermTitle__clusterIds = {}
dic_bitermBody__clusterIds = {}

max_c_id = 0
dic_clus__id = {}

f = open(resultFile, 'w')

t11 = datetime.now()

# c_bitermsFreqs, c_totalBiterms, c_txtIds, c_clusterVecs,
c_CFVector, max_c_id, dic_txtId__CPost, dic_clus__id, dic_bitermTag__clusterIds, dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds, dic_ngram__txtIds, c_itemsCount = cluster_biterm_framework(
    f, list_CPost, c_CFVector, max_c_id, dic_txtId__CPost, wordVectorsDic, dic_clus__id, dic_bitermTag__clusterIds,
    dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds, dic_ngram__txtIds, min_gram, max_gram, oCSimilarityFlgas,
    c_itemsCount)

t12 = datetime.now()
t_diff = t12 - t11
print("total time diff secs=", t_diff.seconds)

f.close()

listtuple_pred_true_text = ReadPredTrueTextPostid(resultFile, ignoreMinusOne)

print('result for', inputfile)
Evaluate(listtuple_pred_true_text)

####test

testfile = parentDir + '/PyMigrationRecommendation/src/notebooks/test_stackoverflow_' + lang + '_true_id_title_tags_body_createtime'

list_CPost_test = readStackOverflowDataSetTagTitleBody(testfile, isStopWord, columnsInFile, tagIgnore, randMax)
print('list_CPost_test', len(list_CPost_test))

# max_c_id=0 #no impact
# c_txtIds, txtId_txt,  no impact

test_cluster_bitermMapping_buffer_framework(list_CPost_test, c_CFVector, dic_txtId__CPost, dic_bitermTag__clusterIds,
                                            dic_bitermTitle__clusterIds, dic_bitermBody__clusterIds, dic_ngram__txtIds,
                                            min_gram, max_gram, max_hitindex, oCSimilarityFlgas, wordVectorsDic)
