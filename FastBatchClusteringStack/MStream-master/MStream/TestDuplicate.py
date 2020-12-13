#
'''import pandas as pd
from functools import reduce

title = pd.read_excel('r_hit.xlsx', sheet_name='title')
tag = pd.read_excel('r_hit.xlsx',  sheet_name='tag')
body = pd.read_excel('r_hit.xlsx', sheet_name='body')

dfs = [title, tag, body]

df_final = reduce(lambda left, right: pd.merge(left, right, on='postid', how='outer'), dfs)

df_final.to_csv('r_hit_combined.csv', index=False)'''
#


from read_pred_true_text import ReadPredTrueText
from groupTxt_ByClass import groupItemsBySingleKeyIndex
from general_util import readStackOverflowDataSetTagTitleBody
from collections import Counter
from compute_util import computeTextSimCommonWord_WordDic
import statistics

max_hit = 10000
clusterWriterFile = 'out_r_tag_pred_true_text.txt'
testFile = r'D:\githubprojects\PyMigrationRecommendation\src\notebooks' \
           r'\test_stackoverflow_r_true_id_title_tags_body_createtime '


# to change terms = oCPost.tagWords

def createTermToClsuetrId(dic_tupple_class):
    dic_term_clusterIds = {}
    dic_cluster_ftrs = {}
    dic_cluster_size = {}

    for clusterId, txt_items in dic_tupple_class.items():
        ftrs = []
        for item in txt_items:
            text = item[2]
            arr = text.split(' ')
            ftrs.extend(arr)
            for term in arr:
                dic_term_clusterIds.setdefault(term, []).append(clusterId)
        ftr_dict = Counter(ftrs)
        dic_cluster_ftrs[clusterId] = ftr_dict
        dic_cluster_size[clusterId] = len(ftrs)

    return [dic_term_clusterIds, dic_cluster_ftrs, dic_cluster_size]


def findHitIndex(dict_sortedCluster_sims, dic_tupple_class, test_oCPost):
    h_count = 0
    found = False
    cluscount = 0
    for clusterId, sim in dict_sortedCluster_sims.items():
        cluscount += 1

        # print('clusterId, sim', clusterId, sim)
        train_pred_true_texts = dic_tupple_class[clusterId]
        # h_count = 0
        for pred_true_text in train_pred_true_texts:
            h_count += 1
            trainTrueLabel = pred_true_text[1]
            if str(trainTrueLabel) == str(test_oCPost.trueLabel):
                rank = int(max(1, h_count / len(test_oCPost.tagWords)))
                print('found\t' + str(rank) + '\t' + str(test_oCPost.soPostId) + '\t' + str(test_oCPost.tagWords)+'\t'+str(test_oCPost.trueLabel))
                found = True
                h_count = max_hit + 100
                break
            # if h_count > max_hit:
            #    break

        # if h_count > max_hit:
        #    break

        if cluscount > 1000:
            break

    if not found:
        print('not\t' + str(h_count) + '\t' + str(test_oCPost.soPostId) + '\t' + str(test_oCPost.tagWords)+'\t'+str(test_oCPost.trueLabel))


listtuple_pred_true_text = ReadPredTrueText(clusterWriterFile)
dic_tupple_class = groupItemsBySingleKeyIndex(listtuple_pred_true_text, 0)  # before 0
# print(dic_tupple_class)
dic_term_clusterIds, dic_cluster_ftrs, dic_cluster_size = createTermToClsuetrId(dic_tupple_class)





#############test
test_list_CPost = readStackOverflowDataSetTagTitleBody(testFile)
# print(test_list_CPost)
for oCPost in test_list_CPost:
    terms = oCPost.tagWords
    test_term_dict = Counter(terms)
    test_term_size = len(terms)

    targetClusterIds = []
    for term in terms:
        if term not in dic_term_clusterIds:
            continue
        targetClusterIds.extend(dic_term_clusterIds[term])
    targetClusterIds = set(targetClusterIds)
    # print(terms, len(targetClusterIds))
    dict_cluster_sims = {}
    for clusterId in targetClusterIds:
        # print('clusterId', clusterId, 'len(dic_tupple_class[clusterId])', len(dic_tupple_class[clusterId]))
        sim, commCount = computeTextSimCommonWord_WordDic(test_term_dict, dic_cluster_ftrs[clusterId],
                                                          test_term_size, dic_cluster_size[clusterId])

        dict_cluster_sims[clusterId] = sim

    # print(dict_cluster_sims.values())
    dict_cluster_sims = {k: v for k, v in sorted(dict_cluster_sims.items(), key=lambda item: item[1], reverse=True)}
    simValues = dict_cluster_sims.values()

    if len(simValues) == 0:
        continue

    meanSim = statistics.mean(simValues)
    maxSim = max(simValues)
    minSim = min(simValues)
    stdSim = meanSim
    if len(simValues) > 1:
        stdSim = statistics.stdev(simValues)
    medianSim = statistics.median(simValues)
    # print(simValues, 'min', minSim, 'max', maxSim, 'median', medianSim, 'avg', meanSim, 'std',
    #      stdSim)

    findHitIndex(dict_cluster_sims, dic_tupple_class, oCPost)
