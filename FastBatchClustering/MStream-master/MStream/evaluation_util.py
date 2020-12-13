from collections import Counter
from evaluation import Evaluate_old


def evaluateByGram(dic_gramkeys_txtInds, seen_list_pred_true_words_index):
    texts_clustered_sum = 0
    max_group_sum = 0
    # bigger_clusters_tri=0
    # bigger_clusters_bi=0
    predsSeen_list_pred_true_words_index = []
    unique_txtIds = []

    temp_txtId_to_pred = {}

    # print("evaluateByGram=len(seen_list_pred_true_words_index)", len(seen_list_pred_true_words_index))

    for mergedKey, txtInds in dic_gramkeys_txtInds.items():
        txtInds = list(set(txtInds))
        texts_clustered_sum += len(txtInds)
        unique_txtIds = unique_txtIds + txtInds
        # if len(txtInds)>1: bigger_clusters_tri+=1
        # print("evaluateByGram",mergedKey, txtInds)
        true_label_list = []
        for txtInd in txtInds:
            # print("evaluateByGram=txtInd", txtInd)
            # temp , may be useful
            temp_txtId_to_pred.setdefault(txtInd, []).append(mergedKey)
            if len(temp_txtId_to_pred[txtInd]) > 1:
                print("batch-eval, temp_txtId_to_pred=", txtInd, temp_txtId_to_pred[txtInd])
                continue
            # temp
            # print("evaluateByGram=seen_list_pred_true_words_index[txtInd]", seen_list_pred_true_words_index[txtInd])
            true_label_list.append(seen_list_pred_true_words_index[txtInd][1])

            predsSeen_list_pred_true_words_index.append(
                [mergedKey, seen_list_pred_true_words_index[txtInd][1], seen_list_pred_true_words_index[txtInd][2],
                 seen_list_pred_true_words_index[txtInd][3]])

        max_group_sum += max(Counter(true_label_list).values())
        # print("true_label_list", len(true_label_list), true_label_list)

    # print("batch-eval", max_group_sum, texts_clustered_sum, "accuracy", max_group_sum/texts_clustered_sum, "#clusters", len(dic_gramkeys_txtInds), '#unique_txtIds', len(set(unique_txtIds)))
    return predsSeen_list_pred_true_words_index


def evaluateByGramUsingDic(dic_gram__docs):
    dic_docId__cluster = {}
    list_pred_true_text = []
    clusterIndex = 0
    for gram, docs in dic_gram__docs.items():
        clusterIndex += 1
        for doc in docs:
            list_pred_true_text.append([clusterIndex, doc.trueLabel, doc.text])
            dic_docId__cluster[doc.documentID] = clusterIndex

    Evaluate_old(list_pred_true_text)
    return dic_docId__cluster
