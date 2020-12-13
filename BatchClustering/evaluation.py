from combine_predtruetext import combinePredTrueText
from read_clust_label import readClustLabel
from groupTxt_ByClass import groupTxtByClass
from groupTxt_ByClass import groupItemsBySingleKeyIndex
from sklearn import metrics
import re
from general_util import print_by_group
import statistics

minIntVal = -1000000

dataFileTxtTrue = "/home/owner/PhD/dr.norbert/dataset/shorttext/data-web-snippets/stc2/Stc2TrueBody"
extClustFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/data-web-snippets/stc2/Stc2Preds"
dataFilePredTrueTxt = "/home/owner/PhD/MStream-master/MStream/result/NewsPredTueTextMStream_WordArr.txt"

dataFilePredTrueTxt = "/home/owner/PhD/MStream-master/MStream/result/mstr-enh"


def ComputePurity(dic_tupple_class):
    totalItems = 0
    maxGroupSizeSum = 0
    for label, pred_true_txts in dic_tupple_class.items():
        totalItems = totalItems + len(pred_true_txts)
        # print("pred label="+label+", #texts="+str(len(pred_true_txts)))
        dic_tupple_class_originalLabel = groupTxtByClass(pred_true_txts, True)
        maxMemInGroupSize = -1000000
        maxMemOriginalLabel = ""
        for orgLabel, org_pred_true_txts in dic_tupple_class_originalLabel.items():
            # print("orgLabel label="+orgLabel+", #texts="+str(len(org_pred_true_txts)))
            if maxMemInGroupSize < len(org_pred_true_txts):
                maxMemInGroupSize = len(org_pred_true_txts)
                maxMemOriginalLabel = orgLabel

        # print("\n")
        # print(str(label)+" purity="+str(maxMemInGroupSize/len(pred_true_txts))+", items="+str(len(pred_true_txts))+", max match#="+str(maxMemInGroupSize))
        # print_by_group(pred_true_txts)
        maxGroupSizeSum = maxGroupSizeSum + maxMemInGroupSize

    purity = maxGroupSizeSum / float(totalItems)
    print("purity majority whole data=" + str(purity))
    return purity


def ReadPredTrueText():
    print("pred_true_text=" + dataFilePredTrueTxt)
    listtuple_pred_true_text = []
    file1 = open(dataFilePredTrueTxt, "r")
    lines = file1.readlines()
    file1.close()

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        arr = re.split("\t", line)
        if len(arr) != 3:
            continue
        tupPredTrueTxt = [arr[0], arr[1], arr[2]]
        listtuple_pred_true_text.append(tupPredTrueTxt)

    return listtuple_pred_true_text


def MergeAndWriteTrainTest():
    clustlabels = readClustLabel(extClustFile)
    listtuple_pred_true_text, uniqueTerms = combinePredTrueText(clustlabels, dataFileTxtTrue)
    # WriteTrainTestInstances(traintestFile, listtuple_pred_true_text)
    return listtuple_pred_true_text


def Evaluate(listtuple_pred_true_text):
    print("evaluate total texts=" + str(len(listtuple_pred_true_text)))
    preds = []
    trues = []
    for pred_true_text in listtuple_pred_true_text:
        preds.append(pred_true_text[0])
        trues.append(pred_true_text[1])

    score = metrics.homogeneity_score(trues, preds)
    print("homogeneity_score-whole-data:   %0.8f" % score)

    score = metrics.completeness_score(trues, preds)
    print("completeness_score-whole-data:   %0.8f" % score)

    score=metrics.v_measure_score(trues, preds)
    print ("v_measure_score-whole-data:   %0.4f" % score)

    nmi_score = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic')
    print("nmi_score-whole-data:   %0.8f" % nmi_score)

    # score=metrics.adjusted_mutual_info_score(trues, preds)
    # print ("adjusted_mutual_info_score-whole-data:   %0.4f" % score)

    # score=metrics.adjusted_rand_score(trues, preds)
    # print ("adjusted_rand_score-whole-data:   %0.4f" % score)

    dic_tupple_class = groupTxtByClass(listtuple_pred_true_text, False)
    dic_tupple_class_true = groupTxtByClass(listtuple_pred_true_text, True)
    print("pred clusters=" + str(len(dic_tupple_class)) + ", true clusters=" + str(len(dic_tupple_class_true)))
    purity = ComputePurity(dic_tupple_class)
    '''print("---Pred distribution")
 for key,value in dic_tupple_class.items():
   print(key, len(value))
 print("---True distribution")
 for key,value in dic_tupple_class_true.items():
   print(key, len(value))'''
    return [purity, nmi_score]


def Evaluate_old(listtuple_pred_true_text, ignoreMinusOne=False):
    preds = []
    trues = []

    new_listtuple_pred_true_text = []

    totalwords = 0

    for pred_true_text in listtuple_pred_true_text:
        if str(pred_true_text[1]) == '-1' and ignoreMinusOne == True:
            continue

        preds.append(pred_true_text[0])
        trues.append(pred_true_text[1])
        new_listtuple_pred_true_text.append([pred_true_text[0], pred_true_text[1], pred_true_text[2]])

        totalwords += len(pred_true_text[2])
        # print(pred_true_text[2], totalwords)

    print("evaluate total texts=" + str(len(new_listtuple_pred_true_text)))

    score = metrics.homogeneity_score(trues, preds)
    print("homogeneity_score-whole-data:   %0.8f" % score)

    score = metrics.completeness_score(trues, preds)
    print("completeness_score-whole-data:   %0.8f" % score)

    score = metrics.v_measure_score(trues, preds)
    print("v_measure_score-whole-data:   %0.8f" % score)

    score = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic')
    print("nmi_score-whole-data:   %0.8f" % score)

    # score=metrics.adjusted_mutual_info_score(trues, preds)
    # print ("adjusted_mutual_info_score-whole-data:   %0.4f" % score)

    # score=metrics.adjusted_rand_score(trues, preds)
    # print ("adjusted_rand_score-whole-data:   %0.4f" % score)

    dic_tupple_class = groupItemsBySingleKeyIndex(new_listtuple_pred_true_text, 0)  # before 0
    dic_tupple_class_true = groupItemsBySingleKeyIndex(new_listtuple_pred_true_text, 1)  # before 1
    print("pred clusters=" + str(len(dic_tupple_class)) + ", true clusters=" + str(len(dic_tupple_class_true)))
    ComputePurity(dic_tupple_class)
    li = [len(dic_tupple_class_true[x]) for x in dic_tupple_class_true if isinstance(dic_tupple_class_true[x], list)]
    print('min', min(li), 'max', max(li), 'median', statistics.median(li), 'avg', statistics.mean(li), 'std',
          statistics.stdev(li), 'sum of li', sum(li))
    print('avg words per text', totalwords / len(new_listtuple_pred_true_text), 'totalwords', totalwords, '#texts',
          len(new_listtuple_pred_true_text))
    '''print("---Pred distribution")
 for key,value in dic_tupple_class.items():
   print(key, len(value))
 print("---True distribution")
 for key,value in dic_tupple_class_true.items():
   print(key, len(value))'''


if __name__ == '__main__':
    # listtuple_pred_true_text = MergeAndWriteTrainTest()
    listtuple_pred_true_text = ReadPredTrueText()
    Evaluate(listtuple_pred_true_text)
