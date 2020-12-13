from scipy.spatial.distance import cosine
import numpy as np
from compute_util import compute_sim_value
from sklearn.feature_extraction import stop_words
import itertools


def loadStopWords(file):
    f = open(file, 'r')
    lines = f.readlines()
    stopWs = []
    for line in lines:
        line = line.strip().lower()
        if len(line) == 0:
            continue
        stopWs.append(line)
    f.close()
    return set(stopWs)


def concatWordsSort_Space(words):
    words.sort()
    combinedWord = ' '.join(words)
    return combinedWord


def concatWords_Space(words):
    combinedWord = ' '.join(words)
    return combinedWord


def generateGrams(words, minGSize, maxGSize):
    words.sort()
    grams = []
    if len(words) <= minGSize:
        return [concatWords_Space(words)]

    for size in range(minGSize, maxGSize + 1):
        subset = list(itertools.combinations(words, size))
        for tuple in subset:
            grams.append(concatWords_Space(list(tuple)))

    return list(set(grams))


def generateGramsConsucetive(words, minGSize, maxGSize):
    words.sort()
    grams = []
    if len(words) <= minGSize:
        return [concatWords_Space(words)]

    for size in range(minGSize, maxGSize + 1):
        for i in range(len(words) - size + 1):
            ng = ''
            for j in range(size):
                ng = ng + ' ' + words[i + j]  # j is the offset

            # print(ng)
            grams.append(ng.strip())

    return list(set(grams))


def concatWordsSort(words):
    words.sort()
    combinedWord = ''.join(words)
    return combinedWord


def combineDocsToSingle(listStrs):
    comText = " ".join(listStrs)
    return comText


def createBinaryWordCooccurenceMatrix(listtuple_pred_true_text):
    binGraph = []
    uniqueWordList = set()
    docWords = []
    for i in range(len(listtuple_pred_true_text)):
        words = listtuple_pred_true_text[i][2]
        uniqueWordList.update(words)
        docWords.append(words)

    dic_word_index = {}
    i = -1
    for word in uniqueWordList:
        i = i + 1
        dic_word_index[word] = i

    m = len(uniqueWordList)
    binGraph = [[0] * m for i in range(m)]
    for words in docWords:
        for i in range(1, len(words)):
            id1 = dic_word_index[words[i - 1]]
            id2 = dic_word_index[words[i]]
            binGraph[id1][id2] = 1

    return [binGraph, dic_word_index, docWords]


def createTerm_Doc_matrix_dic(dic_bitri_keys_selectedClusters_seenBatch):
    term_doc_matrix = []  # n by m matrix

    unique_txtIds = []

    for key, txtInds in dic_bitri_keys_selectedClusters_seenBatch.items():
        unique_txtIds = unique_txtIds + txtInds

    unique_txtIds = set(unique_txtIds)

    n = len(dic_bitri_keys_selectedClusters_seenBatch)
    m = len(unique_txtIds)
    term_doc_matrix = [[0] * m for i in range(n)]
    print("unique_txtIds", len(unique_txtIds))
    dic_txt_index = {}
    i = -1
    for txtInd in unique_txtIds:
        i += 1
        dic_txt_index[txtInd] = i

    rowId = -1
    for key, txtInds in dic_bitri_keys_selectedClusters_seenBatch.items():
        rowId += 1
        for txtInd in txtInds:
            colId = dic_txt_index[txtInd]
            term_doc_matrix[rowId][colId] = 1

    # print(term_doc_matrix)
    return [term_doc_matrix, dic_txt_index]


def createTextBinaryGraphMatrixByCommonWord(listtuple_pred_true_text):
    binGraph = []  # n by m matrix
    n = len(listtuple_pred_true_text)
    m = len(listtuple_pred_true_text)
    binGraph = [[0] * m for i in range(n)]
    for i in range(len(listtuple_pred_true_text)):
        binGraph[i][i] = 0
        for j in range(i + 1, len(listtuple_pred_true_text)):
            txt_i = listtuple_pred_true_text[i][2]
            txt_j = listtuple_pred_true_text[j][2]
            text_sim, commonCount = computeTextSimCommonWord(txt_i, txt_j)
            if commonCount > 0:
                binGraph[i][j] = 1
                # binGraph[j][i]=1

    return binGraph


def getDicWordToClusterEntropy(dic_txt_to_cluster, dic_word_to_txt):
    dic_word_to_clusterEntropy = {}
    dic_word_to_cluster_indecies = {}

    for word, txtIndecies in dic_word_to_txt.items():
        word_cluster_indecies = []
        for txtIndex in txtIndecies:
            word_cluster_indecies.append(dic_txt_to_cluster[txtIndex])
        dic_word_to_cluster_indecies[word] = word_cluster_indecies
        dic_word_to_clusterEntropy[word] = ComputeEntropy(word_cluster_indecies)

    # print(dic_word_to_clusterEntropy)
    # print(dic_word_to_cluster_indecies)

    return [dic_word_to_clusterEntropy, dic_word_to_cluster_indecies]


def getDicTxtToClass_WordToTxt(listtuple_pred_true_text):
    dic_txt_to_cluster = {}
    dic_word_to_txt = {}

    for i in range(len(listtuple_pred_true_text)):
        pred_true_txt = listtuple_pred_true_text[i]
        pred = pred_true_txt[0]
        true = pred_true_txt[1]
        words = pred_true_txt[2]
        dic_txt_to_cluster[i] = pred
        # words=txt.split()#=word_tokenize(txt)
        for word in words:
            dic_word_to_txt.setdefault(word, []).append(i)

            # print("dic_txt_to_cluster")
    # print(dic_txt_to_cluster)

    # print("dic_word_to_txt")
    # print(dic_word_to_txt)

    return [dic_txt_to_cluster, dic_word_to_txt]


def getDocFreq(texts):
    totalDocs = len(texts)
    dicDocFreq = {}
    for text in texts:
        uniqueWords = set(word_tokenize(text))
        for word in uniqueWords:
            if word in dicDocFreq:
                dicDocFreq[word] = dicDocFreq[word] + 1
            else:
                dicDocFreq[word] = 1

    return dicDocFreq


def getDocFreqWithIds(texts):
    totalDocs = len(texts)
    dicDocFreq = {}  # [(textids...), freq]
    dicTextIds = {}

    i = -1
    for text in texts:
        i = i + 1
        uniqueWords = set(word_tokenize(text))
        for word in uniqueWords:
            if word in dicDocFreq:
                textIds = dicTextIds[word]
                docFreq = dicDocFreq[word]
                textIds.append(i)
                docFreq = docFreq + 1
                dicDocFreq.update({word: docFreq})
                dicTextIds.update({word: textIds})
            else:
                dicDocFreq.update({word: 1})
                dicTextIds.update({word: [i]})

    return [dicDocFreq, dicTextIds]


def stem_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    # text = text.translate(string.punctuation)
    tokens = word_tokenize(text)

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def preprocess(str):
    str = re.sub(r'[^a-zA-Z0-9 ]', ' ', str)
    # str=re.sub(r'\b[a-z]\b|\b\d+\b', '', str)
    # str=re.sub(r'lt', ' ', str)
    # str=re.sub(r'gt', ' ', str)
    str = re.sub(r'\s+', ' ', str).strip()
    return str


def getScikitLearn_StopWords():
    return set(list(stop_words.ENGLISH_STOP_WORDS))


def processTxtRemoveStopWordTokenized_wordArr(txt, stopWs):
    # print(txt)
    str = preprocess(txt)
    # print(str)
    word_tokens = word_tokenize(str)
    # print(word_tokens)
    # wordArr=[]
    # for w in word_tokens:
    # if w not in stopWs:
    #  wordArr.append(w)
    wordArr = [w for w in word_tokens if not w in stopWs]
    # print(wordArr, word_tokens, txt)
    return wordArr


def processTextsRemoveStopWordTokenized_wordArr(texts, skStopWords):
    newTexts_wordArr = []
    for text in texts:
        # print(text)
        wordArr = processTxtRemoveStopWordTokenized_wordArr(text, skStopWords)
        newTexts_wordArr.append(wordArr)
    return newTexts_wordArr


def processTxtRemoveStopWordTokenized(txt, stopWs):
    wordArr = processTxtRemoveStopWordTokenized_wordArr(txt, stopWs)
    return ' '.join(wordArr)


def processTextsRemoveStopWordTokenized(texts, skStopWords):
    newTexts = []
    for text in texts:
        sent = processTxtRemoveStopWordTokenized(text, skStopWords)
        newTexts.append(sent)
    return newTexts


def processTxtStopWordsStemming(txt, stopWs, ps):
    str = preprocess(txt)
    word_tokens = word_tokenize(str)
    wordArr = [ps.stem(w) for w in word_tokens if not w in stopWs]
    return ' '.join(wordArr)


def extractHighEntopyWords(dic_word_to_clusterEntropy):
    high_entropy_words = []
    values = list(dic_word_to_clusterEntropy.values())
    mean = statistics.mean(values)
    std = statistics.stdev(values)

    for word, entropy in dic_word_to_clusterEntropy.items():
        if entropy > 1.2 * (mean + std):
            high_entropy_words.append(word)

    # high_entropy_words=[]
    print(
        "high_entropy_words=" + str(len(high_entropy_words)) + ", total words=" + str(len(dic_word_to_clusterEntropy)))

    # return set(list(set(high_entropy_words))[:50])
    return set(high_entropy_words)


def removeWordsByEntropy(words, dic_word_to_clusterEntropy, dic_word_to_cluster_indecies):
    cleaned_word_arr = words

    # words = word_tokenize(text)
    good_words = []
    for word in words:
        ent = dic_word_to_clusterEntropy[word]
        if ent == 0.0:
            good_words.append(word)

    if len(good_words) > 0:
        cleaned_word_arr = good_words

    return cleaned_word_arr


def removeWordsByHighEntropy(words, dic_word_to_clusterEntropy, dic_word_to_cluster_indecies, high_entropy_words):
    cleaned_word_arr = words
    # words = word_tokenize(text)
    good_words = []
    for word in words:
        if word not in high_entropy_words:
            good_words.append(word)

    if len(good_words) > 0:
        cleaned_word_arr = good_words

    return cleaned_word_arr


def RemoveHighClusterEntropyWords(pred_true_txts):
    cleaned_pred_true_txts = []
    dic_txt_to_cluster, dic_word_to_txt = getDicTxtToClass_WordToTxt(pred_true_txts)
    dic_word_to_clusterEntropy, dic_word_to_cluster_indecies = getDicWordToClusterEntropy(dic_txt_to_cluster,
                                                                                          dic_word_to_txt)

    for i in range(len(pred_true_txts)):
        pred_true_text = pred_true_txts[i]
        pred = pred_true_text[0]
        true = pred_true_text[1]
        word_arr = pred_true_text[2]
        cleaned_word_arr = removeWordsByEntropy(word_arr, dic_word_to_clusterEntropy, dic_word_to_cluster_indecies)
        cleaned_pred_true_txts.append([pred, true, cleaned_word_arr])
        # print(text+", cleaned="+cleanedText)

    return cleaned_pred_true_txts


def ExtractHighClusterEntropyWordNo(dic_txt_to_cluster, dic_word_to_txt):
    dic_word_to_clusterEntropy, dic_word_to_cluster_indecies = getDicWordToClusterEntropy(dic_txt_to_cluster,
                                                                                          dic_word_to_txt)

    dic_word_to_clusterEntropy, dic_word_to_cluster_indecies = getDicWordToClusterEntropy(dic_txt_to_cluster,
                                                                                          dic_word_to_txt)

    high_entropy_words = extractHighEntopyWords(dic_word_to_clusterEntropy)
    return high_entropy_words


def RemoveHighClusterEntropyWordsIndex(pred_true_txt_ind_prevPreds):
    cleaned_pred_true_txt_ind_prevPreds = []
    dic_txt_to_cluster, dic_word_to_txt = getDicTxtToClass_WordToTxt(
        pred_true_txt_ind_prevPreds)  # dic_txt_to_cluster (txt_0-> cluster1)
    dic_word_to_clusterEntropy, dic_word_to_cluster_indecies = getDicWordToClusterEntropy(dic_txt_to_cluster,
                                                                                          dic_word_to_txt)

    high_entropy_words = extractHighEntopyWords(dic_word_to_clusterEntropy)

    for i in range(len(pred_true_txt_ind_prevPreds)):
        pred_true_text_ind_prevPred = pred_true_txt_ind_prevPreds[i]
        pred = pred_true_text_ind_prevPred[0]
        true = pred_true_text_ind_prevPred[1]
        word_arr = pred_true_text_ind_prevPred[2]
        ind = pred_true_text_ind_prevPred[3]  # originalListind
        prevPred = pred_true_text_ind_prevPred[4]
        cleaned_word_arr = removeWordsByHighEntropy(word_arr, dic_word_to_clusterEntropy, dic_word_to_cluster_indecies,
                                                    high_entropy_words)
        cleaned_pred_true_txt_ind_prevPreds.append([pred, true, cleaned_word_arr, ind, prevPred])
        # print(text+", cleaned="+cleanedText)

    return cleaned_pred_true_txt_ind_prevPreds


def commonWordSims_clusterGroup(word_arr, dic_ClusterGroups):
    dic_lex_Sim_CommonWords = {}
    maxPredLabel_lex = ''
    maxSim_lex = -1000000
    maxCommon_lex = -100000
    minSim_lex = 10000000000
    for label, dicWords_totalWCount in dic_ClusterGroups.items():
        # listWord_arr=extractBySingleIndex(pred_true_txt_ind_prevPredss, 2)
        # merged = list(itertools.chain.from_iterable(listWord_arr))
        # comText=combineDocsToSingle(listStrs)

        dic_words_i = Counter(word_arr)
        totalWCount_i = len(word_arr)
        dic_words_j = dicWords_totalWCount[0]
        totalWCount_j = dicWords_totalWCount[1]
        txtSim, commonCount = computeTextSimCommonWord_WordDic(dic_words_i, dic_words_j, totalWCount_i, totalWCount_j)

        str_label = str(label)
        # txtSim, commonCount=computeTextSimCommonWord_WordArr(word_arr, merged)
        dic_lex_Sim_CommonWords[str_label] = [txtSim, commonCount]
        if maxSim_lex < txtSim:
            maxSim_lex = txtSim
            maxPredLabel_lex = str_label
            maxCommon_lex = commonCount
        if minSim_lex > txtSim:
            minSim_lex = txtSim

    return [dic_lex_Sim_CommonWords, maxPredLabel_lex, maxSim_lex, maxCommon_lex, minSim_lex]


def commonWordSims(word_arr, dic_itemGroups):
    dic_lex_Sim_CommonWords = {}
    for label, pred_true_txt_ind_prevPredss in dic_itemGroups.items():
        listWord_arr = extractBySingleIndex(pred_true_txt_ind_prevPredss, 2)
        merged = list(itertools.chain.from_iterable(listWord_arr))
        # comText=combineDocsToSingle(listStrs)
        txtSim, commonCount = computeTextSimCommonWord_WordArr(word_arr, merged)
        dic_lex_Sim_CommonWords[str(label)] = [txtSim, commonCount]

    return dic_lex_Sim_CommonWords


def semanticSims(text_Vec, c_clusterVecs, c_txtIds):
    dic_semanticSims = {}
    maxPredLabel_Semantic = -1
    maxSim_Semantic = -1000
    minSim_semantic = 1000000000

    for label, centVec in c_clusterVecs.items():
        str_label = label
        cluster_noTexts = 0
        if str_label in c_txtIds.keys():
            cluster_noTexts = len(c_txtIds[str_label])
        elif label in c_txtIds.keys():
            cluster_noTexts = len(c_txtIds[label])

        cluster_noTexts = cluster_noTexts + 1
        sim = compute_sim_value(np.true_divide(centVec, cluster_noTexts), text_Vec)
        dic_semanticSims[str_label] = sim
        if maxSim_Semantic < sim:
            maxSim_Semantic = sim
            maxPredLabel_Semantic = str_label
        if minSim_semantic > sim:
            minSim_semantic = sim

    return [dic_semanticSims, maxPredLabel_Semantic, maxSim_Semantic, minSim_semantic]


def construct_biterms(words):
    bi_terms = []
    for j in range(len(words)):
        for k in range(j + 1, len(words)):
            bi_terms.append(concatWordsSort_Space([words[j], words[k]]))

    if len(bi_terms) == 0:
        return words
    return bi_terms
