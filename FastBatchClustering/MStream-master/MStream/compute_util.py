from scipy.spatial.distance import cosine
import numpy as np


def concatenateDCTvecs(wordVecs2dList, k):
    dim = len(wordVecs2dList[0])
    concatVec = [0] * k * dim
    nparr = np.array(wordVecs2dList)
    transposeMat = nparr.transpose()
    minK = min(len(wordVecs2dList), k)
    dimIndex = -1
    for transRow in transposeMat:  # transRow=each dimension in N x D matrix
        dimIndex = dimIndex + 1
        dctarr = dct(transRow, 2)
        for i in range(minK):
            concatVec[i * dim + dimIndex] = dctarr[i]

    # print(concatVec)
    return concatVec


def sumListArr(dct2dList):
    sumVec = np.sum(dct2dList, axis=0)
    return sumVec


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


def computeSimBtnList(txtIndsi, txtIndsj):
    if len(txtIndsj) < len(txtIndsi):
        temp = txtIndsj
        txtIndsj = txtIndsi
        txtIndsi = temp

    common = 0
    for ind in txtIndsi:
        if ind in txtIndsj:
            common = common + 1

    return 2 * common / (len(txtIndsi) + len(txtIndsj))


def computeTextSimCommonWord_WordArr(txt_i_wordArr, txt_j_wordArr):
    txt_i_len = len(txt_i_wordArr)
    txt_j_len = len(txt_j_wordArr)

    words_i = collections.Counter(txt_i_wordArr)  # assume words_i small
    words_j = collections.Counter(txt_j_wordArr)

    text_sim, commonCount = computeTextSimCommonWord_WordDic(words_i, words_j, txt_i_len, txt_j_len)

    return [text_sim, commonCount]


def computeTextSimCommonWord(txt_i, txt_j):
    txt_i_wordArr = word_tokenize(txt_i)
    txt_j_wordArr = word_tokenize(txt_j)

    text_sim, commonCount = computeTextSimCommonWord_WordArr(txt_i_wordArr, txt_j_wordArr)
    return [text_sim, commonCount]


def ComputeEntropy(listValues):
    countsDic = collections.Counter(listValues)
    countValueList = list(countsDic.values())
    totalItems = len(listValues)
    entropy = 0
    for value in countValueList:
        entropy = entropy + -1 * (value / totalItems) * math.log(value / totalItems)

    return entropy


def compute_sim_matrix(txtVecs):
    rows = len(txtVecs)
    sim_matrix = [[0 for x in range(rows)] for y in range(rows)]

    for i in range(rows):
        for j in range(i + 1, rows, 1):
            sim_matrix[i][j] = compute_sim_value(txtVecs[i], txtVecs[j])
            sim_matrix[j][i] = sim_matrix[i][j]

    return sim_matrix


def compute_sim_value(vecarr1, vecarr2):
    if np.sum(np.array(vecarr1)) == 0 or np.sum(np.array(vecarr2)) == 0:
        return 0
        # sim_value = 1- cosine(vecarr1, vecarr2) #cosine=distance
    sim_value = 1 - cosine(vecarr1, vecarr2)  # cosine=distance
    return sim_value


def compute_mean_sd(numbers):
    meanVal = 0
    sdVal = 0
    sumVal = 0
    for num in numbers:
        sumVal = sumVal + num

    meanVal = sumVal / len(numbers)
    varainceSumVal = 0
    for num in numbers:
        varainceSumVal = varainceSumVal + (num - meanVal) * (num - meanVal)

    sdVal = math.sqrt(varainceSumVal / len(numbers))

    return [meanVal, sdVal]


def MultiplyTwoSetsOneToOne(set1, set2):
    if len(set1) != len(set2):
        print("len_set1=" + len(set1) + ",len_set2=" + len(set2))
        return set1

    merged = []
    for i in range(len(set1)):
        s1 = set1[i]
        s2 = set2[i]
        merged.append(s1 * s2)

    return merged


def compute_row_sim_I(txtVec, txtVecs, skipRowId):
    rowSimsToI = []
    rowSimsToI_ExcpetSelf = []
    for i in range(len(txtVecs)):
        if i == skipRowId:
            rowSimsToI.append(1)
            continue
        simVal = compute_sim_value(txtVecs[i], txtVec)
        rowSimsToI.append(simVal)
        rowSimsToI_ExcpetSelf.append(simVal)

    return [rowSimsToI, rowSimsToI_ExcpetSelf]


def compute_row_lexsimCommonCount_I(text, texts, skipRowId):
    rowLexSims_CommonCountsToI = []
    rowLexSims_CommonCountsToI_ExcpetSelf = []
    for i in range(len(texts)):
        if i == skipRowId:
            rowLexSims_CommonCountsToI.append([1, len(texts[i])])
            continue
        simVal, commonCount = computeTextSimCommonWord(texts[i], text)
        rowLexSims_CommonCountsToI.append([simVal, commonCount])
        rowLexSims_CommonCountsToI_ExcpetSelf.append([simVal, commonCount])
    return [rowLexSims_CommonCountsToI, rowLexSims_CommonCountsToI_ExcpetSelf]


def computeClose_Far_vec(centVec, X):
    closestVec = []
    farVec = []
    closeSim = -10000
    farSim = 10000

    for i in range(len(X)):
        vec = X[i]
        sim = 1 - cosine(vec, centVec)  # if sim=1, close vec
        if sim >= closeSim:
            closeSim = sim
            closestVec = vec
        if sim <= farSim:
            farSim = sim
            farVec = vec

    return [closestVec, farVec]


def findCloseCluster_GramKey_lexical(keys_list, word_arr, minMatch):
    closeKey_Lexical = None
    maxCommonLength = 0

    for key in keys_list:
        set1 = set(key.split(' '))
        set2 = set(word_arr)
        common = set1.intersection(set2)
        if len(common) >= minMatch and len(common) > maxCommonLength:
            maxCommonLength = len(common)
            closeKey_Lexical = key

    return closeKey_Lexical


def findCloseCluster_GramKey_Semantic(keys_list, word_arr, minMatch, wordVectorsDic, euclidean=True):
    closeKey_Semantic = None
    sent_vec = generate_sent_vecs_toktextdata([word_arr], wordVectorsDic, 300)[0]
    min_dist = sys.float_info.max
    max_sim = 0
    for key in keys_list:
        key_words = key.split(' ')
        set1 = set(key_words)
        set2 = set(word_arr)
        common = set1.intersection(set2)
        key_vec = generate_sent_vecs_toktextdata([key_words], wordVectorsDic, 300)[0]
        # eu_dist=0
        # if euclidean==True:
        #  eu_dist=distance.euclidean(sent_vec, key_vec)
        # else:
        eu_dist = cosine(sent_vec, key_vec)  # cosine=distance
        sim = 1 - eu_dist
        # if len(common)>=minMatch and min_dist>eu_dist:
        if len(common) >= minMatch and max_sim < sim:
            # min_dist=eu_dist
            max_sim = sim
            closeKey_Semantic = key

    return [closeKey_Semantic, max_sim]


def computeTextClusterSimilarity_framework(oCPostProcessed, oCFVector, oCSimilarityFlgas):
    tBiFreTag = oCPostProcessed.txtBitermsFreqs_Tag
    tbi_len_Tag = oCPostProcessed.bi_terms_len_Tag
    tBiFreTitle = oCPostProcessed.txtBitermsFreqs_Title
    tbi_len_Title = oCPostProcessed.bi_terms_len_Title
    tBiFreBody = oCPostProcessed.txtBitermsFreqs_Body
    tbi_len_Body = oCPostProcessed.bi_terms_len_Body
    t_VecTag = oCPostProcessed.text_VecTag
    t_VecTitle = oCPostProcessed.text_VecTitle
    t_VecBody = oCPostProcessed.text_VecBody

    totalSim = 0

    cBiFreTag = oCFVector.txtBitermsFreqs_Tag
    cbi_len_Tag = oCFVector.bi_terms_len_Tag
    cBiFreTitle = oCFVector.txtBitermsFreqs_Title
    cbi_len_Title = oCFVector.bi_terms_len_Title
    cBiFreBody = oCFVector.txtBitermsFreqs_Body
    cbi_len_Body = oCFVector.bi_terms_len_Body
    c_VecTag = oCFVector.text_VecTag
    c_VecTitle = oCFVector.text_VecTitle
    c_VecBody = oCFVector.text_VecBody

    if oCSimilarityFlgas.isTagSim:
        tag_sim, tagCommCount = computeTextSimCommonWord_WordDic(tBiFreTag, cBiFreTag, tbi_len_Tag, cbi_len_Tag)
        tag_sim = tag_sim * oCSimilarityFlgas.tagWeight
        totalSim += tag_sim

    if oCSimilarityFlgas.isTitleSim:
        title_sim, titleCommCount = computeTextSimCommonWord_WordDic(tBiFreTitle, cBiFreTitle, tbi_len_Title,
                                                                     cbi_len_Title)
        title_sim = title_sim * oCSimilarityFlgas.titleWeight
        totalSim += title_sim

    if oCSimilarityFlgas.isBodySim:
        body_sim, bodyCommCount = computeTextSimCommonWord_WordDic(tBiFreBody, cBiFreBody, tbi_len_Body,
                                                                   cbi_len_Body)
        body_sim = body_sim * oCSimilarityFlgas.bodyWeight
        totalSim += body_sim

    return totalSim
