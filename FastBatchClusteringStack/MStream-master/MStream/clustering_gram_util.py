import statistics


def populateNgramStatistics(dic_gram_to_textInds, minTxtIndsForNgram=1):
    ordered_keys_gram_to_textInds = sorted(dic_gram_to_textInds, key=lambda key: len(dic_gram_to_textInds[key]))
    txtIndsSize = []
    for key in ordered_keys_gram_to_textInds:
        # print(key, dic_gram_to_textInds[key])
        if len(dic_gram_to_textInds[key]) > minTxtIndsForNgram: txtIndsSize.append(len(dic_gram_to_textInds[key]))

    size_mean = 0
    size_max = 0
    size_min = 0
    if len(txtIndsSize) >= 1:
        size_mean = statistics.mean(txtIndsSize)
        size_max = max(txtIndsSize)
        size_min = min(txtIndsSize)
    size_std = size_mean
    if len(txtIndsSize) >= 2:
        size_std = statistics.stdev(txtIndsSize)

    return [size_std, size_mean, size_max, size_min]


def filterGrams(dic_ngram__docs, minSize):
    dic_filtered_ngram__docs = {}

    for gram, docs in dic_ngram__docs.items():
        size = len(docs)
        if size < minSize:
            continue
        dic_filtered_ngram__docs[gram] = docs

    return dic_filtered_ngram__docs


def removeCommonDocs(dic_filtered_ngram__docs):
    dic_removed_common__docs = {}

    dic_dup__docId = {}
    for gram, docs in dic_filtered_ngram__docs.items():
        for doc in docs:
            if doc.documentID not in dic_dup__docId:
                dic_dup__docId[doc.documentID] = 0
            dic_dup__docId[doc.documentID] += 1

    for gram, docs in dic_filtered_ngram__docs.items():
        filter_docs = []
        for doc in docs:
            if dic_dup__docId[doc.documentID] == 1:
                filter_docs.append(doc)

        if len(filter_docs) > 0:
            dic_removed_common__docs[gram] = filter_docs

    return dic_removed_common__docs
