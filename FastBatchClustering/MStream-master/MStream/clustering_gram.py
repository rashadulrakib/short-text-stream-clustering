from txt_process_util import generateGramsConsucetive
from clustering_gram_util import populateNgramStatistics
from clustering_gram_util import filterGrams
from clustering_gram_util import removeCommonDocs
from evaluation_util import evaluateByGramUsingDic


def cluster_gram_freq(list_docs, minGSize, maxGSize):
    dic_ngram__docs = {}

    # set_docIds = [document.documentID for document in list_docs]
    # set_docIds = set(set_docIds)

    for document in list_docs:
        words = document.text
        grams = generateGramsConsucetive(words, minGSize, maxGSize)

        for gram in grams:
            dic_ngram__docs.setdefault(gram, []).append(document)

        # print('cluster_gram_freq', words, document.documentID, grams, len(dic_ngram__docs))

    gram_std, gram_mean, gram_max, gram_min = populateNgramStatistics(dic_ngram__docs, 1)
    print('gram_std, gram_mean, gram_max, gram_min', gram_std, gram_mean, gram_max, gram_min,
          'before len(dic_ngram__docs)', len(dic_ngram__docs))

    minClusterSize = gram_mean + 0*gram_std
    dic_filtered_ngram__docs = filterGrams(dic_ngram__docs, minClusterSize)
    print('after len(dic_filtered_ngram__docs)', len(dic_filtered_ngram__docs))
    dic_removed_common__docs = removeCommonDocs(dic_filtered_ngram__docs)
    print('after dic_removed_common__docs', len(dic_removed_common__docs))
    print('###total docs in batch=', len(list_docs))
    dic_docId__cluster = evaluateByGramUsingDic(dic_removed_common__docs)

    del dic_removed_common__docs
    del dic_filtered_ngram__docs
    del dic_ngram__docs

    # return [dic_docId__cluster, set_docIds - dic_docId__cluster.keys()]
    return dic_docId__cluster
