from CPost import CPost

from nltk.stem import PorterStemmer
import re
from datetime import datetime
from txt_process_util import getScikitLearn_StopWords
from txt_process_util import loadStopWords

import random


def readlistWholeJsonDataSet(datasetName):
    ps = PorterStemmer()

    # stopWs=getScikitLearn_StopWords()
    stopWs = loadStopWords('stopWords.txt')

    file1 = open(datasetName, "r")
    lines = file1.readlines()
    file1.close()
    list_pred_true_words_index = []
    i = -1
    for line in lines:
        line = line.strip()
        n = eval(line)
        id = str(n['Id']).strip()
        true = str(n['clusterNo']).strip()
        words = str(n['textCleaned']).strip().split(' ')
        words = [ps.stem(w) for w in words]
        words = [w for w in words if not w in stopWs]
        if len(true) == 0 or len(words) == 0:
            continue
        i += 1
        list_pred_true_words_index.append([-1, true, words, i])
    return list_pred_true_words_index


def readStackOverflowDataSetTagTitleBody(inputfile, isStopWord=True, columnsInFile=6, tagIgnore='<r>', randMax=10):
    # stopWs=getScikitLearn_StopWords()
    stopWs = loadStopWords('stopWords.txt')

    ps = PorterStemmer()

    file1 = open(inputfile, "r")
    lines = file1.readlines()
    file1.close()

    list_CPost = []

    i = -1
    for line in lines:

        line = line.strip().lower()
        if len(line) == 0:
            continue
        arr = re.split("\t", line)
        if len(arr) != columnsInFile:
            continue
        true = arr[0].strip()

        if true == '-1' and randMax > 0 and random.randint(0, randMax) != randMax:  # use 1/10 th data
            continue

        postId = arr[1].strip()
        title = arr[2].strip().replace('"', '').replace("\\", '').strip()
        tags = arr[3].strip().replace('"', '').replace("\\", '').strip()
        body = arr[4].strip().replace('"', '').replace("\\", '').strip()
        createtime = arr[5].strip()

        if tagIgnore != '' and len(tagIgnore) > 0:
            tags = tags.replace(tagIgnore, '').strip()
            # print('tagIgnore', tagIgnore, tags)
        if len(tags) == 0:
            tags = arr[3].strip().replace('"', '').replace("\\", '').strip()

        tagText = ' '.join(tags.strip('<').strip('>').split('><')).strip()
        titleText = title
        bodyText = body

        tagWords = tagText.split(' ')
        titleWords = titleText.split(' ')
        bodyWords = bodyText.split(' ')

        tagWords = [ps.stem(w) for w in tagWords]
        titleWords = [ps.stem(w) for w in titleWords]
        bodyWords = [ps.stem(w) for w in bodyWords]

        if isStopWord:
            titleWords = [w for w in titleWords if not w in stopWs]
            bodyWords = [w for w in bodyWords if not w in stopWs]

        if len(true) == 0 or len(tagWords) == 0 or len(titleWords) == 0 or len(bodyWords) == 0 or len(
                postId) == 0 or len(createtime) == 0:
            continue

        i += 1

        postCreatetime = datetime.strptime(createtime.split("t")[0], "%Y-%m-%d")
        list_CPost.append(CPost(-1, int(true), tagWords, titleWords, bodyWords, i, int(postId), postCreatetime))
        # print(-1, int(true), tagWords, titleWords, bodyWords, i, int(postId), postCreatetime)

    return list_CPost


def readStackOverflowDataSetBody(datasetName, isStopWord=True, columnsInFile=6, texttype='tag',
                                 tagIgnore='<c++>'):  # stackoverflow_javascript_true_id_title_tags/ id=postId

    # stopWs=getScikitLearn_StopWords()
    stopWs = loadStopWords('stopWords.txt')

    ps = PorterStemmer()

    file1 = open(datasetName, "r")
    lines = file1.readlines()
    file1.close()
    list_pred_true_words_index_postid_createtime = []
    i = -1
    for line in lines:

        line = line.strip().lower()
        if len(line) == 0:
            continue
        arr = re.split("\t", line)
        if len(arr) != columnsInFile:
            continue
        true = arr[0].strip()
        postId = arr[1].strip()
        title = arr[2].strip().replace('"', '').replace("\\", '').strip()
        tags = arr[3].strip().replace('"', '').replace("\\", '').strip()
        if tagIgnore != '' and len(tagIgnore) > 0:
            tags = tags.replace(tagIgnore, '').strip()
            print('tagIgnore', tagIgnore, tags)
        if len(tags) == 0:
            tags = arr[3].strip().replace('"', '').replace("\\", '').strip()
            # print("restore readStackOverflowDataSetBody", tags)
        body = arr[4].strip().replace('"', '').replace("\\", '').strip()
        createtime = arr[5].strip()

        if texttype == 'tag':
            text = ' '.join(tags.strip('<').strip('>').split('><')).strip()
        elif texttype == 'title':
            text = title
        elif texttype == 'body':
            text = body
        else:
            text = ' '.join(tags.strip('<').strip('>').split('><'))

        text = text.replace('"', '').replace("\\", '').strip()

        if len(true) == 0 or len(text) == 0 or len(postId) == 0 or len(createtime) == 0:
            continue

        words = text.split(' ')
        if texttype == 'tag':
            words = words
        elif isStopWord == True and texttype != 'tag':
            words = [ps.stem(w) for w in words]
            words = [w for w in words if not w in stopWs]
        else:
            words = [ps.stem(w) for w in words]

        if len(words) == 0:
            words = text.split(' ')
            # continue
        i += 1
        # print(words)
        list_pred_true_words_index_postid_createtime.append([-1, true, words, i, postId, createtime])
    return list_pred_true_words_index_postid_createtime


def readStackOverflowDataSet(datasetName, isStopWord=True):  # stackoverflow_javascript_true_id_title_tags/ id=postId

    # stopWs=getScikitLearn_StopWords()
    stopWs = loadStopWords('stopWords.txt')

    ps = PorterStemmer()

    file1 = open(datasetName, "r")
    lines = file1.readlines()
    file1.close()
    list_pred_true_words_index_postid = []
    i = -1
    for line in lines:

        line = line.strip()
        if len(line) == 0:
            continue
        arr = re.split("\t", line)
        if len(arr) != 4:
            continue
        true = arr[0].strip()
        postId = arr[1].strip()
        # text=arr[2].strip() #+' '+' '.join(arr[3].strip('<').strip('>').split('><'))
        text = ' '.join(arr[3].strip('<').strip('>').split('><'))
        text = text.replace('"', '').replace("\\", '').strip()

        if len(true) == 0 or len(text) == 0 or len(postId) == 0:
            continue

        words = text.split(' ')
        if isStopWord == True:
            words = [ps.stem(w) for w in words]
            words = [w for w in words if not w in stopWs]
        else:
            words = [ps.stem(w) for w in words]

        if len(words) == 0:
            words = text.split(' ')
            # continue
        i += 1
        list_pred_true_words_index_postid.append([-1, true, words, i, postId])
    return list_pred_true_words_index_postid


def readStackOverflowDataSetRaw(datasetName):  # stackoverflow_javascript_true_id_title_tags

    file1 = open(datasetName, "r")
    lines = file1.readlines()
    file1.close()
    list_true_id_title_tags = []

    for line in lines:

        line = line.strip()
        if len(line) == 0:
            continue
        arr = re.split("\t", line)
        if len(arr) != 4:
            continue
        true = arr[0].strip()
        postId = arr[1].strip()
        title = arr[2].strip().replace('"', '').replace("\\", '').strip()
        tags = arr[3].strip().replace('"', '').replace("\\", '').strip()

        if len(true) == 0 or len(title) == 0 or len(postId) == 0 or len(tags) == 0:
            continue

        list_true_id_title_tags.append([true, postId, title, tags])
    return list_true_id_title_tags
