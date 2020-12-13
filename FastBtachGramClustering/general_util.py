from nltk.stem import PorterStemmer
import re
from txt_process_util import getScikitLearn_StopWords
from txt_process_util import loadStopWords


def readlistWholeJsonDataSet(datasetName, isStopWord=True):
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
        if isStopWord:
            words = [w for w in words if not w in stopWs]
        if len(words) == 0:
            words = str(n['textCleaned']).strip().split(' ')
        if len(true) == 0 or len(words) == 0:
            continue
        i += 1
        list_pred_true_words_index.append([-1, true, words, i])
    return list_pred_true_words_index


def readStackOverflowDataSetBody(datasetName, isStopWord=True, columns=6, texttype='tag',
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
        if len(arr) != columns:
            continue
        true = arr[0].strip()
        postId = arr[1].strip()
        title = arr[2].strip().replace('"', '').replace("\\", '').strip()
        tags = arr[3].strip().replace('"', '').replace("\\", '').strip()
        if tagIgnore != '' and len(tagIgnore) > 0:
            tags = tags.replace(tagIgnore, '').strip()
            # print('tagIgnore', tagIgnore, tags)
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
        elif isStopWord and texttype != 'tag':
            words = [ps.stem(w) for w in words]
            words = [w for w in words if not w in stopWs]
        else:
            words = [ps.stem(w) for w in words]

        if len(words) == 0:
            words = text.split(' ')
            # continue
        i += 1
        # print(words)
        # if i > 10000:
        #    break
        list_pred_true_words_index_postid_createtime.append([-1, true, words, i, postId, createtime])
    return list_pred_true_words_index_postid_createtime


def readStackOverflowDataSetTagTitleBody(datasetName, isStopWord=True, columns=6, texttype='tag',
                                         tagIgnore='<c++>'):  # stackoverflow_javascript_true_id_title_tags/ id=postId

    # stopWs=getScikitLearn_StopWords()
    stopWs = loadStopWords('stopWords.txt')

    ps = PorterStemmer()

    file1 = open(datasetName, "r")
    lines = file1.readlines()
    file1.close()
    list_pred_true_words_index_postid_createtime_tag_title_body = []
    i = -1
    for line in lines:

        line = line.strip().lower()
        if len(line) == 0:
            continue
        arr = re.split("\t", line)
        if len(arr) != columns:
            continue
        true = arr[0].strip()
        postId = arr[1].strip()
        title = arr[2].strip().replace('"', '').replace("\\", '').strip()
        tags = arr[3].strip().replace('"', '').replace("\\", '').strip()

        body = arr[4].strip().replace('"', '').replace("\\", '').strip()
        createtime = arr[5].strip()

        if len(true) == 0 or len(postId) == 0 or len(createtime) == 0:
            continue

        i += 1
        # print(words)
        # if i > 10000:
        #    break

        tag_words = ' '.join(tags.strip('<').strip('>').split('><')).strip().split(' ')
        title_words = title.split(' ')
        body_words = body.split(' ')

        title_words = [ps.stem(w) for w in title_words]
        body_words = [ps.stem(w) for w in body_words]

        if isStopWord:
            title_words = [w for w in title_words if not w in stopWs]
            body_words = [w for w in body_words if not w in stopWs]
        if len(title_words) == 0:
            title_words = title.split(' ')
        if len(body_words) == 0:
            body_words = body.split(' ')

        list_pred_true_words_index_postid_createtime_tag_title_body.append(
            [-1, true, tag_words, i, postId, createtime, tag_words, title_words, body_words])
    return list_pred_true_words_index_postid_createtime_tag_title_body


def readStackOverflowDataSetRaw(datasetName, columns=6):  # stackoverflow_javascript_true_id_title_tags

    file1 = open(datasetName, "r")
    lines = file1.readlines()
    file1.close()
    list_true_id_title_tags_body_createtime = []

    for line in lines:

        line = line.strip()
        if len(line) == 0:
            continue
        arr = re.split("\t", line)
        if len(arr) != columns:
            continue
        true = arr[0].strip()
        postId = arr[1].strip()
        title = arr[2].strip().replace('"', '').replace("\\", '').strip()
        tags = arr[3].strip().replace('"', '').replace("\\", '').strip()
        body = arr[4].strip().replace('"', '').replace("\\", '').strip()
        createtime = arr[5].strip()
        # print(createtime.split('\s'), len(createtime.split('\s')))

        # if '0         2008-09-16' in createtime:
        #  print(createtime)
        if len(createtime.split(' ')) > 1:
            print('wrong', createtime)
            continue

        if len(true) == 0 or len(title) == 0 or len(postId) == 0 or len(tags) == 0 or len(body) == 0 or len(
                createtime) == 0:
            continue

        list_true_id_title_tags_body_createtime.append([true, postId, title, tags, body, createtime])
    return list_true_id_title_tags_body_createtime
