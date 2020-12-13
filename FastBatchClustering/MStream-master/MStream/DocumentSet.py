import json
from Document import Document
from txt_process_util import getScikitLearn_StopWords



class DocumentSet:
    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0  # The number of documents
        # self.clusterNoArray = []
        self.documents = []
        with open(dataDir) as input:
            line = input.readline()
            while line:
                self.D += 1
                obj = json.loads(line)
                text = obj['textCleaned']
                trueLabel = obj['clusterNo']
                document = Document(text, wordToIdMap, wordList, int(obj['Id']), trueLabel)
                self.documents.append(document)
                line = input.readline()
        print("number of documents is ", self.D)


    '''def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0  # The number of documents
        maxData = 10000
        self.documents = []
        dataDir = r'D:\githubprojects\PyMigrationRecommendation\src\notebooks' \
                  r'\train_stackoverflow_r_true_id_title_tags_body_createtime '
        with open(dataDir) as input:
            line = input.readline()
            while line:
                self.D += 1

                arr = line.split('\t')
                trueLabel = arr[0]
                postId = int(arr[1])
                title = arr[2]
                tag = arr[3]
                body = arr[4]
                createtime = arr[5]

                document = Document(title, wordToIdMap, wordList, postId, trueLabel)
                self.documents.append(document)
                line = input.readline()

                if self.D > maxData:
                    break

        print("number of documents is ", self.D)'''
