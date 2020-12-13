import json
from Document import Document
from txt_process_util import getScikitLearn_StopWords
from txt_process_util import loadStopWords


class DocumentSet:
    '''def __init__(self, dataDir, wordToIdMap, wordList):
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
        print("number of documents is ", self.D)'''

    def __init__(self, dataDir, wordToIdMap, wordList):
        # stopWs = getScikitLearn_StopWords()
        stopWs = loadStopWords('stopWords.txt')

        self.D = 0  # The number of documents
        maxData = 50000
        self.documents = []
        dataDir = r'D:\githubprojects\PyMigrationRecommendation\src\notebooks' \
                  r'\train_stackoverflow_r_true_id_title_tags_body_createtime '
        with open(dataDir) as input:
            line = input.readline()
            while line:

                line = line.strip().lower()

                arr = line.split('\t')
                trueLabel = arr[0]
                postId = int(arr[1])
                title = arr[2]
                tag = arr[3]
                body = arr[4]
                createtime = arr[5]

                tag = ' '.join(tag.strip('<').strip('>').split('><')).strip()

                text = tag
                text = text.strip().replace('"', '').replace("\\", '').strip()

                ws_org = text.strip().split(' ')
                ws = [w for w in ws_org if not w in stopWs]

                if len(ws) == 0:
                    ws = ws_org

                if len(ws) > 0:
                    self.D += 1
                    document = Document(ws, wordToIdMap, wordList, postId, trueLabel)
                    self.documents.append(document)

                line = input.readline()

                # if self.D > maxData:
                #    break

        print("number of documents is ", self.D)
