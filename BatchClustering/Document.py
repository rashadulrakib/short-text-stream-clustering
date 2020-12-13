isProposed=True

class Document:
    def __init__(self, text, clusterNo, predLabel, wordToIdMap, wordList, documentID):
        self.documentID  = documentID
        self.text = text
        self.clusterNo = clusterNo
        self.predLabel = predLabel 		
        self.wordIdArray = []
        self.wordFreArray = []
        V = len(wordToIdMap)
        wordFreMap = {}
        ws = text.strip().split(' ')
        for w in ws:
            if w not in wordToIdMap:
                V += 1
                wId = V
                wordToIdMap[w] = V
                wordList.append(w)
            else:
                wId = wordToIdMap[w]

            if wId not in wordFreMap:
                wordFreMap[wId] = 1
            else:
                if isProposed==True:
                    wordFreMap[wId] = wordFreMap[wId] + 1
                else:
                    wordFreMap[wId] = 1
        self.wordNum = wordFreMap.__len__()
        w = 0
        for wfm in wordFreMap:
            self.wordIdArray.append(wfm)
            self.wordFreArray.append(wordFreMap[wfm])
            w += 1