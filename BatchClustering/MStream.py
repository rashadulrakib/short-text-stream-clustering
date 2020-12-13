from DocumentSet import DocumentSet
from Model import Model

class MStream:
    dataDir = "data/"

    def __init__(self, K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
        self.K = K
        self.MaxBatch = MaxBatch
        self.AllBatchNum = AllBatchNum
        self.alpha = alpha
        self.beta = beta
        self.iterNum = iterNum
        self.dataset = dataset
        self.timefil = timefil
        self.sampleNum = sampleNum
        self.wordsInTopicNum = wordsInTopicNum
        self.wordList = []
        self.wordToIdMap = {}

    def getDocuments(self):
        self.documentSet = DocumentSet(self.dataDir + self.dataset, self.wordToIdMap, self.wordList)
        self.V = self.wordToIdMap.__len__()
	
        print("self.V=all the words in all documents") #rakib 
        print(self.V) #rakib all the words in all documents	

    def runMStream(self, sampleNo, outputPath, wordVectorsDic):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 3)) + \
                        "beta" + str(round(self.beta, 3)) + "BatchNum" + str(self.AllBatchNum) + "BatchSaved" + str(self.MaxBatch)
        print(ParametersStr)						
        model = Model(self.K, self.MaxBatch, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum, self.dataDir + self.timefil)
        print("before call run_MStream, self.K="+ str(self.K)+",self.iterNum="+str(self.iterNum)+", self.V="+str(self.V))					  
        model.run_MStream(self.documentSet, outputPath, self.wordList, self.AllBatchNum, wordVectorsDic)
        #rakib		
        #print("final dic len="+str(len(model.docsPerCluster)))
        #for key, value in model.docsPerCluster.items():
         #print("cluster id="+str(key)+", items="+str(len(value)))		

    def runMStreamF(self, sampleNo, outputPath, wordVectorsDic):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 3)) + \
                        "beta" + str(round(self.beta, 3)) + "BatchNum" + str(self.AllBatchNum) + "BatchSaved" + str(self.MaxBatch)
        model = Model(self.K, self.MaxBatch, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum, self.dataDir + self.timefil)
        model.run_MStreamF(self.documentSet, outputPath, self.wordList, self.AllBatchNum, wordVectorsDic)

