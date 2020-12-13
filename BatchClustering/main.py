#need to implement removeOutlierConnectedComponentLexicalIndex
#when merging single text to a cluster.
#caculate the closest similarity of each single text to th eclusters.
#if the similarity if the single text is greater than mena and standard deviation of the similarities, then add it to a closest clustyer, other wise leave it in a single text cluster 

#python3 -m venv env
#source ./env/bin/activate 
#python -m pip install pyod
#https://github.com/googlesamples/assistant-sdk-python/issues/236


#remove high entropy words from CF of MStreamF model also

############-coling-alogorithm-###########

#1.cluster portion of texts of a batch by n-gram
#2.initialize gibbs sampling of Mstream (bi-gram feature)
#3.remove the bi-grams from clusters that appear in differnt clusters using entropy
#4.smaple the rest of the texts using gibbs sampling
#5.remove outliers by connected component (bi-gram features)
#6.assign outliers by similarity (s1=t1<->c1, s2=t1<->c2, s3=t1<->c3, s4=t1<->c4)
#  m=(s1+..+s4), std=(s1..s4), max=(s1..s4), if max>m+std, assign to c1,or to a new c5
#7.cluster orders (1,2,3,4,5..) cluster sizes(10,5,6,2,4..). m_o,std_o=(cluster orders), m_s,std_s=(cluster sizes), if order < abs(m_o-std_o) and size <abs(m_s-std_s): delete that cluster from cluster models (mstream, n-gram clusters)

############-end coling-alogorithm-###########
#https://stackoverflow.com/questions/1125968/how-do-i-force-git-pull-to-overwrite-local-files



from MStream import MStream
import json
import time
from word_vec_extractor import extractAllWordVecs
from general_util import readlistWholeJsonDataSet
from word_vec_extractor import extractAllWordVecsPartial
from word_vec_extractor import extractAllWordVecsPartialStemming

dataDir = "data/"
outputPath = "result/"

#dataset='NT-mstream-long' #b 10, itr 2, nmi=89, acc=91, 166 sec
#dataset='NT-mstream-long1'  #b 10, itr 2, nmi=78, acc=66, 310 sec, (NT)
#dataset='NTSB_NL-mstream-shfl1'
#dataset='NTSB-mstream-long1'
#dataset='NTSB-mstream-long'
#dataset='NTS-mstream-long'
#dataset='NTS-mstream-long1' #(NTS)
#dataset='Tweets-T-id-ordered'
#dataset = "Tweets-T"
#dataset = "Tweets1"
#dataset='News-T-id-ordered'
#dataset = "News"
#dataset = "News-T"
#dataset="Stackoverflow-mstream"
#dataset="NTS-mstream"
#dataset="Biomedical-mstream"
#dataset='super_user'
#dataset='stackoverflow_large'
dataset='stackoverflow_large_tweets-T_news-T_suff'

timefil = "timefil"
MaxBatch = 2 # The number of saved batches + 1
batchSize=2000

alpha = 0.03 #0.03 #0.002
beta = 0.03 #0.03
iterNum = 0
sampleNum = 1
wordsInTopicNum = 2
K = 0 


gloveFile = "glove.6B.50d.txt"




embedDim=50

list_pred_true_words_index=readlistWholeJsonDataSet("data/"+dataset)
AllBatchNum = int(len(list_pred_true_words_index)/batchSize)+1 #4 The number of batches you want to devided the dataset to
print("AllBatchNum", AllBatchNum)

all_words=[]
for item in list_pred_true_words_index:
  all_words.extend(item[2])
all_words=list(set(all_words))  

#need to tokenize all_words by porter steamer
  

wordVectorsDic={}
#wordVectorsDic = extractAllWordVecs(gloveFile, embedDim)
#wordVectorsDic=extractAllWordVecsPartial(gloveFile, embedDim, all_words)
#wordVectorsDic=extractAllWordVecsPartialStemming(gloveFile, embedDim, all_words)

def runMStreamF(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    mstream.getDocuments()
    for sampleNo in range(1, sampleNum+1):
        print("SampleNo:"+str(sampleNo))
        mstream.runMStreamF(sampleNo, outputPath, wordVectorsDic)

def runMStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    mstream.getDocuments()
    for sampleNo in range(1, sampleNum+1):
        print("SampleNo:"+str(sampleNo))	
        mstream.runMStream(sampleNo, outputPath, wordVectorsDic)
    	
def runWithAlphaScale(beta, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []

    p = 0.01
    while p <= 0.051:
        alpha = p
        parameters.append(p)
        print("alpha:", alpha, "\tp:", p)
        mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo)
            startTime = time.time()
            mstream.runMStreamF(sampleNo, outputPath)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        p += 0.001

    fileParameters = "MStreamDiffAlpha" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + \
                     str(sampleNum) + "beta" + str(round(beta, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithBetas(alpha, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    beta = 0.01
    while beta <= 0.0501:
        parameters.append(beta)
        print("beta:", beta)
        mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStreamF(sampleNo, outputPath)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        beta += 0.002
    fileParameters = "MStreamDiffBeta" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithNiters(K, MaxBatch, AllBatchNum, alpha, beta, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    iterNum = 0
    while iterNum <= 30.01:
        parameters.append(iterNum)
        print("iterNum:", iterNum)
        mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStreamF(sampleNo, outputPath)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        iterNum += 1
    fileParameters = "MStreamDiffIter" + "K" + str(K) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithBatchNum(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    BatchNum = 5
    while BatchNum <= 30.1:
        parameters.append(BatchNum)
        print("BatchNum:", BatchNum)
        mstream = MStream(K, MaxBatch, BatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStreamF(sampleNo, outputPath)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        BatchNum += 1
    fileParameters = "MStreamDiffBatchNum" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithMaxBatch(K, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    MaxBatch = 1
    while MaxBatch <= 16.1:
        parameters.append(MaxBatch)
        print("MaxBatch:", MaxBatch)
        mstream = MStream(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStreamF(sampleNo, outputPath)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        MaxBatch += 1
    fileParameters = "MStreamDiffMaxBatch" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

if __name__ == '__main__':

    runMStream(K, AllBatchNum, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    #runMStreamF(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithAlphaScale(beta, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithBetas(alpha, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithNiters(K, MaxBatch, AllBatchNum, alpha, beta, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithBatchNum(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithMaxBatch(K, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
