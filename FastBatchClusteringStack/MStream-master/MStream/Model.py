import random
import os
import time
import json
import copy
from datetime import datetime
import statistics

from read_pred_true_text import ReadPredTrueText
from evaluation import Evaluate_old
from ClusteringUtil import findTargetClusters
from clustering_gram import cluster_gram_freq

threshold_fix = 1.1
threshold_init = 0

clusterWriterFile = 'out_r_tag_pred_true_text.txt'
clusterWriter = open(clusterWriterFile, 'w')
minGSize = 2
maxGSize = 2


class Model:

    def __init__(self, K, Max_Batch, V, iterNum, alpha0, beta, dataset, ParametersStr, sampleNo, wordsInTopicNum,
                 timefil):
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.alpha0 = alpha0
        self.beta = beta
        self.K = K
        self.Kin = K
        self.V = V
        self.iterNum = iterNum
        self.beta0 = float(V) * float(beta)
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)
        self.Max_Batch = Max_Batch  # Max number of batches we will consider
        self.phi_zv = []

        self.batchNum2tweetID = {}  # batch num to tweet id
        self.batchNum = 1  # current batch number
        self.readTimeFil(timefil)

    def run_MStream(self, documentSet, outputPath, wordList, AllBatchNum):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently #may not be used
        self.K_current = copy.deepcopy(self.K)  # the number of cluster containing documents currently
        # self.BatchSet = {} # No need to store information of each batch
        self.word_current = {}  # Store word-IDs' list of each batch
        self.f_zs = {}  # feature to cluster z. t1={w1, w2} w1->c1, w2->c1

        # Get batchNum2tweetID by AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)

        t11 = datetime.now()

        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            dic_docId__cluster = self.customInitialize(documentSet)
            # self.intialize(documentSet)
            # self.gibbsSampling(documentSet)
            self.gibbsSamplingPartial(documentSet, dic_docId__cluster)
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")
            self.deleteClusters()

            # print(self.f_zs)

        t12 = datetime.now()
        t_diff = t12 - t11
        print("total time diff secs=", t_diff.seconds)

        listtuple_pred_true_text = ReadPredTrueText(clusterWriterFile)
        Evaluate(listtuple_pred_true_text)
        clusterWriter.close()

    def run_MStreamF(self, documentSet, outputPath, wordList, AllBatchNum):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K)  # the number of cluster containing documents currently
        self.BatchSet = {}  # Store information of each batch
        self.word_current = {}  # Store word-IDs' list of each batch
        self.f_zs = {}  # feature to list of clusters. t1={w1, w2} w1->c1, w2->c1

        # Get batchNum2tweetID by AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)

        t11 = datetime.now()
        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            if self.batchNum <= self.Max_Batch:
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)

                dic_docId__cluster = self.customInitialize(documentSet)
                # self.intialize(documentSet)
                self.gibbsSamplingPartial(documentSet, dic_docId__cluster)

                # self.gibbsSampling(documentSet)
            else:
                # remove influence of batch earlier than Max_Batch
                self.D -= self.BatchSet[self.batchNum - self.Max_Batch]['D']
                for cluster in self.m_z:
                    if cluster in self.BatchSet[self.batchNum - self.Max_Batch]['m_z']:
                        self.m_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['m_z'][cluster]
                        self.n_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_z'][cluster]
                        for word in self.n_zv[cluster]:
                            if word in self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster]:
                                self.n_zv[cluster][word] -= \
                                    self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster][word]
                for cluster in range(self.K):
                    self.checkEmpty(cluster)
                self.BatchSet.pop(self.batchNum - self.Max_Batch)
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)

                dic_docId__cluster = self.customInitialize(documentSet)
                # self.intialize(documentSet)

                # self.gibbsSampling(documentSet)
                self.gibbsSamplingPartial(documentSet, dic_docId__cluster)
            # get influence of only the current batch (remove other influence)
            self.BatchSet[self.batchNum - 1]['D'] = self.D - self.BatchSet[self.batchNum - 1]['D']
            for cluster in self.m_z:
                if cluster not in self.BatchSet[self.batchNum - 1]['m_z']:
                    self.BatchSet[self.batchNum - 1]['m_z'][cluster] = 0
                if cluster not in self.BatchSet[self.batchNum - 1]['n_z']:
                    self.BatchSet[self.batchNum - 1]['n_z'][cluster] = 0
                self.BatchSet[self.batchNum - 1]['m_z'][cluster] = self.m_z[cluster] - \
                                                                   self.BatchSet[self.batchNum - 1]['m_z'][cluster]
                self.BatchSet[self.batchNum - 1]['n_z'][cluster] = self.n_z[cluster] - \
                                                                   self.BatchSet[self.batchNum - 1]['n_z'][cluster]
                if cluster not in self.BatchSet[self.batchNum - 1]['n_zv']:
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster] = {}
                for word in self.n_zv[cluster]:
                    if word not in self.BatchSet[self.batchNum - 1]['n_zv'][cluster]:
                        self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = 0
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = self.n_zv[cluster][word] - \
                                                                              self.BatchSet[self.batchNum - 1]['n_zv'][
                                                                                  cluster][word]
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")
            # print(self.f_zs)
            self.deleteClusters()

        t12 = datetime.now()
        t_diff = t12 - t11
        print("total time diff secs=", t_diff.seconds)

        listtuple_pred_true_text = ReadPredTrueText(clusterWriterFile)
        Evaluate_old(listtuple_pred_true_text)
        clusterWriter.close()

    def readTimeFil(self, timefil):
        try:
            with open(timefil) as timef:
                for line in timef:
                    buff = line.strip().split(' ')
                    if buff == ['']:
                        break
                    self.batchNum2tweetID[self.batchNum] = int(buff[1])
                    self.batchNum += 1
            self.batchNum = 1
        except:
            print("No timefil!")
        # print("There are", self.batchNum2tweetID.__len__(), "time points.\n\t", self.batchNum2tweetID)

    def getAveBatch(self, documentSet, AllBatchNum):
        self.batchNum2tweetID.clear()
        temp = self.D_All / AllBatchNum
        _ = 0
        count = 1
        for d in range(self.D_All):
            if _ < temp:
                _ += 1
                continue
            else:
                document = documentSet.documents[d]
                documentID = document.documentID
                self.batchNum2tweetID[count] = documentID
                count += 1
                _ = 0
        self.batchNum2tweetID[count] = -1

    '''
    improvements
    At initialization, V_current has another choices:
    set V_current is the number of words analyzed in the current batch and the number of words in previous batches,
        which means beta0s of each document are different at initialization.
    Before we process each batch, alpha will be recomputed by the function: alpha = alpha0 * docStored
    '''

    # Compute beta0 for every batch
    def getBeta0(self):
        Words = []
        if self.batchNum < 5:
            for i in range(1, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        if self.batchNum >= 5:
            for i in range(self.batchNum - 4, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        return (float(len(list(set(Words)))) * float(self.beta))

    def customInitialize(self, documentSet):
        self.word_current[self.batchNum] = []
        list_docs = []
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID

            if documentID != self.batchNum2tweetID[self.batchNum]:
                self.D += 1
                list_docs.append(document)
            else:
                break

        dic_docId__cluster = cluster_gram_freq(list_docs, minGSize, maxGSize)
        # dic_docId__cluster, cluster is integer

        cluster_offset = max(self.K, self.K_current)

        self.beta0 = self.getBeta0()
        self.alpha = self.alpha0 * self.D
        print(
            "\t" + str(self.D) + " documents will be analyze. alpha is" + " %.2f." % self.alpha + "\n\tInitialization.",
            end='\n')

        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID

            # This method is getting beta0 before each document is initialized
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                if wordNo not in self.word_current[self.batchNum]:
                    self.word_current[self.batchNum].append(wordNo)
            self.beta0 = self.getBeta0()
            if self.beta0 <= 0:
                print("Wrong V!")
                exit(-1)

            if documentID != self.batchNum2tweetID[self.batchNum]:

                cluster = self.customSampleCluster(d, document, documentID, 0, dic_docId__cluster, cluster_offset)

                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break

        return dic_docId__cluster

    def intialize(self, documentSet):
        self.word_current[self.batchNum] = []
        print('intialize: self.currentDoc, self.D_All', self.currentDoc, self.D_All)
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID

            if documentID != self.batchNum2tweetID[self.batchNum]:
                self.D += 1
            else:
                break
        self.beta0 = self.getBeta0()
        self.alpha = self.alpha0 * self.D
        print(
            "\t" + str(self.D) + " documents will be analyze. alpha is" + " %.2f." % self.alpha + "\n\tInitialization.",
            end='\n')
        print('2nd intialize: self.currentDoc, self.D_All', self.currentDoc, self.D_All)
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID

            # This method is getting beta0 before each document is initialized
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                if wordNo not in self.word_current[self.batchNum]:
                    self.word_current[self.batchNum].append(wordNo)
            self.beta0 = self.getBeta0()
            if self.beta0 <= 0:
                print("Wrong V!")
                exit(-1)

            if documentID != self.batchNum2tweetID[self.batchNum]:
                cluster = self.sampleCluster(d, document, documentID, 0)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break

    def gibbsSamplingPartial(self, documentSet, dic_docId__cluster):
        for i in range(self.iterNum):
            print("\titer is ", i + 1, end='\n')
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                if documentID in dic_docId__cluster:
                    continue
                cluster = self.z[documentID]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                if i != self.iterNum - 1:  # if not last iteration
                    cluster = self.sampleCluster(d, document, documentID, 0)
                elif i == self.iterNum - 1:  # if last iteration
                    cluster = self.sampleCluster(d, document, documentID, 1)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre

        # return

    def gibbsSampling(self, documentSet):
        for i in range(self.iterNum):
            print("\titer is ", i + 1, end='\n')
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                cluster = self.z[documentID]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                if i != self.iterNum - 1:  # if not last iteration
                    cluster = self.sampleCluster(d, document, documentID, 0)
                elif i == self.iterNum - 1:  # if last iteration
                    cluster = self.sampleCluster(d, document, documentID, 1)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre
        return

    def customSampleCluster(self, d, document, documentID, isLast, dic_docId__cluster={}, cluster_offset=1):

        if documentID in dic_docId__cluster:
            kChoosed = dic_docId__cluster[documentID] + cluster_offset

            for f in document.text:
                self.f_zs.setdefault(f, []).append(kChoosed)

            self.K = max(kChoosed, self.K)
            self.K_current = max(kChoosed, self.K_current)
            return kChoosed

        prob = [float(0.0)] * (self.K + 1)

        targetClusters = findTargetClusters(document, self.f_zs)

        # print('self.K', self.K, '#targetClusters', len(targetClusters))
        # for cluster in range(self.K):
        for cluster in targetClusters:

            if cluster not in self.m_z or self.m_z[cluster] == 0:
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster]  # / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]

                # for j in range(wordFre):
                for j in range(1):

                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha  # / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(1):
                # for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2

        allProb = 0  # record the amount of all probabilities
        prob_normalized = []  # record normalized probabilities
        for k in range(self.K + 1):
            allProb += prob[k]
        for k in range(self.K + 1):
            prob_normalized.append(prob[k] / allProb)

        kChoosed = prob_normalized.index(max(prob_normalized))
        if kChoosed == self.K:
            self.K += 1
            self.K_current += 1

        for f in document.text:
            self.f_zs.setdefault(f, []).append(kChoosed)

        return kChoosed

    def sampleCluster(self, d, document, documentID, isLast):
        prob = [float(0.0)] * (self.K + 1)

        targetClusters = findTargetClusters(document, self.f_zs)

        # print('self.K', self.K, '#targetClusters', len(targetClusters))
        # for cluster in range(self.K):
        for cluster in targetClusters:

            if cluster not in self.m_z or self.m_z[cluster] == 0:
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster]  # / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]

                # for j in range(wordFre):
                for j in range(1):

                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha  # / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(1):
                # for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2

        allProb = 0  # record the amount of all probabilities
        prob_normalized = []  # record normalized probabilities
        for k in range(self.K + 1):
            allProb += prob[k]
        for k in range(self.K + 1):
            prob_normalized.append(prob[k] / allProb)

        kChoosed = 0
        if isLast == 0:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        else:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1

        for f in document.text:
            self.f_zs.setdefault(f, []).append(kChoosed)

        return kChoosed

    # Clear the useless cluster
    def checkEmpty(self, cluster):
        if cluster in self.n_z and self.m_z[cluster] == 0:
            self.K_current -= 1
            self.m_z.pop(cluster)
            if cluster in self.n_z:
                self.n_z.pop(cluster)
                self.n_zv.pop(cluster)

    def output(self, documentSet, outputPath, wordList, batchNum):
        outputDir = outputPath + self.dataset + self.ParametersStr + "Batch" + str(batchNum) + "/"
        # try:
        #    isExists = os.path.exists(outputDir)
        #    if not isExists:
        #        os.mkdir(outputDir)
        #        print("\tCreate directory:", outputDir)
        # except:
        #    print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        # try:
        #    self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        # except:
        #    print("\tOutput Phi Words Wrong!")
        # self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):
        self.phi_zv = {}
        for cluster in self.n_zv:
            n_z_sum = 0
            if self.m_z[cluster] != 0:
                if cluster not in self.phi_zv:
                    self.phi_zv[cluster] = {}
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        n_z_sum += self.n_zv[cluster][v]
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(n_z_sum + self.beta0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in array:
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if k not in self.phi_zv:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if cluster in self.m_z and self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key=lambda tc: tc[1], reverse=True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        # outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        # writer = open(outputPath, 'w')
        for d in range(self.startDoc, self.currentDoc):
            document = documentSet.documents[d]
            documentID = document.documentID
            trueLabel = document.trueLabel
            text = document.text

            cluster = self.z[documentID]
            # writer.write(str(documentID) + " " + str(cluster) + "\n")
            clusterWriter.write(
                str(cluster) + "\t" + str(trueLabel) + "\t" + ' '.join(text) + "\n")

        # writer.close()

    def deleteClusters(self):

        c_sizes = self.m_z.values()
        print(c_sizes)
        mean_size = 0
        std_size = 0
        if len(c_sizes) > 2:
            mean_size = statistics.mean(c_sizes)
            std_size = statistics.stdev(c_sizes)

        print('mean_size', mean_size, 'std_size', std_size, 'len(self.m_z.items)', len(self.m_z))
        del_clusters = []
        for clusterId, docCount in self.m_z.items():
            if docCount > mean_size + std_size or docCount < abs(mean_size - std_size) or docCount <= 1:
                del_clusters.append(clusterId)
                print('Todelete:clusterId', clusterId, 'docCount', docCount)

        for clusterId in del_clusters:
            self.m_z[clusterId] = 0
            self.checkEmpty(clusterId)

        print('after:len(self.m_z.items)', len(self.m_z))
