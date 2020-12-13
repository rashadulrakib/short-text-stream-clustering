import os
from datetime import datetime

from general_util import readlistWholeJsonDataSet
from general_util import readStackOverflowDataSet
from evaluation import Evaluate
from read_pred_true_text import ReadPredTrueTextPostid
from clustering_term_online_stack_test_data import trainLoad_cluster_biterm
from clustering_term_online_stack_test_data import test_cluster_biterm
from clustering_term_online_stack_test_data import test_cluster_bitermMapping
from word_vec_extractor import extractAllWordVecsPartialStemming

ignoreMinusOne=False
isSemantic=False


fileDir = os.path.dirname(os.path.abspath(__file__))
#print(fileDir)
parentDir = os.path.dirname(fileDir)
#print(parentDir)
parentDir = os.path.dirname(parentDir)
#print(parentDir)


outputPath = "result/"

trainingFile=outputPath+'train_biterm_r.txt'

trainList_pred_true_text_postid=ReadPredTrueTextPostid(trainingFile, ignoreMinusOne)

print('result for', trainingFile)
Evaluate(trainList_pred_true_text_postid)  

all_words=[]
for item in trainList_pred_true_text_postid:
  all_words.extend(item[2].split(' '))
all_words=list(set(all_words))

gloveFile = "glove.6B.50d.txt"
embedDim=50
wordVectorsDic={}
if isSemantic==True:
  wordVectorsDic=extractAllWordVecsPartialStemming(gloveFile, embedDim, all_words)

c_bitermsFreqs={} 
c_totalBiterms={}
c_wordsFreqs={}
c_totalWords={}
c_txtIds={}
c_clusterVecs={}
txtId_txt={} #txtId_txt: postId->item, item[0], item[1],...

dic_clus__id={} #no impact

dic_biterm__clusterIds={}
dic_word__clusterIds={}

dicTrain_pred__trues={}


t11=datetime.now()

c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, dic_clus__id, dic_biterm__clusterIds, dic_word__clusterIds, dicTrain_pred__trues=trainLoad_cluster_biterm(trainList_pred_true_text_postid, c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, wordVectorsDic, dic_clus__id, dic_biterm__clusterIds, dic_word__clusterIds)


t12=datetime.now()	  
t_diff = t12-t11
#print("total time diff secs=",t_diff.seconds)  


##############end loading train clusters###########







testfile = parentDir+'/PyMigrationRecommendation/src/notebooks/test_stackoverflow_r_true_id_title_tags'
testList_pred_true_words_index_postid=readStackOverflowDataSet(testfile)

last_txtId=0 #no impact
max_c_id=0 #no impact

#test_cluster_biterm(testList_pred_true_words_index_postid, c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, max_c_id, wordVectorsDic, dic_clus__id,  dic_biterm__clusterIds, dicTrain_pred__trues)

test_cluster_bitermMapping(testList_pred_true_words_index_postid, c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, max_c_id, wordVectorsDic, dic_clus__id,  dic_biterm__clusterIds, dic_word__clusterIds, dicTrain_pred__trues)







