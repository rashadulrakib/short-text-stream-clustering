import os
from datetime import datetime

from general_util import readStackOverflowDataSetBody
from evaluation import Evaluate
from read_pred_true_text import ReadPredTrueTextPostid
from clustering_term_online_stack import cluster_biterm
from word_vec_extractor import extractAllWordVecsPartialStemming

from clustering_term_online_stack_test_data import test_cluster_bitermMapping_buffer

ignoreMinusOne=False
isSemantic=False

absFilePath = os.path.abspath(__file__)
print(absFilePath)
fileDir = os.path.dirname(os.path.abspath(__file__))
print(fileDir)
parentDir = os.path.dirname(fileDir)
print(parentDir)
parentDir = os.path.dirname(parentDir)
print(parentDir)

dataDir = "data/"
outputPath = "result/"

min_gram=1
max_gram=2

lang='r'
tagIgnore=''
max_hitindex=5000
textType='tag' #tag, title, body
microDivide=1000000

simtype=textType+'Similarity'
hitranktype=textType+'HitRank'

dic_ngram__txtIds={}
dic_txtId__text={}


#inputfile = parentDir+'/PyMigrationRecommendation/src/notebooks/train_stackoverflow_r_true_id_title_tags'
inputfile = parentDir+'/PyMigrationRecommendation/src/notebooks/train_stackoverflow_'+lang+'_true_id_title_tags_body_createtime'

resultFile=outputPath+'train_biterm_'+lang+'_'+textType+'.txt'

list_pred_true_words_index_postid_createtime=readStackOverflowDataSetBody(inputfile, True, 6, textType, tagIgnore)

print(len(list_pred_true_words_index_postid_createtime))

all_words=[]
for item in list_pred_true_words_index_postid_createtime:
  all_words.extend(item[2])
all_words=list(set(all_words))

gloveFile = "glove.6B.50d.txt"
embedDim=50
wordVectorsDic={}
if isSemantic==True:
  wordVectorsDic=extractAllWordVecsPartialStemming(gloveFile, embedDim, all_words)

if os.path.exists(resultFile):
  os.remove(resultFile)
  
c_bitermsFreqs={} 
c_totalBiterms={}
c_wordsFreqs={}
c_totalWords={}
c_txtIds={}
c_clusterVecs={}
txtId_txt={}
last_txtId=0  
max_c_id=0
dic_clus__id={}

dic_biterm__clusterId_Freq={}
dic_biterm__allClusterFreq={}

dic_biterm__clusterIds={}
c_textItems={}
dic_ngram__textItems={}



f = open(resultFile, 'w')

t11=datetime.now()

c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, dic_clus__id, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq, dic_biterm__clusterIds, c_textItems, dic_ngram__textItems=cluster_biterm(f, list_pred_true_words_index_postid_createtime, c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, max_c_id, wordVectorsDic, dic_clus__id, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq, dic_biterm__clusterIds, c_textItems, dic_ngram__textItems, min_gram, max_gram)


t12=datetime.now()	  
t_diff = t12-t11
print("total time diff secs=",t_diff.seconds)  

f.close()
  
listtuple_pred_true_text=ReadPredTrueTextPostid(resultFile, ignoreMinusOne)

print('result for', inputfile)
Evaluate(listtuple_pred_true_text)  




####test

testfile = parentDir+'/PyMigrationRecommendation/src/notebooks/test_stackoverflow_'+lang+'_true_id_title_tags_body_createtime'
testList_pred_true_words_index_postid_createtime=readStackOverflowDataSetBody(testfile, True, 6, textType, tagIgnore)


last_txtId=0 #no impact
max_c_id=0 #no impact
#c_txtIds, txtId_txt,  no impact

test_cluster_bitermMapping_buffer(testList_pred_true_words_index_postid_createtime, c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, max_c_id, wordVectorsDic, dic_clus__id, dic_biterm__clusterIds, c_textItems, dic_ngram__textItems, min_gram, max_gram, max_hitindex)




