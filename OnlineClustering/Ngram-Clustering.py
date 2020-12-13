import os
from datetime import datetime
import math
from collections import Counter

from general_util import readStackOverflowDataSet
from txt_process_util import generateGrams
from txt_process_util import generateGramsConsucetive

min_gram=1
max_gram=2

dic_ngram__txtIds={}
dic_txtId__text={}

def buildNGramIndex(list_pred_true_words_index_postid):
  for item in list_pred_true_words_index_postid:
    words=item[2]	 
    txtId=item[3]  
	
    dic_txtId__text[txtId]=item     
	
    #grams=generateGrams(words,min_gram,len(words))	
    grams=generateGramsConsucetive(words,min_gram,max_gram)	
    
    for gram in grams:
      dic_ngram__txtIds.setdefault(gram, []).append(txtId)	

  






absFilePath = os.path.abspath(__file__)
#print(absFilePath)
fileDir = os.path.dirname(os.path.abspath(__file__))
#print(fileDir)
parentDir = os.path.dirname(fileDir)
#print(parentDir)
parentDir = os.path.dirname(parentDir)
#print(parentDir)




outputPath = "result/"

inputfile = parentDir+'/PyMigrationRecommendation/src/notebooks/train_stackoverflow_r_true_id_title_tags'

list_pred_true_words_index_postid=readStackOverflowDataSet(inputfile, False)

all_words=[]
for item in list_pred_true_words_index_postid:
  all_words.extend(item[2])
all_words=list(set(all_words))





buildNGramIndex(list_pred_true_words_index_postid)



def computeTextSimCommonWord_WordDic(words_i, words_j, txt_i_len, txt_j_len):  
  text_sim=0
  commonCount=0
  
  len_i=len(words_i)
  len_j=len(words_j)

  if len_i>len_j:
    temp=words_i
    words_i=words_j
    words_j=temp    
 
  for word_i, i_count in words_i.items():
    if word_i in words_j.keys():
      commonCount=commonCount+i_count+words_j[word_i]
      #commonCount=commonCount+ min(i_count,words_j[word_i])	  
  
  if txt_i_len>0 and txt_j_len>0:
    text_sim=commonCount/(txt_i_len+txt_j_len)
    
  return [text_sim, commonCount]




def aggregateTextIds(sortedGrams, dic_ngram__txtIds):
  txtIds=[]
  for sortGram in sortedGrams:
    if sortGram not in dic_ngram__txtIds: 
      continue
    txtIds.extend(dic_ngram__txtIds[sortGram])
	
  txtIds=set(txtIds)

  return txtIds  
    	  


#testPostId	trainPostId	TitleSim	BodySim	TagSim	LuceneHitRank	ProposedHitRank	lucene_hit_duration	Proposed_hit_duration_micro  LuceneTestTrueLabel

print("testpostId"+"\t"+"trainPostId"+"\tTitleSim\tBodySim\tTagSim\tLuceneHitRank\t"+"ProposedHitRank"+"\tlucene_hit_duration\t"+"Proposed_hit_duration_micro"+"\t"+"Proposed_TestTrueLabel"+"\t"+"testText"+"\t"+"trainText")

testfile = parentDir+'/PyMigrationRecommendation/src/notebooks/test_stackoverflow_r_true_id_title_tags'
testList_pred_true_words_index_postid=readStackOverflowDataSet(testfile, False)

for item in testList_pred_true_words_index_postid:
  testTruelabel= item[1] 
  words=item[2]	
  testpostId=item[4]  
  
  t11=datetime.now()
  
  grams=generateGrams(words,min_gram,len(words))
  sortedGrams = list(sorted(grams, key = len, reverse=True))

  flag=False  
  largestGram='' 
  ProposedHitRank=0  
  
  txtIds=aggregateTextIds(sortedGrams, dic_ngram__txtIds)
  
  for txtId in txtIds:
    ProposedHitRank+=1	
    train_item=dic_txtId__text[txtId] 
	  
    trainTruelabel=train_item[1]
    train_words=train_item[2]
    trainPostId=train_item[4]	
	  
    if str(trainTruelabel)==str(testTruelabel):
      #print('found found test words=', words, 'largestGram=', largestGram, 'sortedGrams=',sortedGrams, 'testTruelabel=',testTruelabel, '#txtIds=', len(txtIds), 'testpostId', testpostId) 
		
      #testPostId	trainPostId	TitleSim	BodySim	TagSim	LuceneHitRank	ProposedHitRank	lucene_hit_duration	Proposed_hit_duration  LuceneTestTrueLabel
      text_sim, commonCount = computeTextSimCommonWord_WordDic(Counter(words), Counter(train_words), len(words), len(train_words) )	  
      ProposedHitRank_val=max(1,math.floor(ProposedHitRank/len(sortedGrams)))	  
      t12=datetime.now()	  
      t_diff = t12-t11 		
      print(str(testpostId)+"\t"+str(trainPostId)+"\t0\t0\t"+str(text_sim)+"\t0\t"+str(ProposedHitRank_val)+"\t0\t"+str(t_diff.microseconds)+"\t"+str(testTruelabel)+"\t"+' '.join(words)+"\t"+' '.join(train_words)) 		
      flag=True		
      break  
		
  if flag==False:
    #print('not found test words=', words, 'largestGram=', largestGram, 'sortedGrams=',sortedGrams, 'testTruelabel=',testTruelabel, 'testpostId', testpostId)
    t12=datetime.now()	  
    t_diff = t12-t11 		
    print(str(testpostId)+"\t"+"-100"+"\t0\t0\t0\t0\t"+str(-100)+"\t0\t"+str(t_diff.microseconds)+"\t"+str(testTruelabel)+"\t"+' '.join(words)+"\t"+"")	
	
    	
  
  
