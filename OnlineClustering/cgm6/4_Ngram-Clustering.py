import os
from datetime import datetime
import math
from collections import Counter

from general_util import readStackOverflowDataSet
from txt_process_util import generateGrams
from txt_process_util import generateGramsConsucetive

min_gram=1
max_gram=5

max_hitindex=10000
textType='tag' #tag, title, body

dic_ngram__txtIds={}
dic_txtId__text={}

def buildNGramIndex(list_pred_true_words_index_postid_createtime):
  for item in list_pred_true_words_index_postid_createtime:
    words=item[2]	 
    txtId=item[3]  
	
    dic_txtId__text[txtId]=item     
	
    grams=generateGramsConsucetive(words,min_gram,max_gram) #len(words))	
    
    for gram in grams:
      dic_ngram__txtIds.setdefault(gram, []).append(txtId)	

  






#absFilePath = os.path.abspath(__file__)
#print(absFilePath)
#fileDir = os.path.dirname(os.path.abspath(__file__))
#print(fileDir)
#parentDir = os.path.dirname(fileDir)
#print(parentDir)
#parentDir = os.path.dirname(parentDir)
#print(parentDir)




outputPath = "result/"

inputfile = 'train_stackoverflow_r_true_id_title_tags_body_createtime'


list_pred_true_words_index_postid_createtime=readStackOverflowDataSet(inputfile, False, 6, textType)

all_words=[]
for item in list_pred_true_words_index_postid_createtime:
  all_words.extend(item[2])
all_words=list(set(all_words))





buildNGramIndex(list_pred_true_words_index_postid_createtime)






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
    text_sim=float(commonCount)/(txt_i_len+txt_j_len)
	
  #print(text_sim, commonCount, words_i, words_j, txt_i_len, txt_j_len)	
    
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

simtype='TagSimilarity'
hitranktype='TagHitRank'
print("testpostId"+"\t"+"trainPostId"+"\t"+simtype+"\t"+hitranktype+"\t"+"Proposed_hit_duration_micro"+"\t"+"Proposed_TestTrueLabel"+"\t"+"testText"+"\t"+"trainText"+"\t"+"testCreateTime"+"\t"+"TrainCreateTime"+"\t"+"DaysDiff")

testfile = 'test_stackoverflow_r_true_id_title_tags_body_createtime'
testList_pred_true_words_index_postid_createtime=readStackOverflowDataSet(testfile, False, 6, textType) #tag, title, body


for item in testList_pred_true_words_index_postid_createtime:
  testTruelabel= item[1] 
  words=item[2]	
  testpostId=item[4]  
  testCreateTime=item[5]
  
  testDateTime= datetime.strptime(item[5].split("T")[0] ,"%Y-%m-%d")
  
  t11=datetime.now()
  
  grams=generateGramsConsucetive(words,min_gram,max_gram)#len(words))
  sortedGrams = list(sorted(grams, key = len, reverse=True))

  flag=False  
  largestGram='' 
  ProposedHitRank=0  
  
  txtIds=aggregateTextIds(sortedGrams, dic_ngram__txtIds)
  
  for txtId in txtIds:
    ProposedHitRank+=1	
    if ProposedHitRank > max_hitindex:
      break
	  
    train_item=dic_txtId__text[txtId] 
	  
    trainTruelabel=train_item[1]
    train_words=train_item[2]
    trainPostId=train_item[4]	
    trainCreateTime = train_item[5]	
	  
    if str(trainTruelabel)==str(testTruelabel):
      #print('found found test words=', words, 'largestGram=', largestGram, 'sortedGrams=',sortedGrams, 'testTruelabel=',testTruelabel, '#txtIds=', len(txtIds), 'testpostId', testpostId) 
		
      #testPostId	trainPostId	TitleSim	BodySim	TagSim	LuceneHitRank	ProposedHitRank	lucene_hit_duration	Proposed_hit_duration  LuceneTestTrueLabel
      t12=datetime.now()	  
      t_diff = t12-t11 	
	  
      text_sim, commonCount = computeTextSimCommonWord_WordDic(Counter(words), Counter(train_words), len(words), len(train_words) )	  
      ProposedHitRank_val=int(max(1,math.floor(ProposedHitRank/len(sortedGrams))))	  
      	
      trainDateTime= datetime.strptime(train_item[5].split("T")[0] ,"%Y-%m-%d")
      date_diff=trainDateTime-testDateTime
      date_diff=date_diff.days      	  
	  
      print(str(testpostId)+"\t"+str(trainPostId)+"\t"+str(text_sim)+"\t"+str(ProposedHitRank_val)+"\t"+str(t_diff.microseconds)+"\t"+str(testTruelabel)+"\t"+' '.join(words)+"\t"+' '.join(train_words)+"\t"+testCreateTime+"\t"+trainCreateTime+"\t"+str(date_diff)) 		
      flag=True		
      break  
		
  if flag==False:
    #print('not found test words=', words, 'largestGram=', largestGram, 'sortedGrams=',sortedGrams, 'testTruelabel=',testTruelabel, 'testpostId', testpostId)
    t12=datetime.now()	  
    t_diff = t12-t11 		
    print(str(testpostId)+"\t"+"-100"+"\t0\t"+str(-100)+"\t"+str(t_diff.microseconds)+"\t"+str(testTruelabel)+"\t"+' '.join(words)+"\t"+""+"\t"+""+"\t"+""+"\t"+"")	
	
    	
  
  
