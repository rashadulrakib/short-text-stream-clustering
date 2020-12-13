import os
from datetime import datetime
import math
from collections import Counter
from operator import add
from scipy import spatial
import statistics 
from random import sample

from general_util import readStackOverflowDataSet
from txt_process_util import generateGrams
from txt_process_util import generateGramsConsucetive
from word_vec_extractor import extractAllWordVecsPartialStemming
from sent_vecgenerator import generate_sent_vecs_toktextdata

min_gram=1
max_gram=5

isSemantic=True

max_hitindex=10000
textType='title' #tag, title, body

simtype='TagSimilarity'
hitranktype='TagHitRank'

embeddingfile='/users/grad/rakib/w2vecs/glove/glove.6B.50d.txt'
embedDim=50


dic_ngram__txtIds={}
dic_txtId__text={}
dic_ngram__center={}
dic_txtId__vec={}
wordVectorsDic={}

def buildNGramIndex(list_pred_true_words_index_postid_createtime):
  for item in list_pred_true_words_index_postid_createtime:
    words=item[2]	 
    txtId=item[3] 
    #print('process index for', item)	
	
    text_Vec=None
	
    if isSemantic==True:	
      if txtId in dic_txtId__vec:
        text_Vec=	dic_txtId__vec[txtId]
      else:
        X=generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
        text_Vec=X[0] 

      dic_txtId__vec[txtId]=text_Vec	  
	
    	
	
    dic_txtId__text[txtId]=item     
	
    grams=generateGramsConsucetive(words,min_gram,max_gram) #len(words))	
    
    for gram in grams:
      dic_ngram__txtIds.setdefault(gram, []).append(txtId)

      
      if isSemantic==True:	  
        if gram in dic_ngram__center:
          dic_ngram__center[gram]=list( map(add, dic_ngram__center[gram], text_Vec) )
        else:
          dic_ngram__center[gram]=text_Vec
 
      #print(gram, dic_ngram__center[gram]) 

  






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




#wordVectorsDic=extractAllWordVecsPartialStemming(embeddingfile, embedDim, all_words)

buildNGramIndex(list_pred_true_words_index_postid_createtime)

if isSemantic==True:
  gram_keys=list(dic_ngram__center.keys())
  randkeys = sample(gram_keys, 100)
  delKeys=set(gram_keys)-set(randkeys)
  for key in delKeys:
    if key not in dic_ngram__center:
      continue
    del dic_ngram__center[key]
  
  print(dic_ngram__center.keys(), len(dic_ngram__center))  


###########cluster staticics

c_valCounts=[len(dic_ngram__txtIds[x]) for x in dic_ngram__txtIds if isinstance(dic_ngram__txtIds[x], list) and len(dic_ngram__txtIds[x])>1]
minValCount=min(c_valCounts)
maxValCount=max(c_valCounts)
avgValCount=statistics.mean(c_valCounts)
stdValCount=statistics.stdev(c_valCounts)
medianCount=statistics.median(c_valCounts)
print('total cluster#', len(c_valCounts), 'minValCount', minValCount, 'maxValCount', maxValCount, 'avgValCount', avgValCount, 'stdValCount', stdValCount, 'medianCount', medianCount)
#print("clusterId"+"\t"+"#textCount")
#for gram, txtIds in dic_ngram__txtIds.items():
#  print(gram+"\t"+str(len(set(txtIds))))
##end cluster staticics














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
  
  
def findDuplicateBySemantic(test_item):

  t11=datetime.now()

  testTruelabel= test_item[1] 
  test_words=test_item[2]	
  testpostId=test_item[4]  
  testCreateTime=test_item[5]
  testDateTime= datetime.strptime(test_item[5].split("T")[0] ,"%Y-%m-%d")

  test_X=generate_sent_vecs_toktextdata([test_words], wordVectorsDic, embedDim)
  test_text_Vec=test_X[0]	  
  
  dic_gram__sim={}
  
  for gram, center_Vec in dic_ngram__center.items():
    sim = 1-spatial.distance.cosine(center_Vec, test_text_Vec)
    dic_gram__sim[gram]=sim	
  
  list_sim=list(dic_gram__sim.values())
  sim_stdev=statistics.stdev(list_sim)
  sim_mean=statistics.mean(list_sim)
  
  all_textIds=[]  
  for gram, center_Vec in dic_ngram__center.items():
    gram_sim=dic_gram__sim[gram]
    if gram_sim>=sim_mean+sim_stdev:
      txtIds=dic_ngram__txtIds[gram]
      all_textIds.extend(txtIds)

  all_textIds=set(all_textIds) 
  ProposedHitRank=0
  print('sem-all_textIds', len(all_textIds), 'test_words', test_words)  
  for txtId in all_textIds:
    ProposedHitRank+=1	
    if ProposedHitRank > max_hitindex:
      break
	  
    train_item=dic_txtId__text[txtId] 
	  
    trainTruelabel=train_item[1]
    train_words=train_item[2]
    trainPostId=train_item[4]	
    trainCreateTime = train_item[5]	
	  
    if str(trainTruelabel)==str(testTruelabel):
    
      t12=datetime.now()	  
      t_diff = t12-t11 	
	  
      text_sim, commonCount = computeTextSimCommonWord_WordDic(Counter(test_words), Counter(train_words), len(test_words), len(train_words) )	  
      ProposedHitRank_val=int(max(1,math.floor(ProposedHitRank/len(sortedGrams))))	  
      	
      trainDateTime= datetime.strptime(train_item[5].split("T")[0] ,"%Y-%m-%d")
      date_diff=trainDateTime-testDateTime
      date_diff=date_diff.days      	  
	  
      print(str(testpostId)+"\t"+str(trainPostId)+"\t"+str(text_sim)+"\t"+str(ProposedHitRank_val)+"\t"+str(t_diff.microseconds)+"\t"+str(testTruelabel)+"\t"+' '.join(test_words)+"\t"+' '.join(train_words)+"\t"+testCreateTime+"\t"+trainCreateTime+"\t"+str(date_diff))
	  
      return True	  

    
  
  
  return False  
  
    	  


#testPostId	trainPostId	TitleSim	BodySim	TagSim	LuceneHitRank	ProposedHitRank	lucene_hit_duration	Proposed_hit_duration_micro  LuceneTestTrueLabel


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

    #flag=findDuplicateBySemantic(item) 
    

    if flag==False:
      t12=datetime.now()	  
      t_diff = t12-t11 		
      print(str(testpostId)+"\t"+"-100"+"\t0\t"+str(-100)+"\t"+str(t_diff.microseconds)+"\t"+str(testTruelabel)+"\t"+' '.join(words)+"\t"+""+"\t"+""+"\t"+""+"\t"+"")	
	
    	
  
  
