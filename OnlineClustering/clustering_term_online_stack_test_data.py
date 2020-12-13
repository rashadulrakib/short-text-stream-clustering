from evaluation import Evaluate
from collections import Counter
import statistics 
from random import randint
from datetime import datetime
from sent_vecgenerator import generate_sent_vecs_toktextdata
from txt_process_util import construct_biterms
from txt_process_util import generateGramsConsucetive
from txt_process_util import semanticSims
from operator import add
import math

#need to delete high entropy words in targetClusterIds when compute similarity between ti to the targetClusterIds
 
#minGSize=1
#maxGSize=1
microDivide=1000000



embedDim=50

ignoreMinusOne=False
isSemantic=False


def removeHighEntropyFtrs(c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq):

  ####c_bitermsFreqs[clusterId][biterm]=0

	
  dic_biterm__entropy={}
     
  for biterm, dic_clusterId__Freq in dic_biterm__clusterId_Freq.items():
    entropy =0
    totalFreq=dic_biterm__allClusterFreq[biterm] 
    dic_biterm__entropy[biterm]=entropy 	
    if totalFreq<=0:
      continue	
    for clusterId, Freq in dic_clusterId__Freq.items():
      entropy=entropy + -1*(Freq/totalFreq)*math.log(Freq/totalFreq)    	

    dic_biterm__entropy[biterm]=entropy  

  allentropies=list(dic_biterm__entropy.values())

  mean_entropy=0
  std_entropy=0

  if len(allentropies)>2:
    mean_entropy=statistics.mean(allentropies)
    std_entropy=statistics.stdev(allentropies) 


  listTargetClusters=[]
  
  for biterm, entropy in dic_biterm__entropy.items():
    if entropy>mean_entropy+std_entropy:
      del dic_biterm__clusterId_Freq[biterm] 
      del dic_biterm__allClusterFreq[biterm]
      #print(biterm, entropy)  
       
      for clusterId, dic_biterms__freq in c_bitermsFreqs.items():
        if biterm in dic_biterms__freq:
          clusterBitermFreq=c_bitermsFreqs[clusterId][biterm] 		
          del c_bitermsFreqs[clusterId][biterm]
          c_totalBiterms[clusterId]-=clusterBitermFreq		  
           	  
      	  
        
    
  listTargetClusters=list(c_bitermsFreqs.keys())
  for clusterId in listTargetClusters:
    if c_totalBiterms[clusterId]<=0:
      del c_totalBiterms[clusterId]
      del c_bitermsFreqs[clusterId]
      del c_txtIds[clusterId]	  
      
   

 

  return [c_bitermsFreqs, c_totalBiterms, c_txtIds, dic_biterm__clusterId_Freq, dic_biterm__allClusterFreq]


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
  
  if txt_i_len>0 and txt_j_len>0:
    text_sim=commonCount/(txt_i_len+txt_j_len)
    
  return [text_sim, commonCount]

def computeTextSimCommonWord_WordArr(txt_i_wordArr, txt_j_wordArr): #not used
  
  txt_i_len=len(txt_i_wordArr)
  txt_j_len=len(txt_j_wordArr)  
  
  words_i= collections.Counter(txt_i_wordArr)#assume words_i small
  words_j= collections.Counter(txt_j_wordArr) 
  
  text_sim, commonCount=computeTextSimCommonWord_WordDic(words_i, words_j, txt_i_len, txt_j_len)
 
  return [text_sim, commonCount]


def removeTargetMultiClusterBiTerms(c_bitermsFreqs, c_totalBiterms, c_txtIds, targetClusterIds, txtBitermsFreqs, dic_biterm__clusterIds):

  dic_biterm__TargetClustersTotalFreq={}
  dic_biterm__TargetClustersEntropy={}
  for clusterId in targetClusterIds:
    if clusterId not in c_bitermsFreqs: 
      continue     
    dic_bitermsFreqs=c_bitermsFreqs[clusterId]
	
    for biterm, freq in dic_bitermsFreqs.items():
      if biterm not in dic_biterm__TargetClustersTotalFreq:
        dic_biterm__TargetClustersTotalFreq[biterm]=0
      dic_biterm__TargetClustersTotalFreq[biterm]+=freq
	  
  for clusterId in targetClusterIds:
    if clusterId not in c_bitermsFreqs: 
      continue     
    dic_bitermsFreqs=c_bitermsFreqs[clusterId]
	
    for biterm, freq in dic_bitermsFreqs.items():	  
      if biterm not in dic_biterm__TargetClustersEntropy:
        dic_biterm__TargetClustersEntropy[biterm]=0
      dic_biterm__TargetClustersEntropy[biterm]-=freq/dic_biterm__TargetClustersTotalFreq[biterm]*math.log(freq/dic_biterm__TargetClustersTotalFreq[biterm])        		
  
  
  
  allentropies=list(dic_biterm__TargetClustersEntropy.values())
  #print(dic_biterm__TargetClustersEntropy)

  mean_entropy=0
  std_entropy=0

  if len(allentropies)>2:
    mean_entropy=statistics.mean(allentropies)
    std_entropy=statistics.stdev(allentropies) 
  
  for biterm, entropy in dic_biterm__TargetClustersEntropy.items():
    if entropy>mean_entropy+std_entropy:
      for clusterId in targetClusterIds: 
        if clusterId not in c_bitermsFreqs or biterm not in c_bitermsFreqs[clusterId]:
          continue		
        clusterBitermFreq=c_bitermsFreqs[clusterId][biterm] 		
        del c_bitermsFreqs[clusterId][biterm]
        c_totalBiterms[clusterId]-=clusterBitermFreq	
              
  for clusterId in targetClusterIds:
    if clusterId not in c_totalBiterms:
      continue	
    if c_totalBiterms[clusterId]<=0:
      del c_totalBiterms[clusterId]
      if clusterId in c_bitermsFreqs:	  
        del c_bitermsFreqs[clusterId]
      if clusterId in c_txtIds:		
        del c_txtIds[clusterId]	  	  
      	
        	

  return [c_bitermsFreqs, c_totalBiterms, c_txtIds, txtBitermsFreqs]

def findTextIds(targetClusterIds, c_txtIds):
  textIds=[]
  for cid in targetClusterIds:
    if cid not in c_txtIds:
      continue
    textIds.extend(c_txtIds[cid])	  

  textIds=set(textIds)
  return textIds
  
def findTextItems(targetClusterIds, c_textItems):  
  textItems=[]
  dic_dup={}
  
  for cid in targetClusterIds:
    if cid not in c_textItems:
      continue
	  
    #textItems.extend(c_textItems[cid])	

    for textItem in c_textItems[cid]:
      postId=textItem[4]
    
      if postId not in dic_dup: 	  
        textItems.append(textItem)
      dic_dup[postId]=postId	
  
  #print('len(textItems)', len(textItems))

  
  del dic_dup
  return textItems
  
def aggregateTextItems(sortedGrams, dic_ngram__textItems):
  txtItems=[]
  dic_dup={}
  for sortGram in sortedGrams:
    if sortGram not in dic_ngram__textItems: 
      continue
	
    #txtItems.extend(dic_ngram__textItems[sortGram]) 	
    for textItem in dic_ngram__textItems[sortGram]:
      postId=textItem[4]
	  
      if postId not in dic_dup: 	  
        txtItems.append(textItem)
      dic_dup[postId]=postId		
	

  del dic_dup
  return txtItems 

def findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds):
  targetClusterIds=[]
  
  for biterm, freq in txtBitermsFreqs.items():
    if biterm not in dic_biterm__clusterIds:
      continue
    targetClusterIds.extend(dic_biterm__clusterIds[biterm])     	  

  targetClusterIds=set(targetClusterIds)
  return targetClusterIds  

def findCloseClusterByTargetClusters(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec, dic_biterm__clusterIds, targetClusterIds):
  clusterId_lex=-1
  clusterId_sem=-1
  clusterId=-1
  max_sim=0
  max_sim_lex=0

  dic_lexicalSims={}
  
  for clusId in targetClusterIds:
    if clusId not in c_bitermsFreqs:
      continue	
    #print('####targetClusterIds', len(targetClusterIds))	  
    clusBitermsFreqs=c_bitermsFreqs[clusId]
    txt_j_len= c_totalBiterms[clusId]	

    text_sim, commonCount=computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len, txt_j_len)
    if text_sim> max_sim:
      max_sim=text_sim
      clusterId=clusId	  

    if text_sim> max_sim_lex:
      max_sim_lex=text_sim
      clusterId_lex=clusId
    dic_lexicalSims[clusId]=text_sim     

  '''for clusId, clusBitermsFreqs in c_bitermsFreqs.items():
    txt_j_len= c_totalBiterms[clusId]	

    text_sim, commonCount=computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len, txt_j_len)
    if text_sim> max_sim:
      max_sim=text_sim
      clusterId=clusId	  

    if text_sim> max_sim_lex:
      max_sim_lex=text_sim
      clusterId_lex=clusId
    dic_lexicalSims[clusId]=text_sim''' 	  

    

  lex_sim_values= list(dic_lexicalSims.values())
  

  mean_lex_sim=0
  std_lex_sim=0

  if len(lex_sim_values)>2:
    mean_lex_sim=statistics.mean(lex_sim_values)
    std_lex_sim=statistics.stdev(lex_sim_values)  

  



  if clusterId_lex==-1: #or clusterId_sem==-1:
    #clusterId=len(c_bitermsFreqs)+1    
    clusterId=max_c_id+1
    if isSemantic==True:	
      dic_semanticSims, clusterId_sem, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, c_clusterVecs, c_txtIds)
      sem_sim_values= list(dic_semanticSims.values())
      mean_sem_sim=0
      std_sem_sim=0
      if len(sem_sim_values)>2:	
        mean_sem_sim=statistics.mean(sem_sim_values)
        std_sem_sim=statistics.stdev(sem_sim_values) 
        if maxSim_Semantic>=mean_sem_sim+std_sem_sim: # and randint(0,1)==1: work
          clusterId=clusterId_sem 	  
  #elif clusterId_lex!=clusterId_sem:
  #  clusterId=max_c_id+1
  #elif clusterId_lex==clusterId_sem:
  #  clusterId=clusterId_lex
  #elif max_sim_lex>=mean_lex_sim+std_lex_sim: # and randint(0,1)==1: work
  #  clusterId=clusterId_lex    
  else:
    clusterId=clusterId_lex







  #print(text_Vec, clusterId_lex, clusterId_sem, clusterId)
  return clusterId
  #return clusterId_lex



def populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id, text_Vec, dic_biterm__clusterIds, dic_word__clusterIds):

  c_txtIds.setdefault(clusterId,[]).append(current_txt_id)	

  if clusterId not in c_bitermsFreqs:
    c_bitermsFreqs[clusterId]={}
  if clusterId not in c_totalBiterms:
    c_totalBiterms[clusterId]=0
  c_totalBiterms[clusterId]+=bi_terms_len    
	
  if clusterId not in c_wordsFreqs:
    c_wordsFreqs[clusterId]={}
  if clusterId not in c_totalWords:
    c_totalWords[clusterId]=0
  c_totalWords[clusterId]+=words_len

  
    
  	
  for biterm, bitermFreq in txtBitermsFreqs.items():
    if biterm not in c_bitermsFreqs[clusterId]:
      c_bitermsFreqs[clusterId][biterm]=0
    c_bitermsFreqs[clusterId][biterm]+=bitermFreq	
	
    dic_biterm__clusterIds.setdefault(biterm,[]).append(clusterId)		

  for word, wordFreq in txtWordsFreqs.items():
    if word not in c_wordsFreqs[clusterId]:
      c_wordsFreqs[clusterId][word]=0
    c_wordsFreqs[clusterId][word]+=wordFreq
	
    dic_word__clusterIds.setdefault(word,[]).append(clusterId)	
	
  
  
  if isSemantic==True:
    if clusterId not in c_clusterVecs:
      c_clusterVecs[clusterId]=text_Vec
    else:
      c_clusterVecs[clusterId] = list(map(add, text_Vec, c_clusterVecs[clusterId]))
    
    

  return [c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs ,dic_biterm__clusterIds, dic_word__clusterIds]    
  	
	  

def trainLoad_cluster_biterm(trainList_pred_true_text_postid, c_bitermsFreqs={}, c_totalBiterms={}, c_wordsFreqs={}, c_totalWords={}, c_txtIds={}, c_clusterVecs={}, txtId_txt={}, wordVectorsDic={}, dic_clus__id={},  dic_biterm__clusterIds={}, dic_word__clusterIds={}):
  print("train cluster_bigram")

   
  dicTrain_pred__trues={}
  
  eval_pred_true_txt=[]
  
  line_count=0
  
  t11=datetime.now()
     
  for item in trainList_pred_true_text_postid:
    pred=item[0]  #pred clusId  
    true=item[1]   		
    words=item[2].split(' ')
    postId=item[3]	
    bi_terms=construct_biterms(words)
    #bi_terms=generateGramsConsucetive(words, minGSize, maxGSize)	
    #print(words, bi_terms)	
    
	
    line_count+=1	
   
      	
    txtBitermsFreqs=Counter(bi_terms)
    bi_terms_len= len(bi_terms)	
	
    txtWordsFreqs=Counter(words)
    words_len= len(words) 	
	
    text_Vec=[0]*embedDim 	
    if isSemantic==True:	
      X=generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
      text_Vec=X[0]	
    
    clusterId=int(pred)
    #dicTrain_pred__trues[clusterId]=int(true) 
    dicTrain_pred__trues.setdefault(clusterId, []).append(int(true))   	
    
    dic_clus__id[clusterId]=clusterId 
    current_txt_id=int(postId)	

    txtId_txt[current_txt_id]=item	

    c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, dic_biterm__clusterIds, dic_word__clusterIds=populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id, text_Vec, dic_biterm__clusterIds, dic_word__clusterIds)
	 
    eval_pred_true_txt.append([clusterId, item[1], item[2]])

    #if clusterId>0:    
    #  print(item, bi_terms)
      #print(dic_biterm__clusterIds.keys())	  
    	
    if line_count%1000==0:  
      print('#######-personal-eval_pred_true_txt', len(eval_pred_true_txt))	 	
      Evaluate(eval_pred_true_txt, ignoreMinusOne)

      t12=datetime.now()	  
      t_diff = t12-t11
      print("total time diff secs=",t_diff.seconds)    

  
  return [c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt,  dic_clus__id, dic_biterm__clusterIds, dic_word__clusterIds, dicTrain_pred__trues] 

def test_cluster_bitermMapping(testList_pred_true_words_index_postid, c_bitermsFreqs={}, c_totalBiterms={}, c_wordsFreqs={}, c_totalWords={}, c_txtIds={}, c_clusterVecs={}, txtId_txt={}, last_txtId=0, max_c_id=0, wordVectorsDic={}, dic_clus__id={},  dic_biterm__clusterIds={}, dic_word__clusterIds={}, dicTrain_pred__trues={}):
  #print("test_cluster_bitermMapping")
  
  eval_pred_true_txt=[]
  
  line_count=0

  print("testpostId"+"\t"+"trainPostId"+"\tTitleSim\tBodySim\tTagSim\tLuceneHitRank\t"+"ProposedHitRank"+"\tlucene_hit_duration\t"+"Proposed_hit_duration_micro"+"\t"+"Proposed_TestTrueLabel")
     
  for item in testList_pred_true_words_index_postid:
    t11=datetime.now()  
    pred=item[0]  
    testTrue=int(item[1])
    words=item[2]
    postId=item[4]	
    bi_terms=construct_biterms(words)
    #bi_terms=generateGramsConsucetive(words, minGSize, maxGSize)	
    print(words, bi_terms, pred)	
    
    current_txt_id=int(postId)

    line_count+=1	
      	
    txtBitermsFreqs=Counter(bi_terms)
    bi_terms_len= len(bi_terms)	
	
    txtWordsFreqs=Counter(words)
    words_len= len(words) 	
	
    text_Vec=[0]*embedDim 	
    if isSemantic==True:	
      X=generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
      text_Vec=X[0]
        	
			
    #text->biterms
    #biterms->targetClusterIds
    #targetClusterIds->txtIds  by c_txtIds
    #txtIds->textItems	by txtId_txt

    targetClusterIds=findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds)
	
    print('len(targetClusterIds)', len(targetClusterIds))
    textIds=findTextIds(targetClusterIds, c_txtIds)	
    print('len(textIds)',len(textIds))
    pathCount=0	
    flag=False	
    for textId in textIds:
      trainItem = txtId_txt[textId] 
      trainTrue=int(trainItem[1])	
      trainPostId=trainItem[3]	  
      pathCount+=1
      	  
      if str(testTrue) == str(trainTrue):      	  
        #print('found found', 'testTrue', testTrue, 'testwords', words,'postId', postId, 'pathCount', pathCount, 'len(targetClusterIds)', len(targetClusterIds))
        t12=datetime.now()	  
        t_diff = t12-t11		
        print(str(postId)+"\t"+str(trainPostId)+"\t0\t0\t0\t0\t"+str(len(targetClusterIds))+"\t0\t"+str(t_diff.microseconds)+"\t"+str(testTrue))		
        flag=True
        break
		
      if pathCount>max_hitindex:
        break	
    
    if flag==False:
	     	
      '''targetClusterIds=findTargetClusters(txtWordsFreqs, dic_word__clusterIds)
      textIds=findTextIds(targetClusterIds, c_txtIds)
      pathCount=0	
      flag=False	
      for textId in textIds:
        trainItem = txtId_txt[textId] 
        trainTrue=int(trainItem[1])	
        trainPostId=trainItem[3]			
        pathCount+=1
        if str(testTrue) == str(trainTrue):      	  
          #print('found found', 'testTrue', testTrue, 'testwords', words,'postId', postId, 'pathCount', pathCount, 'len(targetClusterIds)', len(targetClusterIds))	
          t12=datetime.now()	  
          t_diff = t12-t11		
          print(str(postId)+"\t"+str(trainPostId)+"\t0\t0\t0\t0\t"+str(len(targetClusterIds))+"\t0\t"+str(t_diff.microseconds)+"\t"+str(testTrue)) 		  
          flag=True
          break	'''
      
      if flag==False:
        #print('not found', 'testTrue', testTrue, 'testwords', words,'postId', postId, 'pathCount', pathCount, 'len(targetClusterIds)', len(targetClusterIds)) 
        t12=datetime.now()	  
        t_diff = t12-t11			
        print(str(postId)+"\t"+"-100"+"\t0\t0\t0\t0\t-100"+"\t0\t"+str(t_diff.microseconds)+"\t"+str(testTrue))		
		
 
 
 

def test_cluster_bitermMapping_buffer(testList_pred_true_words_index_postid_createtime, c_bitermsFreqs={}, c_totalBiterms={}, c_wordsFreqs={}, c_totalWords={}, c_txtIds={}, c_clusterVecs={}, txtId_txt={}, last_txtId=0, max_c_id=0, wordVectorsDic={}, dic_clus__id={},  dic_biterm__clusterIds={}, c_textItems={}, dic_ngram__textItems={}, min_gram=1, max_gram=2, max_hitindex=10000):
   
  eval_pred_true_txt=[]
  
  line_count=0

  print("testpostId"+"\t"+"trainPostId"+"\t"+"simtype"+"\t"+"hitranktype"+"\t"+"Proposed_hit_duration_micro"+"\t"+"Proposed_TestTrueLabel"+"\t"+"testText"+"\t"+"trainText"+"\t"+"testCreateTime"+"\t"+"TrainCreateTime"+"\t"+"DaysDiff")
     
  for item in testList_pred_true_words_index_postid_createtime:
    t11=datetime.now()  
    pred=item[0]  
    testTrue=int(item[1])
    words=item[2]
    testpostId=item[4]	
    testDateTime= datetime.strptime(item[5].split("t")[0] ,"%Y-%m-%d") #datetime.now() # item[5]	
    #print('testDateTime', item[5])	
    bi_terms=construct_biterms(words)
	
    	
    #print(words, bi_terms, pred)	
    
    #current_txt_id=int(testpostId)

    line_count+=1	
      	
    txtBitermsFreqs=Counter(bi_terms)
    bi_terms_len= len(bi_terms)	
	
    txtWordsFreqs=Counter(words)
    words_len= len(words) 	
	
    text_Vec=[0]*embedDim 	
    if isSemantic==True:	
      X=generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
      text_Vec=X[0]
        	
			
    #text->biterms
    #biterms->targetClusterIds
    #targetClusterIds->txtIds  by c_txtIds
    #txtIds->textItems	by txtId_txt

    targetClusterIds=findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds)
    trainItems=findTextItems(targetClusterIds, c_textItems)	
	
    grams=generateGramsConsucetive(words, min_gram, max_gram)		 
    sortedGrams = list(sorted(grams, key = len, reverse=True))	
    train_Items=aggregateTextItems(sortedGrams, dic_ngram__textItems)
	
    trainItems.extend(train_Items)	
	
    #print('len(targetClusterIds)', len(targetClusterIds), 'len(trainItems)',len(trainItems), words)
    pathCount=0	
    flag=False	
    for trainItem in trainItems: 
      #list_pred_true_words_index_postid in clustring_term_online_stack=	trainItem
      trainTrue=int(trainItem[1])
      train_words=trainItem[2]		  
      trainPostId=trainItem[4]	
        
      pathCount+=1
      	  
      if str(testTrue) == str(trainTrue): 
        #grams=generateGramsConsucetive(words, min_gram, max_gram)		 
        #sortedGrams = list(sorted(grams, key = len, reverse=True))	  
        ProposedHitRank_val=int(max(1,math.floor(pathCount/len(sortedGrams))))
		
        t12=datetime.now()	  
        t_diff = t12-t11		
        #print(str(testpostId)+"\t"+str(trainPostId)+"\t0\t0\t0\t0\t"+str(ProposedHitRank_val)+"\t0\t"+str(t_diff.microseconds/1000000)+"\t"+str(testTrue))	
        text_sim, commonCount = computeTextSimCommonWord_WordDic(Counter(words), Counter(train_words), len(words), len(train_words) )	

        trainDateTime= datetime.strptime(trainItem[5].split("t")[0] ,"%Y-%m-%d") #datetime.now()
        date_diff=trainDateTime-testDateTime
        date_diff=date_diff.days  	
		
        print(str(testpostId)+"\t"+str(trainPostId)+"\t"+str(text_sim)+"\t"+str(ProposedHitRank_val)+"\t"+str(t_diff.microseconds/float(microDivide))+"\t"+str(testTrue)+"\t"+' '.join(words)+"\t"+' '.join(train_words)+"\t"+str(trainDateTime)+"\t"+str(testDateTime)+"\t"+str(date_diff))		
        flag=True
        break
		
      if pathCount>max_hitindex:
        break	
    
    if flag==False:
	     
      '''grams=generateGramsConsucetive(words, min_gram, max_gram)		 
      sortedGrams = list(sorted(grams, key = len, reverse=True))

      flag=False  
      largestGram='' 
      ProposedHitRank=0  
       
      train_Items=aggregateTextItems(sortedGrams, dic_ngram__textItems)
      #print("len(train_Items)", len(train_Items) ) 
      for train_item in train_Items:
        ProposedHitRank+=1	
        
	  
        trainTruelabel=train_item[1]
        train_words=train_item[2]
        trainPostId=train_item[4]	

	  
        if str(trainTruelabel)==str(testTrue):
     
          t12=datetime.now()	  
          t_diff = t12-t11 	
	  
          text_sim, commonCount = computeTextSimCommonWord_WordDic(Counter(words), Counter(train_words), len(words), len(train_words) )	  
          ProposedHitRank_val=int(max(1,math.floor(ProposedHitRank/len(sortedGrams))))	  
      	
          trainDateTime= datetime.strptime(train_item[5].split("t")[0] ,"%Y-%m-%d") #datetime.now()
          date_diff=trainDateTime-testDateTime
          date_diff=date_diff.days      	  
	  
          print(str(testpostId)+"\t"+str(trainPostId)+"\t"+str(text_sim)+"\t"+str(ProposedHitRank_val)+"\t"+str(t_diff.microseconds/float(microDivide))+"\t"+str(testTrue)+"\t"+' '.join(words)+"\t"+' '.join(train_words)+"\t"+str(trainDateTime)+"\t"+str(testDateTime)+"\t"+str(date_diff)) 		
          flag=True		
          break 

        if ProposedHitRank > max_hitindex:
          break'''		  
	  	  
      
      if flag==False:
        #print('not found', 'testTrue', testTrue, 'testwords', words,'postId', postId, 'pathCount', pathCount, 'len(targetClusterIds)', len(targetClusterIds)) 
        t12=datetime.now()	  
        t_diff = t12-t11			
        #print(str(testpostId)+"\t"+"-100"+"\t0\t0\t0\t0\t-100"+"\t0\t"+str(t_diff.microseconds/1000000)+"\t"+str(testTrue))	
        print(str(testpostId)+"\t"+"-100"+"\t0\t"+str(-100)+"\t"+str(t_diff.microseconds/float(microDivide))+"\t"+str(testTrue)+"\t"+' '.join(words)+"\t"+""+"\t"+""+"\t"+""+"\t"+"")		
    	
     	

def test_cluster_biterm(testList_pred_true_words_index_postid, c_bitermsFreqs={}, c_totalBiterms={}, c_wordsFreqs={}, c_totalWords={}, c_txtIds={}, c_clusterVecs={}, txtId_txt={}, last_txtId=0, max_c_id=0, wordVectorsDic={}, dic_clus__id={},  dic_biterm__clusterIds={}, dicTrain_pred__trues={}):
  print("test cluster_bigram")

 

  current_txt_id=last_txtId	
  
  eval_pred_true_txt=[]
  
  line_count=0

  t11=datetime.now()
     
  for item in testList_pred_true_words_index_postid:
    pred=item[0]  
    testTrue=int(item[1])
    words=item[2]
    postId=item[4]	
    bi_terms=construct_biterms(words)
    #print(words, bi_terms, pred)	
    
    current_txt_id+=1

    line_count+=1	
      	
    txtBitermsFreqs=Counter(bi_terms)
    bi_terms_len= len(bi_terms)	
	
    txtWordsFreqs=Counter(words)
    words_len= len(words) 	
	
    text_Vec=[0]*embedDim 	
    if isSemantic==True:	
      X=generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
      text_Vec=X[0]
        	

    targetClusterIds=findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds)
	
    print(targetClusterIds)	
          	
    clusterId=findCloseClusterByTargetClusters(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec, dic_biterm__clusterIds, targetClusterIds)	
	
    if clusterId in dicTrain_pred__trues and  testTrue in dicTrain_pred__trues[clusterId]:
      print('found found', 'clusterId', clusterId, 'testTrue', testTrue, words, postId, 'len', len(dicTrain_pred__trues[clusterId]))	
    else:
      print('not found', 'clusterId', clusterId, 'testTrue', testTrue, words, postId)		
	  
	
    	
	
    #max_c_id=max([max_c_id, clusterId,len(c_bitermsFreqs)])

    #dic_clus__id[clusterId]=max_c_id 	

    #txtId_txt[current_txt_id]=words	

    #c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, dic_biterm__clusterIds=populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id, text_Vec,  dic_biterm__clusterIds) #no need here
	

    '''eval_pred_true_txt.append([clusterId, item[1], item[2]])
    if ignoreMinusOne==True:
      if str(item[1])!='-1':   	
        f.write(str(clusterId)+"	"+str(item[1])+"	"+str(item[2])+"	"+postId+"\n")
    else:
      f.write(str(clusterId)+"	"+str(item[1])+"	"+str(item[2])+"	"+postId+"\n")  	

    
    if line_count%500==0:
       #remove multi-cluster biterms from c_bitermsFreqs   using targetClusterIds; before computing similarity	
       c_bitermsFreqs, c_totalBiterms, c_txtIds, txtBitermsFreqs=removeTargetMultiClusterBiTerms(c_bitermsFreqs, c_totalBiterms, c_txtIds, targetClusterIds, txtBitermsFreqs, dic_biterm__clusterIds)''' 	

    '''if line_count%500==0:

      #print(dic_clus__id)      
      print(len(dic_clus__id)) 	  
      #delete old and small clusters, remove multi-cluster words from clusters
      list_c_sizes=[]
      list_c_ids=[] 	  
      #list_size__cid={}
        	  
      for c_id, txtIds in c_txtIds.items():
        list_c_sizes.append(len(txtIds))
        list_c_ids.append(dic_clus__id[c_id])		
        #list_size__cid[len(txtIds)]=c_id		
      mean_c_size=statistics.mean(list_c_sizes)
      std_c_size=statistics.stdev(list_c_sizes)

      mean_c_id=statistics.mean(list_c_ids)
      std_c_id=statistics.stdev(list_c_ids)	  

      print('preocess', line_count, 'texts', 'mean_c_size', mean_c_size, 'std_c_size', std_c_size)	
      print('preocess', line_count, 'texts', 'mean_c_id', mean_c_id, 'std_c_id', std_c_id)	  
	  
      list_del_cids=[]  
      del_count=0	

	  
      	


      for c_id, orderId in dic_clus__id.items():
        #if float(c_id)<=float(abs(mean_c_id-std_c_id)) or float(orderId)<=float(abs(mean_c_id-std_c_id)):
        if c_id not in c_txtIds:
          continue  		
        c_size=len(c_txtIds[c_id])	  
        if ( float(c_id)<=float(abs(mean_c_id-std_c_id)) or float(orderId)<=float(abs(mean_c_id-std_c_id))) and (c_size<=1 or float(c_size)<=float(abs(mean_c_size-std_c_size))):
        #or float(c_size)>=mean_c_size+std_c_size*1):		
          list_del_cids.append(c_id)  		
	  
	  
	  
		  
      print('#list_del_cids', len(list_del_cids), 'len(c_bitermsFreqs)', len(c_bitermsFreqs))


      listTargetBiterms=[]
	  
      for c_id in list_del_cids:
        BitermsFreqs=c_bitermsFreqs[c_id]  
        for biterm, freq in BitermsFreqs.items():
          if biterm not in dic_biterm__clusterIds:             
            continue			
          clusterIds=set(dic_biterm__clusterIds[biterm])
          if c_id not in clusterIds:			
            continue 			
          clusterIds.remove(c_id)				
          dic_biterm__clusterIds[biterm]=list(clusterIds)		
          if len(dic_biterm__clusterIds[biterm])==0:
            del dic_biterm__clusterIds[biterm]
			
  		
        		
	  
        del c_bitermsFreqs[c_id]
        del c_totalBiterms[c_id]
        del c_txtIds[c_id] 
        del c_wordsFreqs[c_id] 
        del c_totalWords[c_id]
        del dic_clus__id[c_id]
        if isSemantic==True:		
          del c_clusterVecs[c_id]
        		
       
            
			
      
    	
    if line_count%1000==0:  
      print('#######-personal-eval_pred_true_txt', len(eval_pred_true_txt))	 	
      Evaluate(eval_pred_true_txt, ignoreMinusOne)

      t12=datetime.now()	  
      t_diff = t12-t11
      print("total time diff secs=",t_diff.seconds) '''	   

  last_txtId=current_txt_id
  return [c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, dic_clus__id, dic_biterm__clusterIds]   
    	