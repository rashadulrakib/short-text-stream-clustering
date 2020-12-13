from evaluation import Evaluate_old
from collections import Counter
import statistics 
from random import randint
from datetime import datetime
from sent_vecgenerator import generate_sent_vecs_toktextdata
from txt_process_util import construct_biterms
from txt_process_util import generateGramsConsucetive
from txt_process_util import semanticSims
from operator import add 


embedDim=50

ignoreMinusOne=True






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

def computeTextSimCommonWord_WordArr(txt_i_wordArr, txt_j_wordArr): #not used
  
  txt_i_len=len(txt_i_wordArr)
  txt_j_len=len(txt_j_wordArr)  
  
  words_i= collections.Counter(txt_i_wordArr)#assume words_i small
  words_j= collections.Counter(txt_j_wordArr) 
  
  text_sim, commonCount=computeTextSimCommonWord_WordDic(words_i, words_j, txt_i_len, txt_j_len)
 
  return [text_sim, commonCount]


def findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds):
  targetClusterIds=[]
  
  for biterm, freq in txtBitermsFreqs.items():
    if biterm not in dic_biterm__clusterIds:
      continue
    targetClusterIds.extend(dic_biterm__clusterIds[biterm])     	  

  targetClusterIds=set(targetClusterIds)
  return targetClusterIds 
  
def findCloseClusterByTargetClusters(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec, targetClusterIds):
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
    
  #elif clusterId_lex!=clusterId_sem:
  #  clusterId=max_c_id+1
  #elif clusterId_lex==clusterId_sem:
  #  clusterId=clusterId_lex
  elif max_sim_lex>=mean_lex_sim+std_lex_sim: # and randint(0,1)==1: work
    clusterId=clusterId_lex    
  else:
    clusterId=clusterId_lex







  #print(text_Vec, clusterId_lex, clusterId_sem, clusterId)
  return clusterId
  #return clusterId_lex  

def findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec):
  clusterId_lex=-1
  clusterId_sem=-1
  clusterId=-1
  max_sim=0
  max_sim_lex=0

  dic_lexicalSims={}

  for clusId, clusBitermsFreqs in c_bitermsFreqs.items():
    txt_j_len= c_totalBiterms[clusId]	

    text_sim, commonCount=computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len, txt_j_len)
    if text_sim> max_sim:
      max_sim=text_sim
      clusterId=clusId	  

    if text_sim> max_sim_lex:
      max_sim_lex=text_sim
      clusterId_lex=clusId
    dic_lexicalSims[clusId]=text_sim 	  

  #dic_semanticSims, clusterId_sem, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, c_clusterVecs, c_txtIds)  

  lex_sim_values= list(dic_lexicalSims.values())
  #sem_sim_values= list(dic_semanticSims.values()) 

  mean_lex_sim=0
  std_lex_sim=0

  if len(lex_sim_values)>2:
    mean_lex_sim=statistics.mean(lex_sim_values)
    std_lex_sim=statistics.stdev(lex_sim_values)  

  #mean_sem_sim=statistics.mean(sem_sim_values)
  #std_sem_sim=statistics.stdev(sem_sim_values)



  '''if clusterId_lex==-1: #or clusterId_sem==-1:
    #clusterId=len(c_bitermsFreqs)+1    
    clusterId=max_c_id+1	
  #elif clusterId_lex!=clusterId_sem:
  #  clusterId=max_c_id+1
  #elif clusterId_lex==clusterId_sem:
  #  clusterId=clusterId_lex
  elif max_sim_lex>=mean_lex_sim+std_lex_sim: # and randint(0,1)==1:
    clusterId=clusterId_lex    
  else:
    clusterId=max_c_id+1''' 
	
  if (max_sim_lex>=mean_lex_sim+std_lex_sim) and clusterId_lex!=-1: # and randint(0,1)==1:
    clusterId=clusterId_lex 

  if clusterId_lex==-1: #or clusterId_sem==-1:
    #clusterId=len(c_bitermsFreqs)+1    
    clusterId=max_c_id+1 	







  #print(text_Vec, clusterId_lex, clusterId_sem, clusterId)
  return clusterId
  #return clusterId_lex

'''def findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec):
  clusterId_lex=-1
  clusterId_sem=-1
  clusterId=-1
  max_sim_lex=0
  max_sim_sem=0
  
  #if randint(0,1)==1:
  #  clusterId=max_c_id+1
    #print('random cluster', clusterId)	
  #  return clusterId	
     
  dic_lexicalSims={}
  for clusId, clusBitermsFreqs in c_bitermsFreqs.items():
    txt_j_len= c_totalBiterms[clusId]	    	
    text_sim, commonCount=computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len, txt_j_len)
    if text_sim> max_sim_lex:
      max_sim_lex=text_sim
      clusterId_lex=clusId
    dic_lexicalSims[clusId]=text_sim 
  if clusterId_lex==-1:
    return max_c_id+1 	
  lex_sim_values= list(dic_lexicalSims.values())
  mean_lex_sim=0
  std_lex_sim=0
  if len(lex_sim_values)>2:
    mean_lex_sim=statistics.mean(lex_sim_values)
    std_lex_sim=statistics.stdev(lex_sim_values) 
  if max_sim_lex>=mean_lex_sim+std_lex_sim:
    return clusterId_lex	
   
	
	
	
	
  
  dic_semanticSims, clusterId_sem, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, c_clusterVecs, c_txtIds)
  if clusterId_sem==-1:
    return max_c_id+1
  sem_sim_values= list(dic_semanticSims.values())
  mean_sem_sim=0
  std_sem_sim=0
  if len(sem_sim_values)>2:  
    mean_sem_sim=statistics.mean(sem_sim_values)
    std_sem_sim=statistics.stdev(sem_sim_values)
  if mean_sem_sim>=mean_sem_sim+std_sem_sim:
    return clusterId_sem 		
  	

  max_SR_sim=0
  SR_clusId=-1 
  dic_SRSims={}
  for clusId, lex_sim in dic_lexicalSims.items(): 
    sem_sim=dic_semanticSims[clusId]
    if max(lex_sim,sem_sim)<=0:
      continue	
    sim=min(lex_sim,sem_sim)/max(lex_sim,sem_sim)*(lex_sim+sem_sim)	
    if sim>max_SR_sim:
      max_SR_sim=sim
      SR_clusId=clusId
    dic_SRSims[clusId]=sim 	  
  if SR_clusId==-1:  
    return max_c_id+1 
  SR_sim_values= list(dic_SRSims.values())
  mean_SR_sim=0
  std_SR_sim=0
  if len(SR_sim_values)>2:  
    mean_SR_sim=statistics.mean(SR_sim_values)
    std_SR_sim=statistics.stdev(SR_sim_values)
  
  
  
  
  if max_SR_sim>=mean_SR_sim+std_SR_sim: # and randint(0,1)==1:
    return SR_clusId
  #elif mean_sem_sim>=mean_sem_sim+std_sem_sim:
  #  return clusterId_sem 	
  

   	
	
    	
    	
      
  #print(text_Vec, clusterId_lex, clusterId_sem, clusterId)
  #return clusterId
  return max_c_id+1'''
  

def populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id, text_Vec, dic_biterm__clusterIds):

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
	
  
  
  
  '''if clusterId not in c_clusterVecs:
    c_clusterVecs[clusterId]=text_Vec
  else:
    c_clusterVecs[clusterId] = list(map(add, text_Vec, c_clusterVecs[clusterId]))'''
    
    

  return [c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, dic_biterm__clusterIds]    
  	
	  

def cluster_biterm(f, list_pred_true_words_index, c_bitermsFreqs={}, c_totalBiterms={}, c_wordsFreqs={}, c_totalWords={}, c_txtIds={}, c_clusterVecs={}, txtId_txt={}, last_txtId=0, max_c_id=0, wordVectorsDic={}, dic_clus__id={}, dic_biterm__clusterIds={}):
  print("cluster_bigram")

  current_txt_id=last_txtId	
  
  eval_pred_treu_txt=[]
  
  line_count=0

  t11=datetime.now()
     
  for item in list_pred_true_words_index:
    words=item[2]
    bi_terms=construct_biterms(words)
    #bi_terms=generateGramsConsucetive(words,2,2) 	
    
    current_txt_id+=1

    line_count+=1	
      	
    txtBitermsFreqs=Counter(bi_terms)
    bi_terms_len= len(bi_terms)	
	
    txtWordsFreqs=Counter(words)
    words_len= len(words) 	
	
    #X=generate_sent_vecs_toktextdata([words], wordVectorsDic, embedDim)
    #text_Vec=X[0]
    text_Vec=[0]*embedDim    

    targetClusterIds=findTargetClusters(txtBitermsFreqs, dic_biterm__clusterIds)  	
    
    
    #clusterId=findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec)
	
    clusterId=findCloseClusterByTargetClusters(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, max_c_id, text_Vec, targetClusterIds)	
	
    max_c_id=max([max_c_id, clusterId,len(c_bitermsFreqs)])

    dic_clus__id[clusterId]=max_c_id 	

    txtId_txt[current_txt_id]=words	

    c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, dic_biterm__clusterIds=populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, c_clusterVecs, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id, text_Vec, dic_biterm__clusterIds)
    
    eval_pred_treu_txt.append([clusterId, item[1], item[2]])
    if ignoreMinusOne==True:
      if str(item[1])!='-1':   	
        f.write(str(clusterId)+"	"+str(item[1])+"	"+str(item[2])+"\n")
    else:
      f.write(str(clusterId)+"	"+str(item[1])+"	"+str(item[2])+"\n")  	

    

    if line_count%500==0:

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

      #print('preocess', line_count, 'texts', 'mean_c_size', mean_c_size, 'std_c_size', std_c_size)	
      #print('preocess', line_count, 'texts', 'mean_c_id', mean_c_id, 'std_c_id', std_c_id)	  
	  
      list_del_cids=[]  
      del_count=0	

	  
      '''for c_id, txtIds in c_txtIds.items():
        c_size=	len(txtIds)
        ##print('c_id=', c_id, 'c_size=', c_size)		
        #if c_size<=2 :#or del_count<15:
        #  list_del_cids.append(c_id)
        #  print('delete cluster=',c_id, '#size=', c_size) 		  		  
          #del_count+=1	  
        	  
        #if c_size<=1 or float(c_size)<=float(abs(mean_c_size-std_c_size)) or float(c_size)>=mean_c_size+std_c_size or float(c_size)>=mean_c_size:  		
        #if float(c_size)<float(abs(mean_c_size)):
        #  list_del_cids.append(c_id)
          #print('delete cluster=',c_id, '#size=', c_size)  		  
		  
        #float(c_id)<=float(abs(mean_c_id-std_c_id))		  
        if (c_size<=1 or float(c_size)<=float(abs(mean_c_size-std_c_size))) or float(c_size)>=mean_c_size: #and del_count<100:  		   		
          list_del_cids.append(c_id)
          del_count+=1
        		
        #  print('delete cluster=',c_id, '#size=', c_size) 		  
          
      #list_c_sizes.sort(reverse=True)
	  
      #for c_size in list_c_sizes[0:20]:
      #  list_del_cids.append(list_size__cid[c_size])''' 	


      for c_id, orderId in dic_clus__id.items():
        c_size=len(c_txtIds[c_id])	  
        #if (float(c_id)<=float(abs(mean_c_id-std_c_id)) or float(orderId)<=float(abs(mean_c_id-std_c_id))):
        #if (c_size<=1 or float(c_size)<=float(abs(mean_c_size-std_c_size)) or float(c_size)>=mean_c_size+std_c_size*1):
        if (float(c_id)<=float(abs(mean_c_id-std_c_id)) or float(orderId)<=float(abs(mean_c_id-std_c_id))) and (c_size<=1 or float(c_size)<=float(abs(mean_c_size-std_c_size)) or float(c_size)>=mean_c_size+std_c_size):		
          list_del_cids.append(c_id)  		
	  
	  
	  
		  
      #print('#list_del_cids', len(list_del_cids), 'len(c_bitermsFreqs)', len(c_bitermsFreqs))

      
	  
      for c_id in list_del_cids:
        del c_bitermsFreqs[c_id]
        del c_totalBiterms[c_id]
        del c_txtIds[c_id] 
        del c_wordsFreqs[c_id] 
        del c_totalWords[c_id]
        del dic_clus__id[c_id]		
        #del c_clusterVecs[c_id]
        
      
      
    	
    if line_count%1000==0:  
      print('#######-process texts=', len(eval_pred_treu_txt))	 	
      Evaluate_old(eval_pred_treu_txt, ignoreMinusOne)

      t12=datetime.now()	  
      t_diff = t12-t11
      print("total time diff secs=",t_diff.seconds) 	   

  last_txtId=current_txt_id
  return [c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, c_clusterVecs, txtId_txt, last_txtId, dic_clus__id, dic_biterm__clusterIds]  
    	