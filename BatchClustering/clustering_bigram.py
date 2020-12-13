from evaluation import Evaluate
from collections import Counter

def concatWordsSort(words):
  words.sort()
  combinedWord=' '.join(words)
  return combinedWord

def construct_biterms(words):
  bi_terms=[]
  for j in range(len(words)):
    for k in range(j+1,len(words)): 				   
      bi_terms.append(concatWordsSort([words[j], words[k]]))
	  
  return bi_terms

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

def findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len):
  clusterId=-1
  max_sim=0

  for clusId, clusBitermsFreqs in c_bitermsFreqs.items():
    #txtIds=c_txtIds[clusId]
    #wordsFreqs=c_wordsFreqs[clusId]
    txt_j_len= c_totalBiterms[clusId]	
    	
    text_sim, commonCount=computeTextSimCommonWord_WordDic(txtBitermsFreqs, clusBitermsFreqs, bi_terms_len, txt_j_len)
    if text_sim> max_sim:
      max_sim=text_sim
      clusterId=clusId	  
      	  
  

  if clusterId==-1:
    clusterId=len(c_bitermsFreqs)+1    
	
    	
    	
      
  
  return clusterId
  

def populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id):

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
  
  for word, wordFreq in txtWordsFreqs.items():
    if word not in c_wordsFreqs[clusterId]:
      c_wordsFreqs[clusterId][word]=0
    c_wordsFreqs[clusterId][word]+=wordFreq
    

  return [c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords]    
  	
	  

def cluster_bigram(list_pred_true_words_index, c_bitermsFreqs={}, c_totalBiterms={}, c_wordsFreqs={}, c_totalWords={}, c_txtIds={}, txtId_txt={}, last_txtId=0):
  print("cluster_bigram")

  current_txt_id=last_txtId	
  
  eval_pred_treu_txt=[]
  
  f = open("result/personal_cluster_biterm.txt", 'a')
     
  for item in list_pred_true_words_index:
    words=item[2]
    bi_terms=construct_biterms(words)
    
    current_txt_id+=1	
      	
    txtBitermsFreqs=Counter(bi_terms)
    bi_terms_len= len(bi_terms)	
	
    txtWordsFreqs=Counter(words)
    words_len= len(words) 	
    
    clusterId=findCloseCluster(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len)

    txtId_txt[current_txt_id]=words	

    c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords=populateClusterFeature(c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords, txtBitermsFreqs, bi_terms_len, txtWordsFreqs, words_len, clusterId, current_txt_id)

    #print(clusterId, bi_terms, c_bitermsFreqs, c_totalBiterms, c_txtIds, c_wordsFreqs, c_totalWords)	
    #print(clusterId, bi_terms)
    
    eval_pred_treu_txt.append([clusterId, item[1], item[2]])
    f.write(str(clusterId)+"	"+str(item[1])+"	"+str(item[2])+"\n")	

    	
  f.close()
  
  print('#######-personal-eval_pred_treu_txt', len(eval_pred_treu_txt))	 	
  Evaluate(eval_pred_treu_txt) 

  last_txtId=current_txt_id
  return [c_bitermsFreqs, c_totalBiterms, c_wordsFreqs, c_totalWords, c_txtIds, txtId_txt, last_txtId]  
    	