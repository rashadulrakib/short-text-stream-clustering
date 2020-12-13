from nltk.tokenize import word_tokenize
import math
#from compute_util import concatenateDCTvecs
#from data_loader import load_data
#from STC import STC
#from tensorflow.python.keras.optimizers import SGD

#not sure how this function is used
def generate_sent_vecslist_toktextdatas(list_toktextdatas, termsVectorsDic, dim):
 print("Start generating sentence vecs list...")
 print(len(list_toktextdatas))
 
 list_toktextdatavecs = []

 for toktextdatas in list_toktextdatas:
  toktextdatavecs = generate_sent_vecs_toktextdata(toktextdatas, termsVectorsDic, dim)
  list_toktextdatavecs.append(toktextdatavecs)
    
 return list_toktextdatavecs

def generate_weighted_sent_vecs_toktextdata(texts, termsVectorsDic, dicDocFreq, dim):
  totalDocs=len(texts)
  #print("Start generating weighted sentence vecs for multiple sentences...")
  #print(totalDocs)
 
  toktextdatavecs = []

  for i in range(totalDocs): 
    words = word_tokenize(texts[i])
    sum_vecs = [0] * dim
    missingCount=0  
    for word in words:
      if word in termsVectorsDic:
      #sum_vecs=sum_vecs+termsVectorsDic[word]
        for j in range(len(sum_vecs)):
          sum_vecs[j]=sum_vecs[j]+termsVectorsDic[word][j]*math.log(totalDocs/dicDocFreq[word])
      else:
        missingCount=missingCount+1   
 
    for j in range(len(sum_vecs)):
      sum_vecs[j]=sum_vecs[j]/(len(words)-missingCount+1)
    #if missingCount>0:  
    #  print("missing Word Count="+str(missingCount)+", original len="+str(len(words)))
    toktextdatavecs.append(sum_vecs)  
    
  return toktextdatavecs   

def extractWordVecs(words, termsVectorsDic):
  wordVecs=[]
  for word in set(words):
    if word in termsVectorsDic:
      wordVecs.append(termsVectorsDic[word])
  return wordVecs	  
  
def generate_sent_vecs_toktextdata(docsWords, termsVectorsDic, dim):
 #print("Start generating sentence vecs for multiple sentences...")
 #print(len(texts))
 
 toktextdatavecs = []

 for i in range(len(docsWords)): 
  words = docsWords[i]
  sum_vecs = [0] * dim
  missingCount=0  
  for word in words:
   if word in termsVectorsDic:
     for j in range(len(sum_vecs)):
       sum_vecs[j]=(sum_vecs[j]+termsVectorsDic[word][j])
   else:
     missingCount=missingCount+1   
  
  #to use   
  for j in range(len(sum_vecs)):
    sum_vecs[j]=sum_vecs[j]/(len(words)-missingCount+1)
  #end to use   
  #if missingCount>0:  
  #  print("missing Word Count="+str(missingCount)+", original len="+str(len(words)))
  toktextdatavecs.append(sum_vecs)  
    
 return toktextdatavecs
 
def generate_sent_vecs_toktextdata_autoencoder(docsWords, termsVectorsDic, dim, n_clusters=1):
  #print("Start generating sentence vecs for multiple sentences...")
  #print(len(texts))
 
  x=load_data(termsVectorsDic, docsWords)

  dec = STC(dims=[x.shape[-1], 500, 500, 2000, 20], n_clusters=n_clusters)

  dec.pretrain(x=x, y=None, optimizer='adam',
                     epochs=20, batch_size=64,
                     save_dir="data/")
  #dec.model.summary()
  #t0 = time()
  dec.compile(SGD(0.1, 0.9), loss='kld')  
   
    
  return x 

def generate_sent_vecs_toktextdata_DCT(texts, termsVectorsDic, dim, k):
  #print("Start generating sentence DCT vecs for multiple sentences...")
  #print(len(texts))
  toktextdatavecs=[]  
  for text in texts:
    words = word_tokenize(text)
    wordVecs=extractWordVecs(words, termsVectorsDic)
    if len(wordVecs)==0:
      print("len(wordVecs)==0")	
      print(text, len(wordVecs))
      continue	  
    dct1dArr=concatenateDCTvecs(wordVecs, k)   	
    toktextdatavecs.append(dct1dArr)
	
  return toktextdatavecs	
    	
    	
      
