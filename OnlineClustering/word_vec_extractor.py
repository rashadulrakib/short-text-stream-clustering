from nltk.stem import PorterStemmer 

def extractAllWordVecsPartialStemming(embeddingfile, embedDim, words):
  wordVectorsDic = {}
  ps = PorterStemmer()
  #ps.stem(w)
  t=[]
  for word in words:
    t.append(ps.stem(word))
  words=set(t)	
  
  print("total words in extractAllWordVecsPartial=", len(words))   
  
  file = open(embeddingfile, 'r', encoding="utf8") 
  #vectorlines=[]
  lineProgCount = 0
  while True:
    line = file.readline().strip()
    if len(line)==0:
      break	
    if not line: 
      break
    lineProgCount+=1
    if lineProgCount % 100000 ==0:
      print(lineProgCount)	  
    #vectorlines.append(line)
    veclinearr = line.split()
    if len(veclinearr) < 20:
      continue
    vecword = ps.stem(veclinearr[0])
    if vecword in words:	
      wordVectorsDic[vecword]=list(map(float, veclinearr[1:]))	      
  file.close()
  
  print("wordVectorsDic length", len(wordVectorsDic), len(words))
  return wordVectorsDic 


def extractAllWordVecsPartial(embeddingfile, dim, words=[]):
  wordVectorsDic = {}

  print("total words in extractAllWordVecsPartial=", len(words))   
  
  file = open(embeddingfile, 'r', encoding="utf8") 
  #vectorlines=[]
  lineProgCount = 0
  while True:
    line = file.readline().strip()
    if len(line)==0:
      break	
    if not line: 
      break
    lineProgCount+=1
    if lineProgCount % 100000 ==0:
      print(lineProgCount)	  
    #vectorlines.append(line)
    veclinearr = line.split()
    if len(veclinearr) < 20:
      continue
    vecword = veclinearr[0]
    vecnumbers = list(map(float, veclinearr[1:]))
    if vecword in words:	
      wordVectorsDic[vecword]=vecnumbers	      
  file.close()
  
  print("wordVectorsDic length", len(wordVectorsDic), len(words))
  return wordVectorsDic 
   

def extractAllWordVecs(embeddingfile, dim):
  wordVectorsDic = {}
  
  file = open(embeddingfile, 'r', encoding="utf8") 
  vectorlines=[]
  while True:
    line = file.readline().strip()
    if len(line)==0:
      break	
    if not line: 
      break
    vectorlines.append(line)	  
     
	  
  file.close()
  
  lineProgCount = 0

  for vecline in vectorlines:
    veclinearr = vecline.strip().split()
    lineProgCount=lineProgCount+1
    if lineProgCount % 100000 ==0:
      print(lineProgCount)
 
    if len(veclinearr) < 20:
      continue
    vecword = veclinearr[0]
    vecnumbers = list(map(float, veclinearr[1:]))
    wordVectorsDic[vecword]=vecnumbers 	  
  
  print("wordVectorsDic length", len(wordVectorsDic))
  return wordVectorsDic  

'''def extractAllWordVecs(embeddingfile, dim):
  wordVectorsDic = {}
  file=open(embeddingfile,"r", encoding="utf8")
  vectorlines = file.readlines()
  file.close()
  
  lineProgCount = 0

  for vecline in vectorlines:
    veclinearr = vecline.strip().split()
    lineProgCount=lineProgCount+1
    if lineProgCount % 100000 ==0:
      print(lineProgCount)
 
    if len(veclinearr) < 20:
      continue
    vecword = veclinearr[0]
    vecnumbers = list(map(float, veclinearr[1:]))
    wordVectorsDic[vecword]=vecnumbers 	  
  
  print("wordVectorsDic length", len(wordVectorsDic))
  return wordVectorsDic  '''

def extract_word_vecs_list(list_toktextdatas, embeddingfile, dim):
 print("list_toktextdatas", len(list_toktextdatas))
 
 terms = [] 

 for toktextdatas in list_toktextdatas:
  for word_tokens in toktextdatas:
   terms.extend(word_tokens)

 terms=set(terms)
 print("terms length", len(terms))

 file=open(embeddingfile,"r")
 vectorlines = file.readlines()
 file.close()

 lineProgCount = 0
 termsVectors = []

 for vecline in vectorlines:
  vecarr = vecline.strip().split()
  lineProgCount=lineProgCount+1
  if lineProgCount % 100000 ==0:
   print(lineProgCount)
 
  if len(vecarr) < 20:
   continue
 
  w2vecword = vecarr[0]
  if w2vecword in terms:
   termsVectors.append(vecline)

 del vectorlines

 termsVectorsDic = {}

 for vecline in termsVectors:
  veclinearr = vecline.strip().split()
  vecword = veclinearr[0]
  vecnumbers = list(map(float, veclinearr[1:]))
  termsVectorsDic[vecword]=vecnumbers 
  
 print("termsVectorsDic length", len(termsVectorsDic))
 
 return termsVectorsDic


def extract_word_vecs(toktextdatas, embeddingfile, dim):
 print("toktextdatas", len(toktextdatas))
 
 terms = [] 

 for word_tokens in toktextdatas:
  terms.extend(word_tokens)

 terms=set(terms)
 print("terms length", len(terms))

 file=open(embeddingfile,"r")
 vectorlines = file.readlines()
 file.close()

 lineProgCount = 0
 termsVectors = []

 for vecline in vectorlines:
  vecarr = vecline.strip().split()
  lineProgCount=lineProgCount+1
  if lineProgCount % 100000 ==0:
   print(lineProgCount)
 
  if len(vecarr) < 20:
   continue
 
  w2vecword = vecarr[0]
  if w2vecword in terms:
   termsVectors.append(vecline)

 del vectorlines

 termsVectorsDic = {}

 for vecline in termsVectors:
  veclinearr = vecline.strip().split()
  vecword = veclinearr[0]
  vecnumbers = list(map(float, veclinearr[1:]))
  termsVectorsDic[vecword]=vecnumbers 
  
 print("termsVectorsDic length", len(termsVectorsDic))
 
 return termsVectorsDic


def populateTermVecs(terms, embeddingfile, dim):
 termsVectorsDic = {}

 file=open(embeddingfile,"r")
 vectorlines = file.readlines()
 file.close()

 lineProgCount = 0
 termsVectors = []

 for vecline in vectorlines:
  vecarr = vecline.strip().split()
  lineProgCount=lineProgCount+1
  if lineProgCount % 100000 ==0:
   print(lineProgCount)
 
  if len(vecarr) < 20:
   continue
 
  w2vecword = vecarr[0]
  if w2vecword in terms:
   termsVectors.append(vecline)

 del vectorlines

 termsVectorsDic = {}

 for vecline in termsVectors:
  veclinearr = vecline.strip().split()
  vecword = veclinearr[0]
  vecnumbers = list(map(float, veclinearr[1:]))
  termsVectorsDic[vecword]=vecnumbers 
  
 print("termsVectorsDic length", len(termsVectorsDic))

 return termsVectorsDic
  
