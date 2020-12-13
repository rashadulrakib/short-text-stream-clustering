from txt_process_util import getDocFreqWithIds

def textIdsUsedInFlagArray(textIds,textFlags):
  for textId in textIds:
    if textFlags[textId]==1:
      return True
	  
  return False	  

def clusterByLeadingOnOverlappingWords(texts):
  print("clusterByLeadingOnOverlappingWords", len(texts))
  clusterLabels=[0 for i in range(len(texts))] #all are in same cluster
  dicDocFreq,dicTextIds=getDocFreqWithIds(texts) #no text is used
  textFlags = [0 for i in range(len(texts))]
  dicDocFreq = {r: dicDocFreq[r] for r in sorted(dicDocFreq, key=dicDocFreq.get, reverse=True)}
  sortedKeys=list(dicDocFreq.keys())
  #print(sortedKeys)
  #print(dicTextIds)
  maxWordKey=sortedKeys[0]
  maxWordFreq=dicDocFreq[maxWordKey]
  maxTextIds=dicTextIds[maxWordKey]
  
  for ind in maxTextIds:
    textFlags[ind]=1
    clusterLabels[ind]=1

  nextClusterLabel=2
	
  totalDocsClustered=len(maxTextIds)
  
  #print(textFlags)
  for i in range(1,len(sortedKeys)):
    wordKey=sortedKeys[i]
    textIds=dicTextIds[wordKey]
    if textIdsUsedInFlagArray(textIds,textFlags):
      continue
    totalDocsClustered=totalDocsClustered+len(textIds)
    for ind in textIds:
      textFlags[ind]=1
      clusterLabels[ind]=nextClusterLabel
	
    if totalDocsClustered>=len(texts):
      break	
    nextClusterLabel=nextClusterLabel+1 	  
    
      	  
  
  
  
  
  
  
    
  
  '''
  aaa bbb cc
  aaa dd ee
  ff gg hh cc
  ff kk l
  df[aaa]=[(1,2),2] #doc ids, num_of_docs
  df[ff]=[(3,4),2] #doc ids, num_of_docs
  df[cc]=[(1,3),2]
  
  flag_doc_used_arr=[0,0,0,0]
  df[aaa]=max(df...)
  total_docs_used=2
  flag_doc_used_arr=[1,2,0,0]
  
  for each df with_high num_of_docs (sorted)
    df[cc]=(1,3)
    flag_doc_used_arr[1] is already used, so do not use df[cc]
    total_docs_used=total_docs_used+df[ff]
    if total_docs_used >=4: break 	
  '''
  #print("len(clusterLabels)", len(clusterLabels), totalDocsClustered)
  return [clusterLabels, totalDocsClustered] 