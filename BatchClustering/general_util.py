from groupTxt_ByClass import groupTxtByClass
import statistics
from scipy.spatial.distance import cosine
from groupTxt_ByClass import groupItemsBySingleKeyIndex

def readlistWholeJsonDataSet(datasetName):
  file1=open(datasetName,"r")
  lines = file1.readlines()
  file1.close()
  list_pred_true_words_index=[]
  i=-1  
  for line in lines:
    line=line.strip()  
    n = eval(line)
    id=str(n['Id']).strip()  
    true=str(n['clusterNo']).strip()
    words=str(n['textCleaned']).strip().split(' ')
    if len(true)==0 or len(words)==0:
      continue
    i+=1 	  
    list_pred_true_words_index.append([-1, true, words, i])
  return list_pred_true_words_index

def calculateFarCloseDist(centVec, X):
  centfarD=-10000
  centCloseD=100000
  for vec in X:
    d=cosine(vec, centVec)
    if centfarD<d:
      centfarD=d
    if centCloseD>d:
      centCloseD=d	
  return [centfarD, centCloseD]	  

def findMaxKeyAndValue(dic_tupple_class):
  maxKey=None
  maxValue=-1
  
  for key, items in dic_tupple_class.items():
    _items=len(items)
    if _items>maxValue:
      maxValue=_items
      maxKey=key	  
  
  return [maxKey, maxValue]

def productLexSemanticSims(dic_lex_Sim_CommonWords, dic_semanticSims):
  maxPredLabel_lex=''
  maxSim_lex=-1000
  maxCommon_lex=-1000
  maxPredLabel_Semantic=''
  maxSim_Semantic=-1000
  maxSim_product=-1000
  maxPredLabel_product=''
  minSim_semantic=1000000000
  
  for label, sim_commonCount in dic_lex_Sim_CommonWords.items():
    lex_sim=sim_commonCount[0]
    comCount=sim_commonCount[1]    
    if maxCommon_lex<comCount:
      maxSim_lex=lex_sim
      maxPredLabel_lex=label
      maxCommon_lex=comCount
    if label in dic_semanticSims.keys():
      sem_sim=dic_semanticSims[label]	
      if maxSim_Semantic<sem_sim:
        maxSim_Semantic=sem_sim
        maxPredLabel_Semantic=label	
      if minSim_semantic>sem_sim:
        minSim_semantic=sem_sim	  
	
      #print("do for semantics and lexical=product")	  	
      product_semanSim_lexCount=sem_sim*comCount
      if maxSim_product<product_semanSim_lexCount:
        maxSim_product=product_semanSim_lexCount
        maxPredLabel_product=label		
      
      #print("remove key from semantics")  
      dic_semanticSims.pop(label)	  
     
  for label, sim in dic_semanticSims.items():  
    if maxSim_Semantic<sim:
      maxSim_Semantic=sim
      maxPredLabel_Semantic=label
    if minSim_semantic>sim:
        minSim_semantic=sim	  
  
  return [maxPredLabel_lex, maxSim_lex, maxCommon_lex ,maxPredLabel_Semantic, maxSim_Semantic, maxSim_product, maxPredLabel_product, minSim_semantic]

def maxSim_Count_lex(dic_lex_Sim_CommonWords):
  maxPredLabel_lex=''
  maxSim_lex=-1000
  maxCommon_lex=-1000

  for label, sim_commonCount in dic_lex_Sim_CommonWords.items():
    sim=sim_commonCount[0]
    comCount=sim_commonCount[1]    
    if maxCommon_lex<comCount:
      maxSim_lex=sim
      maxPredLabel_lex=label
      maxCommon_lex=comCount      	  
      	  
  return [maxPredLabel_lex, maxSim_lex, maxCommon_lex]

def maxSim_Count_semantic(dic_semanticSims):
  maxPredLabel_Semantic=''
  maxSim_Semantic=-1000
  
  for label, sim in dic_semanticSims.items():  
    if maxSim_Semantic<sim:
      maxSim_Semantic=sim
      maxPredLabel_Semantic=label
    
  return [maxPredLabel_Semantic, maxSim_Semantic]
  

def findMinMaxLabel(listtuple_pred_true):
  minPred=1000000000
  maxPred=-10000000
  minTrue=1000000000
  maxTrue=-100000000
  for pred_true in listtuple_pred_true:
    predLabel=int(str(pred_true[0]))
    trueLabel=int(str(pred_true[1]))
    if minPred>predLabel:
      minPred=predLabel
    if maxPred<predLabel:
      maxPred=predLabel
    if minTrue>trueLabel:
      minTrue=trueLabel
    if maxTrue<trueLabel:
      maxTrue=trueLabel	
  
  return [minPred, maxPred, minTrue, maxTrue]	  
    	
def extrcatLargeClusterItems(listtuple):
  dic_tupple_class=groupTxtByClass(listtuple, False)
  itemCounts=[]
  items_to_cluster=[]
  items_to_not_cluster=[]  
  for label, tuples in dic_tupple_class.items():
    #if len(tuples)<3:
    #  continue	
    itemCounts.append(len(tuples))
  std=statistics.stdev(itemCounts)
  mean=statistics.stdev(itemCounts)
  
  for label, tuples in dic_tupple_class.items():
    no_items=len(tuples)
    if no_items>=mean+1.2*std:
      items_to_cluster.extend(tuples)
    else:	  
      items_to_not_cluster.extend(tuples)    
  
  return [items_to_cluster, items_to_not_cluster]     

def extractBySingleIndex(items, itemIndex):
  subcolumnItems=[]
  for tuple in items:
    subcolumnItems.append(tuple[itemIndex])
  return subcolumnItems

def findIndexByItems(searchItems, originalList):
  searchIndecies=[]
  
  for item in searchItems:
    searchIndecies.append(originalList.index(item)) 
  
  return searchIndecies

def change_pred_label(list_preds, newPredSeed):
  newlist_preds=[]   
  for pred_tuple in list_preds:
    pred_tuple[0]=str(int(pred_tuple[0])+newPredSeed)	  
    newlist_preds.append(pred_tuple)

  return newlist_preds   

def print_by_group(listtuple_pred_true_text):
  dic_tupple_class=groupTxtByClass(listtuple_pred_true_text, False)
  for label, pred_true_txts in sorted(dic_tupple_class.items()):
    Print_list_pred_true_text(pred_true_txts)
  
def Print_list_pred_true_text(listtuple_pred_true_text):
  for pred_true_text in listtuple_pred_true_text:
    pred=pred_true_text[0]
    true=pred_true_text[1]
    text=pred_true_text[2]	
    print(pred_true_text)
    #print("["+str(pred)+","+str(true)+"],"+str(text))	
	
def split_pred_true_txt_from_list(list_pred_true_text):
    preds=[]
    trues=[]
    texts=[]
    for pred_true_txt in list_pred_true_text:
        preds.append(pred_true_txt[0])
        trues.append(pred_true_txt[1])
        texts.append(pred_true_txt[2])		
    return [preds, trues, texts]
	
def combine_pred_true_txt_from_list(preds, trues, texts):
    list_pred_true_text=[]
    i=-1	
    for pred in preds:
        i=i+1
        tr=trues[i]
        txt=texts[i]
        list_pred_true_text.append([pred, tr, txt])		
        		
    return list_pred_true_text	
	
def extractSeenNotClustered(predsSeen_list_pred_true_words_index, sub_list_pred_true_words_index):
  not_clustered_inds_batch=[]
  
  predSeenIndex=[]
  for item in predsSeen_list_pred_true_words_index:
    predSeenIndex.append(item[3])

  for item in sub_list_pred_true_words_index:
    index = item[3]
    if index in predSeenIndex:
      continue
    not_clustered_inds_batch.append([item[0], item[1], item[2], item[3]])	  
  
  return not_clustered_inds_batch	
