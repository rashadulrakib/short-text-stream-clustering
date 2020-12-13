from txt_process_util import createTextBinaryGraphMatrixByCommonWord
from txt_process_util import createBinaryWordCooccurenceMatrix
from nltk.tokenize import word_tokenize

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def clusterByConnectedComponentWordCooccurIndex(pred_true_txt_inds):
  _components=0,
  newPred_OldPred_true_text_inds=[]
  
  word_binGraph, dic_word_index, docWords=createBinaryWordCooccurenceMatrix(pred_true_txt_inds)
  word_graph = csr_matrix(word_binGraph)
  
  _components, word_preds = connected_components(csgraph=word_graph, directed=False, return_labels=True)
  
  for i in range(len(pred_true_txt_inds)):
    pred_true_text_ind=pred_true_txt_inds[i]
    
    oldPredLabel=pred_true_text_ind[0]
    trueLabel=pred_true_text_ind[1]
    wordArr=docWords[i] 
    #text=pred_true_text_ind[2]	
    ind=pred_true_text_ind[3]
	
    if len(wordArr)==0:
      continue	
    #wordArr=word_tokenize(text)
    wordId=dic_word_index[wordArr[0]]
    newPredLabel=word_preds[wordId]	
    
    newPred_OldPred_true_text_inds.append([newPredLabel, oldPredLabel, trueLabel, wordArr, ind])

  return [_components, newPred_OldPred_true_text_inds]  

def clusterByConnectedComponentIndex(pred_true_txt_inds):
  _components=0,
  newPred_OldPred_true_text_inds=[]

  binGraph=createTextBinaryGraphMatrixByCommonWord(pred_true_txt_inds)
  graph = csr_matrix(binGraph)
   
  _components, new_preds = connected_components(csgraph=graph, directed=False, return_labels=True)

  #print("--each cluster--")  
  for i in range(len(pred_true_txt_inds)):
    pred_true_text_ind=pred_true_txt_inds[i]
    newPredLabel=new_preds[i]
    oldPredLabel=pred_true_text_ind[0]
    trueLabel=pred_true_text_ind[1]
    text=pred_true_text_ind[2]
    ind=pred_true_text_ind[3] 	
    newPred_OldPred_true_text_inds.append([newPredLabel, oldPredLabel, trueLabel, text, ind])
    #print(newPred_OldPred_true_text_inds[i])	
  
  #print("_components="+str(_components))  
  
  return [_components, newPred_OldPred_true_text_inds]
  
def clusterByConnectedComponent(pred_true_txts):
  _components=0
  newPred_OldPred_true_texts=[]
 
  binGraph=createTextBinaryGraphMatrixByCommonWord(pred_true_txts)
  graph = csr_matrix(binGraph)

  _components, new_preds = connected_components(csgraph=graph, directed=False, return_labels=True)

  #print("--each cluster--")  
  for i in range(len(pred_true_txts)):
    pred_true_text=pred_true_txts[i]
    newPredLabel=new_preds[i]
    oldPredLabel=pred_true_text[0]
    trueLabel=pred_true_text[1]
    text=pred_true_text[2] 	
    newPred_OldPred_true_texts.append([newPredLabel, oldPredLabel, trueLabel, text])
    #print(newPred_OldPred_true_texts[i])	
  
  #print("_components="+str(_components))
  return [_components, newPred_OldPred_true_texts]  