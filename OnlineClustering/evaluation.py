#https://www.geeksforgeeks.org/ml-v-measure-for-evaluating-clustering-performance/

from groupTxt_ByClass import groupItemsBySingleKeyIndex
from sklearn import metrics
import re
import statistics 

def ComputePurity(dic_tupple_class, groupByIndex=1):
 totalItems=0
 maxGroupSizeSum =0
 for label, pred_true_txts in dic_tupple_class.items():
  totalItems=totalItems+len(pred_true_txts)
  #print("pred label="+label+", #texts="+str(len(pred_true_txts)))
  dic_tupple_class_originalLabel=groupItemsBySingleKeyIndex(pred_true_txts, groupByIndex)
  maxMemInGroupSize=-1000000
  maxMemOriginalLabel=""
  for orgLabel, org_pred_true_txts in dic_tupple_class_originalLabel.items():
   #print("orgLabel label="+orgLabel+", #texts="+str(len(org_pred_true_txts)))
   if maxMemInGroupSize < len(org_pred_true_txts):
    maxMemInGroupSize=len(org_pred_true_txts)
    maxMemOriginalLabel=orgLabel
  
  #print("\n")
  #print(str(label)+" purity="+str(maxMemInGroupSize/len(pred_true_txts))+", items="+str(len(pred_true_txts))+", max match#="+str(maxMemInGroupSize))
  #print_by_group(pred_true_txts)  
  maxGroupSizeSum=maxGroupSizeSum+maxMemInGroupSize
  
 purity=maxGroupSizeSum/float(totalItems)
 print("acc majority whole data="+str(purity))
 return purity


def Evaluate(listtuple_pred_true_text, ignoreMinusOne=False):
 
 preds = []
 trues = []
 
 new_listtuple_pred_true_text=[]
 totalwords=0
 
 for pred_true_text in listtuple_pred_true_text:
   if str(pred_true_text[1])=='-1' and ignoreMinusOne==True:
     continue   
 
   totalwords+= len(pred_true_text[2]) 
   preds.append(pred_true_text[0])
   trues.append(pred_true_text[1])
   new_listtuple_pred_true_text.append([pred_true_text[0], pred_true_text[1], pred_true_text[2]])   

 print("evaluate total texts="+str(len(new_listtuple_pred_true_text)))
 
 score = metrics.homogeneity_score(trues, preds)  
 print ("homogeneity_score-whole-data:   %0.8f" % score)

 score=metrics.completeness_score(trues, preds)
 print ("completeness_score-whole-data:   %0.8f" % score)
 
 score=metrics.v_measure_score(trues, preds)
 print ("v_measure_score-whole-data:   %0.8f" % score)
 
 score = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic' )  
 print ("nmi_score-whole-data:   %0.8f" % score)
 
 #score=metrics.adjusted_mutual_info_score(trues, preds)
 #print ("adjusted_mutual_info_score-whole-data:   %0.4f" % score)
 
 #score=metrics.adjusted_rand_score(trues, preds)
 #print ("adjusted_rand_score-whole-data:   %0.4f" % score)
 
 
 dic_tupple_class=groupItemsBySingleKeyIndex(new_listtuple_pred_true_text, 1) #before 0
 dic_tupple_class_true=groupItemsBySingleKeyIndex(new_listtuple_pred_true_text, 1)  #before 1
 print ("pred clusters="+str(len(groupItemsBySingleKeyIndex(new_listtuple_pred_true_text, 0)))+", true clusters="+str(len(dic_tupple_class_true)))
 ComputePurity(dic_tupple_class, 0)
 li=[len(dic_tupple_class_true[x]) for x in dic_tupple_class_true if isinstance(dic_tupple_class_true[x], list)]
 print('min', min(li) , 'max', max(li) , 'median', statistics.median(li)   , 'avg', statistics.mean(li) , 'std',statistics.stdev(li) , 'sum of li', sum(li))
 print('avg words per text', totalwords/len(new_listtuple_pred_true_text), 'totalwords', totalwords, '#texts', len(new_listtuple_pred_true_text))
 '''print("---Pred distribution")
 for key,value in dic_tupple_class.items():
   print(key, len(value))
 print("---True distribution")
 for key,value in dic_tupple_class_true.items():
   print(key, len(value))''' 
   
def Evaluate_old(listtuple_pred_true_text, ignoreMinusOne=False):
 
 preds = []
 trues = []
 
 new_listtuple_pred_true_text=[]
 
 totalwords=0
 
 for pred_true_text in listtuple_pred_true_text:
   if str(pred_true_text[1])=='-1' and ignoreMinusOne==True:
     continue   
 
   preds.append(pred_true_text[0])
   trues.append(pred_true_text[1])
   new_listtuple_pred_true_text.append([pred_true_text[0], pred_true_text[1], pred_true_text[2]]) 
   
   totalwords+= len(pred_true_text[2])
   #print(pred_true_text[2], totalwords)

 print("evaluate total texts="+str(len(new_listtuple_pred_true_text)))
 
 score = metrics.homogeneity_score(trues, preds)  
 print ("homogeneity_score-whole-data:   %0.8f" % score)

 score=metrics.completeness_score(trues, preds)
 print ("completeness_score-whole-data:   %0.8f" % score)
 
 score=metrics.v_measure_score(trues, preds)
 print ("v_measure_score-whole-data:   %0.8f" % score)
 
 score = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic' )  
 print ("nmi_score-whole-data:   %0.8f" % score)
 
 #score=metrics.adjusted_mutual_info_score(trues, preds)
 #print ("adjusted_mutual_info_score-whole-data:   %0.4f" % score)
 
 #score=metrics.adjusted_rand_score(trues, preds)
 #print ("adjusted_rand_score-whole-data:   %0.4f" % score)
 
 
 dic_tupple_class=groupItemsBySingleKeyIndex(new_listtuple_pred_true_text, 0) #before 0
 dic_tupple_class_true=groupItemsBySingleKeyIndex(new_listtuple_pred_true_text, 1)  #before 1
 print ("pred clusters="+str(len(dic_tupple_class))+", true clusters="+str(len(dic_tupple_class_true)))
 ComputePurity(dic_tupple_class)
 li=[len(dic_tupple_class_true[x]) for x in dic_tupple_class_true if isinstance(dic_tupple_class_true[x], list)]
 print('min', min(li) , 'max', max(li) , 'median', statistics.median(li)   , 'avg', statistics.mean(li) , 'std',statistics.stdev(li) , 'sum of li', sum(li))
 print('avg words per text', totalwords/len(new_listtuple_pred_true_text), 'totalwords', totalwords, '#texts', len(new_listtuple_pred_true_text))
 '''print("---Pred distribution")
 for key,value in dic_tupple_class.items():
   print(key, len(value))
 print("---True distribution")
 for key,value in dic_tupple_class_true.items():
   print(key, len(value))'''    
 

