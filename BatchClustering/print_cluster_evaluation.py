from groupTxt_ByClass import groupTxtByClass
from sklearn import metrics
from compute_custom_purity import ComputePurity
from read_pred_true_text import ReadPredTrueText

def printClusterEvaluation_list(listtuple_pred_true_text):
  preds = []
  trues = []
  for pred_true_text in listtuple_pred_true_text:
    preds.append(pred_true_text[0])
    trues.append(pred_true_text[1])
 
  score = metrics.homogeneity_score(trues, preds)  
  print ("homogeneity_score-whole-data:   %0.4f" % score)   			
  score = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic' )  
  print ("nmi_score-whole-data:   %0.4f" % score)
  dic_tupple_class=groupTxtByClass(listtuple_pred_true_text, False)
  ComputePurity(dic_tupple_class)

def printClusterEvaluation_file(pred_true_text_file):
  listtuple_pred_true_text=ReadPredTrueText(pred_true_text_file)
  printClusterEvaluation_list(listtuple_pred_true_text)
  
def appendResultFile(pred_true_text_file, fileName):
  f = open(fileName, 'a')
  for tuple in pred_true_text_file:
    f.write(str(tuple[0])+"	"+str(tuple[1])+"	"+str(tuple[2])+"\n")		
  f.close()  
