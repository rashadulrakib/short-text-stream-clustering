def groupItemsBySingleKeyIndex(listItems, keyIndex):
  dic_itemGroups={}
  for item in listItems:
    key=str(item[keyIndex]) 
    dic_itemGroups.setdefault(key, []).append(item)
   
  return dic_itemGroups

def groupTxtByClass_Txtindex(listtuple_pred_true_text, isByTrueLabel):
  dic_tupple_class_Txtindex={}
  if isByTrueLabel == False:
    #print("isByPredLabel")
    i=-1	
    for tuple_pred_true_text in listtuple_pred_true_text:
      i=i+1
      predLabel = str(tuple_pred_true_text[0])
      trueLabel = tuple_pred_true_text[1]
      txt = tuple_pred_true_text[2]  
      dic_tupple_class_Txtindex.setdefault(predLabel, []).append([predLabel, trueLabel, txt, i])
  else:
    #print("isByTrueLabel")
    i=-1
    for tuple_pred_true_text in listtuple_pred_true_text:
      i=i+1
      predLabel = tuple_pred_true_text[0]
      trueLabel = str(tuple_pred_true_text[1])
      txt = tuple_pred_true_text[2]  
      dic_tupple_class_Txtindex.setdefault(trueLabel, []).append([predLabel, trueLabel, txt, i])
	  
  return dic_tupple_class_Txtindex 

##group txt by class
def groupTxtByClass(listtuple_pred_true, isByTrueLabel):
 dic_tupple_class = {}
 if isByTrueLabel == False:
  for tuple_pred_true in listtuple_pred_true:
   predLabel = str(tuple_pred_true[0])
   dic_tupple_class.setdefault(predLabel, []).append(tuple_pred_true)
 else:
  for tuple_pred_true in listtuple_pred_true:
   trueLabel = str(tuple_pred_true[1])  
   dic_tupple_class.setdefault(trueLabel, []).append(tuple_pred_true)  

 #for key, value in dic_tupple_class.items():
 # print(key, len(value))

 return dic_tupple_class
