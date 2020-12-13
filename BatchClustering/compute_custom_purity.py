from groupTxt_ByClass import groupTxtByClass

def ComputePurity(dic_tupple_class):
 totalItems=0
 maxGroupSizeSum =0
 minIntVal = -1000000  
 for label, pred_true_txts in dic_tupple_class.items():
  totalItems=totalItems+len(pred_true_txts)
  #print("pred label="+label+", #texts="+str(len(pred_true_txts)))
  dic_tupple_class_originalLabel=groupTxtByClass(pred_true_txts, True)
  maxMemInGroupSize=minIntVal
  maxMemOriginalLabel=""
  for orgLabel, org_pred_true_txts in dic_tupple_class_originalLabel.items():
   #print("orgLabel label="+orgLabel+", #texts="+str(len(org_pred_true_txts)))
   if maxMemInGroupSize < len(org_pred_true_txts):
    maxMemInGroupSize=len(org_pred_true_txts)
    maxMemOriginalLabel=orgLabel
  
  #print("\n")
  maxGroupSizeSum=maxGroupSizeSum+maxMemInGroupSize
  
 purity=maxGroupSizeSum/float(totalItems)
 print("purity majority whole data="+str(purity)+", totalItems="+str(totalItems)+", clusters="+str(len(dic_tupple_class)))
 return purity