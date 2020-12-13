def writePredTrueTexts(instFile, tup_pred_true_txts):
 file1=open(instFile,"w")
 for tup_pred_true_txt in tup_pred_true_txts:
  file1.write(tup_pred_true_txt[0]+"\t"+tup_pred_true_txt[1]+"\t"+tup_pred_true_txt[2]+"\n")  

 file1.close()