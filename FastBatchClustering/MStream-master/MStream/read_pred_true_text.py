import re

def ReadPredTrueText(dataFilePredTrueTxt, ignoreMinusOne=False): 
 print("pred_true_text_file name="+dataFilePredTrueTxt)
 listtuple_pred_true_text=[]
 file1=open(dataFilePredTrueTxt,"r")
 lines = file1.readlines()
 file1.close()
 
 for line in lines:
  line = line.strip()
  if len(line)==0:
   continue
  arr = re.split("\t", line)
  if len(arr)!=3:
    continue   
  if ignoreMinusOne==True and str(arr[1])=='-1':
    continue
	
  tupPredTrueTxt = [arr[0], arr[1], arr[2]] 
  listtuple_pred_true_text.append(tupPredTrueTxt)
  
 return listtuple_pred_true_text
 
 
def ReadPredTrueTextPostid(dataFilePredTrueTxt, ignoreMinusOne=False): 
 print("pred_true_text_file name="+dataFilePredTrueTxt)
 list_pred_true_text_postid=[]
 file1=open(dataFilePredTrueTxt,"r")
 lines = file1.readlines()
 file1.close()
 
 for line in lines:
  line = line.strip()
  if len(line)==0:
   continue
  arr = re.split("\t", line)
  if len(arr)!=4:
    continue   
  if ignoreMinusOne==True and str(arr[1])=='-1':
    continue
	
  tupPredTrueTxtPostId = [arr[0], arr[1], arr[2], arr[3]] 
  list_pred_true_text_postid.append(tupPredTrueTxtPostId)
  
 return list_pred_true_text_postid 