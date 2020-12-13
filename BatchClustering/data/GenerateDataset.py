import re
import random
from groupTxt_ByClass import groupItemsBySingleKeyIndex

def SampleData(dataset, percent, random=False):
  file1=open(dataset,"r")
  linesNews = file1.readlines()
  file1.close()
  if random==False:
    return linesNews
  length=len(linesNews)-1
  total=int(length*percent)
  randIndecies = [random.randint(0,length) for i in range(total)]
  sublist = [linesNews[index] for index in randIndecies]
  return sublist

def ExtractPredTrueText(mstream_sublist):
  pred_true_texts=[]
  for item in mstream_sublist:
    n = eval(item)
    pred=str(n['Id']).strip()  
    true=str(n['clusterNo']).strip()
    text=str(n['textCleaned']).strip()
    if len(true)==0 or len(text)==0:
      continue 	
    pred_true_texts.append([pred, true, text])
  return pred_true_texts	

def RenameTrueLabel(pred_true_texts,startTrueSeed, startIdSeed):
  lastTrueLabel=startTrueSeed
  lastId=startIdSeed
  renamed_pred_true_texts=[]  
  dic_group = groupItemsBySingleKeyIndex(pred_true_texts,1)
  groupsLen=len(dic_group)
  #print(groupsLen)
  for trueLabel, items_pred_true_text in dic_group.items():
    lastTrueLabel=lastTrueLabel+1
    for item_pred_true_text in items_pred_true_text:
      lastId=lastId+1
      renamed_pred_true_texts.append([str(lastId).zfill(6), str(lastTrueLabel), item_pred_true_text[2]])	  
	  
  return [lastTrueLabel, lastId, renamed_pred_true_texts]  

def TransformToMstream(renamed_pred_true_texts):
  trans_pred_true_texts=[]  
  for pred_true_text in renamed_pred_true_texts:
    newdata='{"Id": "'+pred_true_text[0]+'", "clusterNo": '+pred_true_text[1]+', "textCleaned":"'+pred_true_text[2]+'"}'
    print(newdata)	
    trans_pred_true_texts.append(newdata)

  return trans_pred_true_texts  	
    

sublist=SampleData("News-T", 1.00)
pred_true_texts=ExtractPredTrueText(sublist)
lastTrueLabel, lastId, renamed_pred_true_texts=RenameTrueLabel(pred_true_texts,0, 0)
TransformToMstream(renamed_pred_true_texts)


sublist=SampleData("Tweets-T", 1.00)
pred_true_texts=ExtractPredTrueText(sublist)
lastTrueLabel, lastId, renamed_pred_true_texts=RenameTrueLabel(pred_true_texts,lastTrueLabel, lastId)
TransformToMstream(renamed_pred_true_texts)


'''sublist=SampleData("Stackoverflow-mstream", 1.00)
pred_true_texts=ExtractPredTrueText(sublist)
lastTrueLabel, lastId, renamed_pred_true_texts=RenameTrueLabel(pred_true_texts,lastTrueLabel, lastId)
TransformToMstream(renamed_pred_true_texts)


sublist=SampleData("Biomedical-mstream", 1.00)
pred_true_texts=ExtractPredTrueText(sublist)
lastTrueLabel, lastId, renamed_pred_true_texts=RenameTrueLabel(pred_true_texts,lastTrueLabel, lastId)
TransformToMstream(renamed_pred_true_texts)


sublist=SampleData("news_41_200853-mstream", 1.00)
pred_true_texts=ExtractPredTrueText(sublist)
lastTrueLabel, lastId, renamed_pred_true_texts=RenameTrueLabel(pred_true_texts,lastTrueLabel, lastId)
TransformToMstream(renamed_pred_true_texts)'''











'''file1=open("News-T","r")
linesNewsT = file1.readlines()
file1.close()
length=len(linesNewsT)
total=int(length*0.25)
randIndecies = [random.randint(0,length) for i in range(total)]
sublist2 = [linesNewsT[index] for index in randIndecies]'''

'''file1=open("Tweets","r")
linesTweets = file1.readlines()
file1.close()
length=len(linesTweets)-1
total=int(length*0.5)
randIndecies = [random.randint(0,length) for i in range(total)]
sublist3 = [linesTweets[index] for index in randIndecies]
pred_true_texts=[]
for item in sublist3:
  i=i+1
  idStr = str(i).zfill(6)
  n = eval(item)
  pred=n['Id']  
  true=n['clusterNo']
  text=n['textCleaned']
  pred_true_texts.append([pred, true, text])  
dic_group = groupItemsBySingleKeyIndex(pred_true_texts,1)
groups=len(dic_group)
print(groups)'''



'''file1=open("Tweets-T","r")
linesTweetsT = file1.readlines()
file1.close()
length=len(linesTweetsT)
total=int(length*0.25)
randIndecies = [random.randint(0,length) for i in range(total)]
sublist4 = [linesTweetsT[index] for index in randIndecies]'''

'''file1=open("Stackoverflow-mstream","r")
linesStackoverflowT = file1.readlines()
file1.close()
length=len(linesStackoverflowT)-1
total=int(length*0.5)
randIndecies = [random.randint(0,length) for i in range(total)]
sublist5 = [linesStackoverflowT[index] for index in randIndecies]
pred_true_texts=[]
for item in sublist5:
  i=i+1
  idStr = str(i).zfill(6)
  n = eval(item)
  pred=n['Id']  
  true=n['clusterNo']
  text=n['textCleaned']
  pred_true_texts.append([pred, true, text])  
dic_group = groupItemsBySingleKeyIndex(pred_true_texts,1)
groups=len(dic_group)
print(groups)'''

'''jointList=sublist1+sublist3+sublist5
i=0
for item in jointList:
  i=i+1
  nstr = str(i).zfill(6)
  n = eval(item)
  n['Id']=nstr   
  #print(n)'''





