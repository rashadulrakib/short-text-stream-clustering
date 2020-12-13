import re

def concatWordsSort(words):
  words.sort()
  combinedWord=' '.join(words)
  return combinedWord

file1=open("NT-mstream-long1","r")
lines = file1.readlines()
file1.close()

file1=open("BMM-NT-mstream-long1","w")

count=1 
for line in lines:
  line = line.strip()
  if len(line)==0:
    continue
  n = eval(line)
  Id=str(n['Id']).strip()  
  true=str(n['clusterNo']).strip()
  text=str(n['textCleaned']).strip()
  if len(true)==0 or len(text)==0:
    continue
  #{"clusterNo": 2, "textCleaned": "spd expands labor demand talk german coalition", "bitermText": "spd,expands spd,labor spd,demand spd,talk spd,german spd,coalition expands,labor expands,demand expands,talk expands,german expands,coalition labor,demand labor,talk labor,german labor,coalition demand,talk demand,german demand,coalition talk,german talk,coalition german,coalition"}
  words=text.strip().split()
  bi_grams=[]				
  for j in range(len(words)):
    for k in range(j+1,len(words)): 				   
      bi_grams.append(concatWordsSort([words[j], words[k]]))
  for j in range(len(words)-1):
    bi_grams.append(concatWordsSort([words[j], words[j+1]]))  				
  bttext=','.join(bi_grams)  
  
  #{"Id": "000001", "clusterNo": 65, "textCleaned": "centrepoint winter white gala london"}
    
  file1.write('{"clusterNo": '+str(true)+', "textCleaned": "'+text+'", "bitermText": "'+bttext+'"}\n')
  count=count+1  
  
file1.close()  
    