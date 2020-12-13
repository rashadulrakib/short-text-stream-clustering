import re
import statistics 

file1=open("D:/githubprojects/stackoverflow.com-PostLinks/stackoverflow_true_title_tags","r")
lines = file1.readlines()
file1.close()

file1=open("stackoverflow_large","w")

dic_c__items={}

count=1 
for line in lines:
  line = line.strip()
  if len(line)==0:
    continue
  arr = re.split("\t", line)
  #print(arr)
  if len(arr)!=3:
    continue
  true=arr[0].strip()
  text=arr[1].strip()
  text=text.replace('"', '').replace("\\",'')
  
  
  if len(true)==0 or len(text)==0:
    continue
	
  dic_c__items.setdefault(true, []).append(text)	
  #{"Id": "000001", "clusterNo": 65, "textCleaned": "centrepoint winter white gala london"}
  test_str='{"Id": "'+str(count).zfill(6)+'", "clusterNo": '+str(true)+', "textCleaned":"'+text+'"}\n'
  s=eval(test_str)  
  file1.write('{"Id": "'+str(count).zfill(6)+'", "clusterNo": '+str(true)+', "textCleaned":"'+text+'"}\n')
  count=count+1  
  
file1.close() 

li=[len(dic_c__items[x]) for x in dic_c__items if isinstance(dic_c__items[x], list)]
print('min', min(li) , 'max', max(li) , 'median', statistics.median(li)   , 'avg', statistics.mean(li) , 'std',statistics.stdev(li) , 'sum of li', sum(li)) 

mean=statistics.mean(li)
std=statistics.stdev(li)

file1=open("stackoverflow_large","w")


count=1 
for line in lines:
  line = line.strip()
  if len(line)==0:
    continue
  arr = re.split("\t", line)
  #print(arr)
  if len(arr)!=3:
    continue
  true=arr[0].strip()
  text=arr[1].strip()
  text=text.replace('"', '').replace("\\",'')
  
  
  if len(true)==0 or len(text)==0:
    continue

  c_items=len(dic_c__items[true]) 
  if c_items<=1 or c_items<abs(mean-std) or c_items>abs(mean+std):
    continue  
	
  #{"Id": "000001", "clusterNo": 65, "textCleaned": "centrepoint winter white gala london"}
  test_str='{"Id": "'+str(count).zfill(6)+'", "clusterNo": '+str(true)+', "textCleaned":"'+text+'"}\n'
  s=eval(test_str)  
  file1.write('{"Id": "'+str(count).zfill(6)+'", "clusterNo": '+str(true)+', "textCleaned":"'+text+'"}\n')
  count=count+1  
  
file1.close()

    