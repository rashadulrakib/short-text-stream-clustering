import re

file1=open("stackoverflow_javascript_true_id_title_tags","r")
lines = file1.readlines()
file1.close()

file1=open("stackoverflow_javascript","w")

count=1 
for line in lines:
  line = line.strip()
  if len(line)==0:
    continue
  arr = re.split("\t", line)
  #print(arr)
  if len(arr)!=4:
    continue
  true=arr[0].strip()
  text=arr[2].strip()
  text=text.replace('"', '').replace("\\",'')
  
  
  if len(true)==0 or len(text)==0:
    continue
  #{"Id": "000001", "clusterNo": 65, "textCleaned": "centrepoint winter white gala london"}
  test_str='{"Id": "'+str(count).zfill(6)+'", "clusterNo": '+str(true)+', "textCleaned":"'+text+'"}\n'
  s=eval(test_str)  
  file1.write('{"Id": "'+str(count).zfill(6)+'", "clusterNo": '+str(true)+', "textCleaned":"'+text+'"}\n')
  count=count+1  
  
file1.close()  
    