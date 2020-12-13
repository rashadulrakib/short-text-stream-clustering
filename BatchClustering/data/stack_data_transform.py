import pandas as pd
import sys
from nltk.stem import PorterStemmer
sys.path.append('..')

#https://xxx-cook-book.gitbooks.io/python-cook-book/Import/import-from-parent-folder.html

from txt_process_util import getScikitLearn_StopWords
from txt_process_util import processTxtStopWordsStemming

stop_words=getScikitLearn_StopWords()
ps = PorterStemmer()

#files=[r'C:\Users\mona\Dropbox\www-2021\super_user_QueryResults_10_110.csv',
#r'C:\Users\mona\Dropbox\www-2021\super_user_QueryResults_111_1000.csv',
#r'C:\Users\mona\Dropbox\www-2021\super_user_QueryResults_1001_.csv']

files=[r'C:\Users\mona\Dropbox\www-2021\super_user_QueryResults_111_1000.csv']

list_df=[]

for file in files:
  df=pd.read_csv(file)
  list_df.append(df)
  
df_all=pd.concat(list_df)  

file1=open("super_user","w")
count=1
dic_label={}
dic_txt={}
tags=[]

for i in df_all.index:
  title=str(df_all.at[i, 'Title']).strip().lower()
  #tag=str(df_all.at[i, 'Tags']).strip().lower().replace('\n','').replace("['",'').replace("']",'').replace("' '",'').replace("<",'').replace(">",'').replace("><",'-')
  tag=str(df_all.at[i, 'Tags']).strip().lower()
  tags.append(tag)
  
  st_txt=processTxtStopWordsStemming(title, stop_words, ps).strip()
  if len(st_txt)==0 or len(tag)==0 or st_txt in dic_txt:
    continue
	
  dic_txt[st_txt]=st_txt	
	
  if tag not in dic_label:
    label_count=len(dic_label)+1
    dic_label[tag]=label_count
  	
  
  print(tag, st_txt)
  #file1.write('{"Id": "'+str(count).zfill(6)+'", "clusterNo": "'+str(dic_label[tag])+'", "textCleaned":"'+st_txt+'"}\n')
  file1.write('{"Id": "'+str(count).zfill(6)+'", "clusterNo": "'+tag+'", "textCleaned":"'+st_txt+'"}\n')
  count+=1

file1.close()

print(dic_label)

print(len(df_all.Tags.unique())) 
print(len(dic_label))
print(len(set(tags)))