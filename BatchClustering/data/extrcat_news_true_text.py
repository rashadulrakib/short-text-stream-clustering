import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english')) 

file1=open('News_Category_Dataset_v2.json',"r")
linesNews = file1.readlines()
file1.close()

lineNo=0
trues=[]
texts=[]

for line in linesNews:
  lineNo+=1
  try:
    n = eval(line)
    
    true=n['category'].strip()
    text=n['headline'].strip()
    text=re.sub(r'[^\w\d\s]',' ',text)
    text=re.sub(r'\s+',' ',text).strip()	
    word_tokens = word_tokenize(text)  
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 	
    text=' '.join(filtered_sentence).strip()	
    #print(text)	
    if len(true)==0 or len(text)==0:
      continue	
    #print(true.lower()+"\t"+text.lower())
    trues.append(true.lower())
    texts.append(text.lower())	
  except:
    #print('Linux function was not executed')
    continue

#print(len(set(trues)))	

trueIndex=list(set(trues))

for i in range(len(trues)):
  try:
    true=trues[i]
    text=texts[i]
    tIndex=trueIndex.index(true)+1
    print(str(tIndex)+"\t"+text)
  except:
    continue
  

  