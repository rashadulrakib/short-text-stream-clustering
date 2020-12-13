from datetime import time
from lxml import etree
import time
import pickle
import sys
from statistics import mean
from statistics import stdev 
from rake_nltk import Rake
import re


sys.path.append('..')

#sys.setdefaultencoding('utf8')

import os

#https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede

#https://archive.org/details/stackexchange

#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



from xmlreader.DataConverter import DataConverter

import networkx as nx

#https://necromuralist.github.io/data_science/posts/connected-components/

#https://xxx-cook-book.gitbooks.io/python-cook-book/Import/import-from-parent-folder.html

path = '/users/grad/rakib/stackoverflow/stackoverflow/stackoverflow.PostLinks.xml'

def safe_str(obj):
    try: 
      st=str(obj)
      st=st.replace('\t', ' ')	  
      return st
    except UnicodeEncodeError:
        return ''
    return ""
	
def preprocess(str):
 str=re.sub(r'[^a-zA-Z0-9 ]', ' ', str)
 str=re.sub(r'\b[a-z]\b|\b\d+\b', '', str)
 str=re.sub(r'lt', ' ', str)
 str=re.sub(r'gt', ' ', str)
 str=re.sub(r'\s+',' ',str).strip()
 return str	
	

#load the user data
start_time = time.time()

#done, r, cplaus, php, csharp python android javascript

f_lang=None
#f_java = open('stackoverflow_java_id_title_tags', 'r')
#f_android = open('stackoverflow_android_id_title_tags', 'r')
#f_javascript = open('stackoverflow_javascript_id_title_tags', 'r')
#f_python = open('stackoverflow_python_id_title_tags', 'r')
#f_php = open('stackoverflow_php_id_title_tags', 'r')
#f_cSharp = open('stackoverflow_csharp_id_title_tags', 'r')
#f_lang = open('stackoverflow_cplus_id_title_tags', 'r')
#f_jquery = open('stackoverflow_jquery_id_title_tags', 'r')
f_lang = open('stackoverflow_java_id_title_tags_body_createtime', 'r')
#f_mysql = open('stackoverflow_mysql_id_title_tags', 'r')

post_ids=[]
for line in f_lang:
  line=line.strip()
  arr=line.split('\t')
  #print(len(arr))
  if len(arr)!=5:
    continue
  
  post_ids.append(int(arr[0]))

post_ids=set(post_ids)
   
print('len(post_ids)', len(post_ids))

#f_java.close()
#f_android.close()
#f_javascript.close()
#f_python.close()
#f_php.close()
#f_cSharp.close()
#f_cPlus.close()
#f_jquery.close()
#f_r.close()
#f_mysql.close()

f_lang.close()


#del f_java
#del f_android
#del f_javascript
#del f_python
#del f_php
#del f_cSharp
#del f_cPlus
#del f_jquery
#del f_r
#del f_mysql
del f_lang

count = 0

duplicate_pairs=0
list_edges=[]

print('start', path)

for event, elem in etree.iterparse(path, events=("start", "end", "start-ns", "end-ns")):
    if elem.tag == "row" and event == "start":
        count = count + 1
        if (count % 100000 == 0):
            print("Progress of reading users: " + str(count))
            #break			
        postLink = DataConverter.readPostLink(elem)
        linkTypeId=postLink.get_linkTypeId()	
        if (linkTypeId!=1):
          continue		
        #post.print_post()	
        post_id=postLink.get_postId()
        related_post_id=postLink.get_relatedPostId() 
		
        if post_id not in post_ids or related_post_id not in post_ids:
          continue		
		
        if (linkTypeId==1) and post_id!=related_post_id:
          duplicate_pairs+=1		  
          list_edges.append((post_id, related_post_id)) 
          #print('list_edges.append((post_id, related_post_id))', post_id, related_post_id)

        		  
  
    elem.clear()
    del elem
	
print("--- %s seconds ---" % (time.time() - start_time))
print('duplicate_pairs', duplicate_pairs)


G=nx.DiGraph(list_edges)
cycles=nx.simple_cycles(G)

list_compSize=[]
total_nodes=[]

dic_txtId__clusterId={}

components=0

for cycle in cycles:
  print(cycle)
  len_comp=len(cycle) 
  list_compSize.append(len_comp)
  total_nodes.extend(cycle)
  components+=1 #clusterid
  for post_id in cycle:
    dic_txtId__clusterId[post_id]= components
  
  
total_nodes=set(total_nodes)   

print("--- %s seconds ---" % (time.time() - start_time))  
print('components', len(list(nx.simple_cycles(G))), 'max', max(list_compSize), 'min', min(list_compSize), 'avg', mean(list_compSize), 'stdev', stdev(list_compSize), 'total_nodes', len(total_nodes))

G.clear()
del G

#list_compSize.clear()
del list_compSize

total_nodes.clear()
del total_nodes

f_java_w = open('stackoverflow_java_true_id_title_tags_body_createtime', 'w')
f_java_r = open('stackoverflow_java_id_title_tags_body_createtime', 'r')

for line in f_java_r:
  line=line.strip()
  arr=line.split('\t')
  
  if len(arr)!=5:
    continue
  id=int(arr[0])
  true_label='-1'
  if id in dic_txtId__clusterId:
    true_label= str(dic_txtId__clusterId[id])
	
  body = arr[3].strip()
  body=preprocess(body)  
  r= Rake(min_length=2, max_length=4)
  r.extract_keywords_from_text(body)
  phrases=r.get_ranked_phrases()
  del r
  
  max_phrases=min(10, len(phrases))
  target_phs=phrases[0:max_phrases]
  target_body=' '.join(target_phs)
  
  #if true_label!='-1':
  #  print(true_label+"	"+arr[0]+"	"+arr[1]+"	"+arr[2]+"	"+target_body+"	"+arr[4])  
  	
  f_java_w.write(true_label+"	"+arr[0]+"	"+arr[1]+"	"+arr[2]+"	"+target_body+"	"+arr[4]+"\n")

f_java_r.close()
f_java_w.close()

del f_java_r
del f_java_w

print('finish writing')