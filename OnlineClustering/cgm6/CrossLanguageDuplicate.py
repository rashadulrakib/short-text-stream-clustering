from datetime import time
from lxml import etree
import time
import pickle
import sys
from statistics import mean
from statistics import stdev 

sys.path.append('..')

#sys.setdefaultencoding('utf8')

import os

#https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede

#https://archive.org/details/stackexchange

#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



from xmlreader.DataConverter import DataConverter

import networkx
from networkx import (DiGraph,Graph)
#from networkx import DiGraph

#https://necromuralist.github.io/data_science/posts/connected-components/

#https://xxx-cook-book.gitbooks.io/python-cook-book/Import/import-from-parent-folder.html

path = 'D:/githubprojects/stackoverflow.com-PostLinks/PostLinks.xml'

def safe_str(obj):
    try: 
      st=str(obj)
      st=st.replace('\t', ' ')	  
      return st
    except UnicodeEncodeError:
        return ''
    return ""

def getPostId_tag(filename):
  dic_PostId__tag={}
  f_lang = open(filename, 'r')
  for line in f_lang:
    line=line.strip()
    arr=line.split('\t')
    if len(arr)!=4:
      continue
    dic_PostId__tag[int(arr[1])]=arr[3]	  
	  
	  
  f_lang.close()
  del f_lang 
  return dic_PostId__tag  
  
def targetTags_found_in_PostId(comp_postId, dic_PostId__tag, target_tags): 
  list_cross_dups=[]
  if comp_postId in dic_PostId__tag:
    lang_tag=dic_PostId__tag[comp_postId].lower() #<java><python>
    for target_tag in target_tags:
      if target_tag in lang_tag:
        list_cross_dups.append(comp_postId) 
        #print(comp_postId)		  
  return list_cross_dups  
   
   
def foundPostIds(post_id, related_post_id, list_dic_PostId__tag):
  for dic_PostId__tag in list_dic_PostId__tag:
    if post_id in dic_PostId__tag or related_post_id in dic_PostId__tag:  
      return True  
  
  return False

def writeToFile(fileName, list_cross_duplicates_postId):
  f=open(fileName, 'w')
  for postId in list_cross_duplicates_postId:
    f.write(str(postId)+"\n")
  f.close()
  del f  

#load the user data
start_time = time.time()

dic_PostId__java_tag=getPostId_tag('stackoverflow_java_true_id_title_tags')
dic_PostId__android_tag=getPostId_tag('stackoverflow_android_true_id_title_tags')
dic_PostId__javascript_tag=getPostId_tag('stackoverflow_javascript_true_id_title_tags')
dic_PostId__python_tag=getPostId_tag('stackoverflow_python_true_id_title_tags')
dic_PostId__php_tag=getPostId_tag('stackoverflow_php_true_id_title_tags')
dic_PostId__csharp_tag=getPostId_tag('stackoverflow_csharp_true_id_title_tags')
dic_PostId__cplus_tag=getPostId_tag('stackoverflow_cplus_true_id_title_tags')
dic_PostId__r_tag=getPostId_tag('stackoverflow_r_true_id_title_tags')

list_dic_PostId__tag=[]
list_dic_PostId__tag.append(dic_PostId__java_tag)
list_dic_PostId__tag.append(dic_PostId__android_tag)
list_dic_PostId__tag.append(dic_PostId__javascript_tag)
list_dic_PostId__tag.append(dic_PostId__python_tag)
list_dic_PostId__tag.append(dic_PostId__php_tag)
list_dic_PostId__tag.append(dic_PostId__csharp_tag)
list_dic_PostId__tag.append(dic_PostId__cplus_tag)
list_dic_PostId__tag.append(dic_PostId__r_tag)





count = 0

duplicate_pairs=0
list_edges=[]

print('start', path)

for event, elem in etree.iterparse(path, events=("start", "end", "start-ns", "end-ns")):
    if elem.tag == "row" and event == "start":
        count = count + 1
        if (count % 1000000 == 0):
            print("Progress of reading users: " + str(count))
            #break			
        postLink = DataConverter.readPostLink(elem)
        linkTypeId=postLink.get_linkTypeId()		
        #post.print_post()	
        if linkTypeId!=1:
          continue		
        post_id=postLink.get_postId()
        related_post_id=postLink.get_relatedPostId() 
		
        
        if foundPostIds(post_id, related_post_id, list_dic_PostId__tag)==False:
          continue  		
		
        #if (linkTypeId==1 or linkTypeId==3) and post_id!=related_post_id:
        if (linkTypeId==1) and post_id!=related_post_id:		
          duplicate_pairs+=1		  
          list_edges.append((post_id, related_post_id)) 
         

        		  
  
    elem.clear()
    del elem
	
print("--- %s seconds ---" % (time.time() - start_time))
print('duplicate_pairs', duplicate_pairs)
#undirected = DiGraph()
undirected = Graph()
undirected.add_edges_from(list_edges)	
	
components=0
list_compSize=[]
#for component in networkx.connected_components(undirected):
total_nodes=[]


#target_tags=set(['<r>', '<c++>', 'java', 'android', 'php', 'c#', 'javascript', 'python'])
list_cross_duplicates_java=[]
list_cross_duplicates_android=[]
list_cross_duplicates_javascript=[]
list_cross_duplicates_python=[]
list_cross_duplicates_php=[]
list_cross_duplicates_csharp=[]
list_cross_duplicates_cplus=[]
list_cross_duplicates_r=[]

#for component in networkx.weakly_connected_components(undirected):
for component in networkx.connected_components(undirected):
  component=list(component) 
  len_comp=len(component) 
  list_compSize.append(len_comp)
  total_nodes.extend(component)
  components+=1 #clusterid
  print('len_comp', len_comp, 'clusterid', components)
  for comp_postId in component:
    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__java_tag, ['<r>', 'android', 'php', 'c#', 'javascript', 'python'])
    list_cross_duplicates_java.extend(list_cross_dups)	
     	
    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__android_tag, ['<c++>', 'java',  'php', 'c#', 'javascript', 'python'])
    list_cross_duplicates_android.extend(list_cross_dups)		
	
    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__javascript_tag, ['<r>', 'java', 'android', 'php', 'c#', 'python'])
    list_cross_duplicates_javascript.extend(list_cross_dups)		
   	
    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__python_tag, ['<c++>', 'java', 'android', 'php', 'c#', 'javascript'])
    list_cross_duplicates_python.extend(list_cross_dups)		

    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__php_tag, ['<r>', 'java', 'android', 'c#', 'javascript', 'python'])
    list_cross_duplicates_php.extend(list_cross_dups)		
 
    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__csharp_tag, ['<c++>', 'java', 'android', 'php', 'javascript', 'python'])
    list_cross_duplicates_csharp.extend(list_cross_dups)		

    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__cplus_tag, ['<r>', 'java', 'android', 'php', 'c#', 'javascript', 'python'])
    list_cross_duplicates_cplus.extend(list_cross_dups)		
   	
    
    list_cross_dups=targetTags_found_in_PostId(comp_postId, dic_PostId__r_tag, ['<c++>', 'java', 'android', 'php', 'c#', 'javascript', 'python'])
    list_cross_duplicates_r.extend(list_cross_dups)	
 	
    
writeToFile('list_cross_duplicates_java', set(list_cross_duplicates_java))
writeToFile('list_cross_duplicates_android', set(list_cross_duplicates_android))
writeToFile('list_cross_duplicates_javascript', set(list_cross_duplicates_javascript))
writeToFile('list_cross_duplicates_python', set(list_cross_duplicates_python))
writeToFile('list_cross_duplicates_php', set(list_cross_duplicates_php))
writeToFile('list_cross_duplicates_csharp', set(list_cross_duplicates_csharp))
writeToFile('list_cross_duplicates_cplus', set(list_cross_duplicates_cplus))
writeToFile('list_cross_duplicates_r', set(list_cross_duplicates_r))


     
    	
        	
    	
  
  
total_nodes=set(total_nodes)  
print("--- %s seconds ---" % (time.time() - start_time))  
print('components', components, 'max', max(list_compSize), 'min', min(list_compSize), 'avg', mean(list_compSize), 'stdev', stdev(list_compSize), 'total_nodes', len(total_nodes))

undirected.clear()
del undirected

#list_compSize.clear()
del list_compSize

total_nodes.clear()
del total_nodes
	
print("--- %s seconds ---" % (time.time() - start_time))

#total_posts=len(dic_txtId__clusterId)
#print('total_posts', total_posts)
#posts_found=0




