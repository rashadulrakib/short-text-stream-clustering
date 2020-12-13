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
#from networkx import (DiGraph,Graph)
from networkx import DiGraph

#https://necromuralist.github.io/data_science/posts/connected-components/

#https://xxx-cook-book.gitbooks.io/python-cook-book/Import/import-from-parent-folder.html

path = '/users/grad/rakib/stackoverflow/stackoverflow/stackoverflow.Posts.xml'

def safe_str(obj):
    try: 
      st=str(obj)
      st=st.replace('\t', ' ')	
      st=st.replace("\n", " ")	  
      return st
    except UnicodeEncodeError:
        return ''
    return ""

#load the user data
start_time = time.time()
count = 0
f_java = open('stackoverflow_java_id_title_tags_body_createtime', 'w')
f_android = open('stackoverflow_android_id_title_tags_body_createtime', 'w')
f_javascript = open('stackoverflow_javascript_id_title_tags_body_createtime', 'w')
f_python = open('stackoverflow_python_id_title_tags_body_createtime', 'w')
f_php = open('stackoverflow_php_id_title_tags_body_createtime', 'w')
f_cSharp = open('stackoverflow_csharp_id_title_tags_body_createtime', 'w')
f_cPlus = open('stackoverflow_cplus_id_title_tags_body_createtime', 'w')
f_jquery = open('stackoverflow_jquery_id_title_tags_body_createtime', 'w')
f_r = open('stackoverflow_r_id_title_tags_body_createtime', 'w')
f_mysql = open('stackoverflow_mysql_id_title_tags_body_createtime', 'w')

for event, elem in etree.iterparse(path, events=("start", "end", "start-ns", "end-ns")):
    if elem.tag == "row" and event == "start":
        count = count + 1
        if (count % 1000000 == 0):
            print("Progress of reading users: " + str(count))
        post = DataConverter.readPost(elem)
        #post.print_post()		
        PostTypeId=post.get_postTypeId()
        if PostTypeId!=1:
          continue 
        title=post.get_title()
        tags=post.get_tags()
        id=str(post.get_id())
        body=post.get_body()
        creationDate=str(post.get_creationDate())		
		
        title=safe_str(title)
        tags=safe_str(tags)
        body=safe_str(body)		
       	
		
        if title=='' or tags=='' or id=='' or body=='' or creationDate=='':
          continue

        str_data=id+"	"+title+"	"+tags+"	"+body+"	"+creationDate
        		
        		
        if '<java>' in tags:
          f_java.write(str_data+"\n") 
        elif '<android>' in tags:
          f_android.write(str_data+"\n") 
        elif '<javascript>' in tags:
          f_javascript.write(str_data+"\n") 
        elif '<python>' in tags:
          f_python.write(str_data+"\n") 
        elif '<php>' in tags:
          f_php.write(str_data+"\n") 
        elif '<c#>' in tags:
          f_cSharp.write(str_data+"\n") 
        elif '<c++>' in tags:
          f_cPlus.write(str_data+"\n") 
        elif '<jquery>' in tags:
          f_jquery.write(str_data+"\n") 
        elif '<r>' in tags:
          f_r.write(str_data+"\n") 
        elif '<mysql>' in tags:
          f_mysql.write(str_data+"\n") 
        	  
		  
		  
    elem.clear()
    del elem
print("--- %s seconds ---" % (time.time() - start_time))
f_java.close()
f_android.close()
f_javascript.close()
f_python.close()
f_php.close()
f_cSharp.close()
f_cPlus.close()
f_jquery.close()
f_r.close()
f_mysql.close()

