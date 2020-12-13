from general_util import readStackOverflowDataSetRaw
import pandas as pd

inputfile = 'D:/githubprojects/PyMigrationRecommendation/src/notebooks/stackoverflow_cplus_true_id_title_tags'

list_true_id_title_tags=readStackOverflowDataSetRaw(inputfile)

print(len(list_true_id_title_tags))
#data_transposed = zip(list_true_id_title_tags)
df = pd.DataFrame(list_true_id_title_tags, columns=["truelabel", "postid", 'title', 'tag', ])

df__1=df[df['truelabel']=='-1']
print(df__1)
df_not_1 = df[df['truelabel']!='-1']

df_gr_not_1=df_not_1.groupby("truelabel")#.filter(lambda x: len(x) > 1)
#df.groupby("A").filter(lambda x: len(x) > 1)
#df.groupby("A").filter(lambda x: len(x) > 1)

print(df_gr_not_1)

list_test=[]
list_train_not_1=[]

for name, group in df_gr_not_1:
  g_size=len(group)
  if g_size<=1:
    continue
  print(name)
  print(len(group))
  #data_dict = group.to_dict() 
  #print(data_dict)
  list=group.values.tolist()
  print(list[0]) #single entry []
  print(list[1:len(list)]) ##list [[]]
  list_test.append(list[0])
  list_train_not_1.extend(list[1:len(list)])
  
  
  print("\n")


list_train__1=df__1.values.tolist()  
print(len(list_test))  
print(len(list_train_not_1))  
print(len(list_train__1))

list_train= list_train_not_1+list_train__1
print(len(list_train))

df_train = pd.DataFrame(list_train)
df_test = pd.DataFrame(list_test)

print(df_train)
print(df_test)

print(df_train)
df_train.to_csv('D:/githubprojects/PyMigrationRecommendation/src/notebooks/train_stackoverflow_cplus_true_id_title_tags', sep='\t', index=False, header=False)
df_test.to_csv('D:/githubprojects/PyMigrationRecommendation/src/notebooks/test_stackoverflow_cplus_true_id_title_tags', sep='\t', index=False, header=False)




