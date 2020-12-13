#when merging single text to a cluster.
#caculate the closest similarity of each single text to the clusters.
#if the similarity if the single text is greater than mena and standard deviation of the similarities, then add it to a closest clustyer, other wise leave it in a single text cluster

from itr_clustering_multipass_external_arg_classification import enhanceClustersByIterativeClassification
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from groupTxt_ByClass import groupTxtByClass
from cluster_file_tfidf import clusterByTfIdfFeature
from cluster_file_word_embedding import clusterByWordEmbeddingFeature
from sklearn import metrics
from print_cluster_evaluation import printClusterEvaluation_list
import re
import numpy as np
from read_pred_true_text import ReadPredTrueText
from cluster_hdbscan import ClusterByHDbScan
from word_vec_extractor import extractAllWordVecs
from general_util import Print_list_pred_true_text
from general_util import print_by_group
from general_util import change_pred_label
from outliers_detection_lexical import DetectNonOutliersLexical
from outliers_detection_connectedComp import removeOutlierConnectedComponentLexical
from txt_process_util import RemoveHighClusterEntropyWords

def DetectNonOutliersByThreshold(dic_tupple_class, avgItemsInCluster_in_a_batch):
  non_outlier_pred_true_txts_in_all_clusters=[]
  outlier_pred_true_txts_in_all_clusters=[]
  for label, pred_true_txts in dic_tupple_class.items():
    itemsInCluster=len(pred_true_txts)
    if itemsInCluster>avgItemsInCluster_in_a_batch:
      #print("cluster label="+str(label)+", "+str(itemsInCluster))
      textsArr=[]
      for pred_true_txt in pred_true_txts:
        textsArr.append(pred_true_txt[2])
      vectorizer = TfidfVectorizer( max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
      x_train = vectorizer.fit_transform(textsArr)
      contratio = 0.3
      isf = IsolationForest(n_estimators=100, max_samples='auto', contamination=contratio, max_features=1.0, bootstrap=True, verbose=0, random_state=0, behaviour='new')
      #isf=IsolationForest(n_estimators=100, max_samples='auto', contamination=contratio, max_features=1.0, bootstrap=True, verbose=0, random_state=0)
      outlierPreds = isf.fit(x_train).predict(x_train)
      non_outlier_pred_true_txts_in_a_cluster=[]
      for i in range(len(outlierPreds)):
        outlierPred=outlierPreds[i]
        if outlierPred !=-1:
          non_outlier_pred_true_txts_in_a_cluster.append(pred_true_txts[i])	
          non_outlier_pred_true_txts_in_all_clusters.append(pred_true_txts[i])
        else:
         outlier_pred_true_txts_in_all_clusters.append(pred_true_txts[i])		
    else:
      non_outlier_pred_true_txts_in_all_clusters.extend(pred_true_txts)
  dic_tupple_class_filteres=groupTxtByClass(non_outlier_pred_true_txts_in_all_clusters, False)
  printClusterEvaluation_list(non_outlier_pred_true_txts_in_all_clusters)
  print ("true clusters="+str(len(groupTxtByClass(non_outlier_pred_true_txts_in_all_clusters, True))))  
  #ComputePurity(dic_tupple_class_filteres) 	
  
  return [non_outlier_pred_true_txts_in_all_clusters, outlier_pred_true_txts_in_all_clusters]  
        		 
def DetectNonOutliers(listtuple_pred_true_text):
 printClusterEvaluation_list(listtuple_pred_true_text)
 dic_tupple_class=groupTxtByClass(listtuple_pred_true_text, False)
 print ("true clusters="+str(len(groupTxtByClass(listtuple_pred_true_text, True))))
 #ComputePurity(dic_tupple_class)
 
 totalItems=0
 itemsInClusterList=[]
 for label, pred_true_txts in dic_tupple_class.items():
   itemsInCluster=len(pred_true_txts)
   #print("itemsInCluster="+str(itemsInCluster))   
   totalItems=totalItems+itemsInCluster
   itemsInClusterList.append(itemsInCluster)
 
 totalClusters=len(dic_tupple_class)
 avgItemsInCluster_in_a_batch= float(totalItems)/totalClusters
 std=np.std(itemsInClusterList)
 print("totalItems="+str(totalItems)+",avgItemsInCluster_in_a_batch="+str(avgItemsInCluster_in_a_batch)+",std="+str(std)) 
 non_outlier_pred_true_txts_in_all_clusters, outlier_pred_true_txts_in_all_clusters = DetectNonOutliersByThreshold(dic_tupple_class, avgItemsInCluster_in_a_batch)
 print("total #outliers="+str(len(outlier_pred_true_txts_in_all_clusters)))
 #print("#non_outlier_pred_true_txts_in_all_clusters#") 
 #print(non_outlier_pred_true_txts_in_all_clusters)
 #print("#outlier_pred_true_txts_in_all_clusters#")
 #print(outlier_pred_true_txts_in_all_clusters)
 #print("--Batch End--")
 return [non_outlier_pred_true_txts_in_all_clusters, outlier_pred_true_txts_in_all_clusters,avgItemsInCluster_in_a_batch] 
      
 
#dataFilePredTrueTxt="/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText7" #batchId_PredTrueText10_to_16.txt"

#predTrueTextfiles=["/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText1"]

predTrueTextfiles=["/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText1",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText2",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText3",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText4",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText5",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText6",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText7",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText8",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText9",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText10",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText11",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText12",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText13",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText14",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText15",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText16"
]


'''"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText1_to_9.txt",
"/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText10_to_16.txt",
"/home/owner/PhD/MStream-master/MStream/result/NewsPredTueTextMStream_WordArr.txt",
"/home/owner/PhD/MStream-master/MStream/result/NewsPredTueTextMStreamSemantic_WordArr.txt"'''


#predTrueTextfiles=["/home/owner/PhD/MStream-master/MStream/result/batchId_PredTrueText1"]

merged_new_label_pred_true_txts_all_batch=[]
non_outlier_pred_true_txts_all_batch=[]
mstream_pred_true_txts_all_batch=[]

gloveFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/glove.42B.300d/glove.42B.300d.txt"
wordVectorsDic = extractAllWordVecs(gloveFile, 300)


for predTrueTextfile in predTrueTextfiles:
   listtuple_pred_true_text = ReadPredTrueText(predTrueTextfile)
      
   #remove outliers from each cluster by connected components
   #assign those outliers to the clusters based on common words
   #find out the entropy of each word using cluster distribution
   #remove high entropy words (needs logic) from each text
   #find the embedding of each text
   #cluster the texts using hac + sd method
   #cluster text by tf-idf feature    
   
   outlier_pred_true_texts, non_outlier_pred_true_txts, avgItemsInCluster=removeOutlierConnectedComponentLexical(listtuple_pred_true_text)
   
   #change pred labels
   newOutlier_pred_true_txts=change_pred_label(outlier_pred_true_texts, 1000)
   #end change pred labels     
   non_outlier_pred_true_txts.extend(newOutlier_pred_true_txts)
   
   #print("print_by_group(outlier_pred_true_texts)")   
   #print_by_group(outlier_pred_true_texts)
   print("print_by_group(non_outlier_pred_true_txts)")
   print_by_group(non_outlier_pred_true_txts)
    	  
   print("listtuple_pred_true_text")   
   printClusterEvaluation_list(listtuple_pred_true_text)   
   #print("outlier_pred_true_texts")
   #printClusterEvaluation_list(outlier_pred_true_texts)   
   print("non_outlier_pred_true_txts")
   printClusterEvaluation_list(non_outlier_pred_true_txts)
   #cleaned_pred_true_txts=RemoveHighClusterEntropyWords(non_outlier_pred_true_txts)
   cleaned_pred_true_txts=non_outlier_pred_true_txts     
   
   clusterByTfIdfFeature(cleaned_pred_true_txts)    
   clusterByWordEmbeddingFeature(cleaned_pred_true_txts, wordVectorsDic)
         
   #non_outlier_pred_true_txts, outlier_pred_true_txts,avgItemsInCluster = DetectNonOutliersLexical(listtuple_pred_true_text)
   
   #Print_list_pred_true_text(listtuple_pred_true_text) 
   #print("#real Batch end#")
   #non_outlier_pred_true_txts_in_all_clusters, outlier_pred_true_txts_in_all_clusters,avgItemsInCluster_in_a_batch =DetectNonOutliers(listtuple_pred_true_text)

   #clusterByTfIdfFeature(listtuple_pred_true_text)    
   #clusterByWordEmbeddingFeature(listtuple_pred_true_text, wordVectorsDic)
   
   #change pred labels
   '''newPredSeed=1000
   new_outlier_pred_true_txts_in_all_clusters=[]   
   for pred_true_txt in outlier_pred_true_txts_in_all_clusters:
      predLabel= int(pred_true_txt[0])+newPredSeed
      trueLabel= pred_true_txt[1]
      txt=pred_true_txt[2]
      #print(predLabel, trueLabel, txt)
      new_outlier_pred_true_txts_in_all_clusters.append(
      [str(predLabel), trueLabel, txt])	  
   #end change pred labels   
   
   print(len(new_outlier_pred_true_txts_in_all_clusters))
   print(len(non_outlier_pred_true_txts_in_all_clusters))
   merged_new_pred_true_txts=new_outlier_pred_true_txts_in_all_clusters+ non_outlier_pred_true_txts_in_all_clusters
   print(len(merged_new_pred_true_txts))
   
   print("After merge")
   printClusterEvaluation_list(merged_new_pred_true_txts)

   merged_new_label_pred_true_txts_all_batch=merged_new_label_pred_true_txts_all_batch+merged_new_pred_true_txts  
   
   non_outlier_pred_true_txts_all_batch=non_outlier_pred_true_txts_all_batch+ non_outlier_pred_true_txts_in_all_clusters 

   mstream_pred_true_txts_all_batch=mstream_pred_true_txts_all_batch+non_outlier_pred_true_txts_in_all_clusters+ outlier_pred_true_txts_in_all_clusters'''
   
   #ClusterByHDbScan(non_outlier_pred_true_txts_in_all_clusters, avgItemsInCluster_in_a_batch)
   
   
   #job_folder=predTrueTextfile.split("/")[len(predTrueTextfile.split("/"))-1]
   #enhanceClustersByIterativeClassification(non_outlier_pred_true_txts_in_all_clusters, 55, 95, 10,job_folder)
   print("\n")

'''print("merged_new_label_pred_true_txts_all_batch")
print(len(merged_new_label_pred_true_txts_all_batch))
printClusterEvaluation_list(merged_new_label_pred_true_txts_all_batch)

print("non_outlier_pred_true_txts_all_batch")
print(len(non_outlier_pred_true_txts_all_batch))
printClusterEvaluation_list(non_outlier_pred_true_txts_all_batch)

print("mstream_pred_true_txts_all_batch")
print(len(mstream_pred_true_txts_all_batch))
printClusterEvaluation_list(mstream_pred_true_txts_all_batch)'''