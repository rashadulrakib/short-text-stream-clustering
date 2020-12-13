#from CacheEmbeddings import getWordEmbedding
from sklearn.cluster import KMeans
from groupTxt_ByClass import groupTxtByClass
#from general_util import split_pred_true_txt_from_list
#from general_util import combine_pred_true_txt_from_list
#from print_cluster_evaluation import printClusterEvaluation_list
from evaluation import Evaluate
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from txt_process_util import stem_text
from txt_process_util import getScikitLearn_StopWords
from txt_process_util import processTextsRemoveStopWordTokenized
from txt_process_util import getDocFreq
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sent_vecgenerator import generate_sent_vecs_toktextdata
from sent_vecgenerator import generate_weighted_sent_vecs_toktextdata
from sent_vecgenerator import generate_sent_vecs_toktextdata_DCT
from general_util import extrcatLargeClusterItems
from general_util import findMinMaxLabel
from general_util import change_pred_label
from general_util import print_by_group
#from outliers_detection_connectedComp import removeOutlierConnectedComponentLexicalByItem
import numpy as np

def clusteringDCT(pred_true_txt_ind_prevPreds, wordVectorsDic, batchDocs, maxPredLabel):
  print("#m-stream-cleaned")
  Evaluate(pred_true_txt_ind_prevPreds)
    
  pred_true_text_ind_prevPreds_to_cluster, pred_true_text_ind_prevPreds_to_not_cluster=extrcatLargeClusterItems(pred_true_txt_ind_prevPreds)
  print("3 rd="+str(pred_true_text_ind_prevPreds_to_cluster[0][3]))
  print("4 rd="+str(pred_true_text_ind_prevPreds_to_cluster[0][4]))
  
  '''minPredToC, maxPredToC, minTrueToC, maxTrueToC=findMinMaxLabel(pred_true_text_ind_prevPreds_to_cluster)
  print("minPred, maxPred, minTrue, maxTrue=(pred_true_text_ind_prevPreds_to_cluster)") 
  print(minPredToC, maxPredToC, minTrueToC, maxTrueToC)
  
  minPredToNC, maxPredToNC, minTrueToNC, maxTrueToNC=findMinMaxLabel(pred_true_text_ind_prevPreds_to_not_cluster)
  print("minPred, maxPred, minTrue, maxTrue=(pred_true_text_ind_prevPreds_to_not_cluster)") 
  print(minPredToNC, maxPredToNC, minTrueToNC, maxTrueToNC)'''
  
  all_pred_clusters=len(groupTxtByClass(pred_true_txt_ind_prevPreds, False))
  pred_clusters=len(groupTxtByClass(pred_true_text_ind_prevPreds_to_cluster, False))
  non_pred_clusters=len(groupTxtByClass(pred_true_text_ind_prevPreds_to_not_cluster, False))  
 
  print("#clusters="+str(pred_clusters))
  print("#not clusters="+str(non_pred_clusters))
  print("this clustering with embedding DCT")
  pred_clusters=non_pred_clusters-pred_clusters
  print("#update clusters="+str(pred_clusters))  
 
  nparr=np.array(pred_true_text_ind_prevPreds_to_cluster)
  print("3 rd="+str(pred_true_text_ind_prevPreds_to_cluster[0][3]))
  print("4 rd="+str(pred_true_text_ind_prevPreds_to_cluster[0][4]))  
  preds=list(nparr[:,0])
  trues=list(nparr[:,1])
  texts=list(nparr[:,2])
  inds=list(nparr[:,3])
  prevPreds=list(nparr[:,4])  
    
  skStopWords=getScikitLearn_StopWords()
  texts= processTextsRemoveStopWordTokenized(texts, skStopWords)	
      
  '''dicDocFreq=getDocFreq(texts)
  dctCoffs=1
  X=generate_sent_vecs_toktextdata_DCT(texts, wordVectorsDic, 300,dctCoffs)  
  #vectorizer = TfidfVectorizer(tokenizer=stem_text,max_df=0.5,min_df=1)
  #vectorizer = TfidfVectorizer(max_df=0.5,min_df=2, stop_words='english')
  #X = vectorizer.fit_transform(texts)'''
  
  '''svd = TruncatedSVD(50)
  #svd = PCA(n_components=50)	
  normalizer = Normalizer(copy=False)
  lsa = make_pipeline(svd, normalizer)
  #X=X.toarray()	
  X = lsa.fit_transform(X)'''
  
  '''km = KMeans(n_clusters=pred_clusters, init='k-means++', max_iter=100,random_state=0)	
  km.fit(X)
  list_km_pred_true_text=combine_pred_true_txt_from_list(km.labels_, trues, texts)
  print("#k-means")	
  Evaluate(list_km_pred_true_text)'''	  
 
  '''ward = AgglomerativeClustering(n_clusters=pred_clusters, linkage='ward').fit(X)
  list_hr_pred_true_text=combine_pred_true_txt_from_list(ward.labels_, trues, texts)
  print("#hr-ward-DCT")
  print(min(ward.labels_), max(ward.labels_))
  pred_true_text_ind_prevPreds_to_not_cluster_hr=change_pred_label(pred_true_text_ind_prevPreds_to_not_cluster, pred_clusters+1)  
  Evaluate(list_hr_pred_true_text)
  Evaluate(list_hr_pred_true_text+pred_true_text_ind_prevPreds_to_not_cluster_hr)
  '''
  
  X = generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
  ward = AgglomerativeClustering(n_clusters=pred_clusters, linkage='ward').fit(X)
  list_hr_pred_true_text_ind_prevPred=np.column_stack((ward.labels_, trues, texts, inds, prevPreds)).tolist()
  print("#hr-ward-AVG")
  pred_true_text_ind_prevPreds_to_not_cluster_hr=change_pred_label(pred_true_text_ind_prevPreds_to_not_cluster, pred_clusters+1)  
  Evaluate(list_hr_pred_true_text_ind_prevPred)
  Evaluate(list_hr_pred_true_text_ind_prevPred+pred_true_text_ind_prevPreds_to_not_cluster_hr)
  #print_by_group(list_hr_pred_true_text+pred_true_text_ind_prevPreds_to_not_cluster_hr)

  print("#spectral-avg")
  clustering = SpectralClustering(n_clusters=pred_clusters,assign_labels="discretize", random_state=0).fit(X)
  list_sp_pred_true_text_ind_prevPred=np.column_stack((clustering.labels_, trues, texts, inds, prevPreds)).tolist()
  pred_true_text_ind_prevPreds_to_not_cluster_spec=change_pred_label(pred_true_text_ind_prevPreds_to_not_cluster, pred_clusters+1)   
  Evaluate(list_sp_pred_true_text_ind_prevPred)
  Evaluate(list_sp_pred_true_text_ind_prevPred+pred_true_text_ind_prevPreds_to_not_cluster_spec)
  
  #outliers, non_outliers, avgItemsInCluster, maxPredLabel=removeOutlierConnectedComponentLexicalByItem(list_sp_pred_true_text+pred_true_text_ind_prevPreds_to_not_cluster_spec, batchDocs, maxPredLabel)  

  	  