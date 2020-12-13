from sklearn.cluster import KMeans
from groupTxt_ByClass import groupTxtByClass
from groupTxt_ByClass import groupItemsBySingleKeyIndex
from general_util import split_pred_true_txt_from_list
from general_util import combine_pred_true_txt_from_list
from print_cluster_evaluation import printClusterEvaluation_list
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
from sent_vecgenerator import generate_sent_vecs_toktextdata_autoencoder
import numpy as np

def clusterByWordEmbeddingIntelligent(list_pred_true_text_ind_prevind, wordVectorsDic):
  print("pred_mstreams")
  printClusterEvaluation_list(list_pred_true_text_ind_prevind)
  dic_itemGroups=groupItemsBySingleKeyIndex(list_pred_true_text_ind_prevind,0)
  
  pred_clusters= int(len(dic_itemGroups)/1.0) #needs to be determined carefully
  
  dic_group_sizes=[len(dic_itemGroups[x]) for x in dic_itemGroups if isinstance(dic_itemGroups[x], list)]
  print(dic_group_sizes)
  
  print("#clusters="+str(pred_clusters))
  
  nparr=np.array(list_pred_true_text_ind_prevind) 
  preds=list(nparr[:,0])
  trues=list(nparr[:,1])
  word_arr=list(nparr[:,2])
  inds=list(nparr[:,3])
  X=generate_sent_vecs_toktextdata(word_arr, wordVectorsDic, 300) 
  #X=generate_sent_vecs_toktextdata_autoencoder(word_arr, wordVectorsDic, 300, pred_clusters)

  svd = TruncatedSVD(50)
  #svd = PCA(n_components=50)	
  normalizer = Normalizer(copy=False)
  lsa = make_pipeline(svd, normalizer)
  #X=X.toarray()	
  X = lsa.fit_transform(X)  
  
  ward = AgglomerativeClustering(n_clusters=pred_clusters, linkage='ward').fit(X)
  list_hr_pred_true_text=combine_pred_true_txt_from_list(ward.labels_, trues, word_arr)
  print("hr-ward")	
  printClusterEvaluation_list(list_hr_pred_true_text)	
	
  clustering = SpectralClustering(n_clusters=pred_clusters,assign_labels="discretize", random_state=0).fit(X)
  list_sp_pred_true_text=combine_pred_true_txt_from_list(clustering.labels_, trues, word_arr)
  print("spectral")	
  printClusterEvaluation_list(list_sp_pred_true_text)

def clusterByWordEmbeddingFeature(list_pred_true_text, wordVectorsDic):
  print("pred_mstreams")
  printClusterEvaluation_list(list_pred_true_text)
  dic_tupple_class=groupTxtByClass(list_pred_true_text, False)
  pred_clusters=len(dic_tupple_class)
  print("#clusters="+str(pred_clusters))

  preds, trues, texts= split_pred_true_txt_from_list(list_pred_true_text)
  skStopWords=getScikitLearn_StopWords()
  texts= processTextsRemoveStopWordTokenized(texts, skStopWords)	
      
  dicDocFreq=getDocFreq(texts)
	
  X = generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
  
  #X = generate_weighted_sent_vecs_toktextdata(texts, wordVectorsDic, dicDocFreq, 300) #not good
  
  svd = TruncatedSVD(100)
  #svd = PCA(n_components=50)	
  normalizer = Normalizer(copy=False)
  lsa = make_pipeline(svd, normalizer)
  #X=X.toarray()	
  X = lsa.fit_transform(X)
  
  km = KMeans(n_clusters=pred_clusters, init='k-means++', max_iter=100,random_state=0)	
  km.fit(X)
  list_km_pred_true_text=combine_pred_true_txt_from_list(km.labels_, trues, texts)
  print("k-means")	
  printClusterEvaluation_list(list_km_pred_true_text)	  
 
  ward = AgglomerativeClustering(n_clusters=pred_clusters, linkage='ward').fit(X)
  list_hr_pred_true_text=combine_pred_true_txt_from_list(ward.labels_, trues, texts)
  print("hr-ward")	
  printClusterEvaluation_list(list_hr_pred_true_text)	
	
  clustering = SpectralClustering(n_clusters=pred_clusters,assign_labels="discretize", random_state=0).fit(X)
  list_sp_pred_true_text=combine_pred_true_txt_from_list(clustering.labels_, trues, texts)
  print("spectral")	
  printClusterEvaluation_list(list_sp_pred_true_text)   

  brc = Birch(branching_factor=50, n_clusters=pred_clusters, threshold=0.5, compute_labels=True)
  brc.fit_predict(X)
  list_brc_pred_true_text=combine_pred_true_txt_from_list(brc.labels_, trues, texts)
  print("brc")	
  printClusterEvaluation_list(list_brc_pred_true_text)
	
  gmm = GaussianMixture(n_components=pred_clusters, covariance_type='full')
  gmm_labels=gmm.fit_predict(X)
  list_gmm_pred_true_text=combine_pred_true_txt_from_list(gmm_labels, trues, texts)
  print("gmm")	
  printClusterEvaluation_list(list_gmm_pred_true_text) 