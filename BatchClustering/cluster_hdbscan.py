import hdbscan
from sklearn.cluster import DBSCAN
from groupTxt_ByClass import groupTxtByClass
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import math
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from print_cluster_evaluation import printClusterEvaluation_list

def ClusterByHDbScan(listtuple_pred_true_text, avgItemsInCluster_in_a_batch):
  print("\nClusterByHDbScan") 
  printClusterEvaluation_list(listtuple_pred_true_text)
  print(len(listtuple_pred_true_text), avgItemsInCluster_in_a_batch)
  
  dic_tupple_class_predicted=groupTxtByClass(listtuple_pred_true_text, False)  
  numberOfClusters_predicted=len(dic_tupple_class_predicted) 
  
  dic_tupple_class_true=groupTxtByClass(listtuple_pred_true_text, True)  
  numberOfClusters_true=len(dic_tupple_class_true)
  
  print("numberOfClusters_true="+str(numberOfClusters_true)+", numberOfClusters_predicted="+str(numberOfClusters_predicted))
  
  train_data = []
  train_predlabels = []
  train_trueLabels = [] 
  
  for pred_true_text in listtuple_pred_true_text:
    train_predlabels.append(pred_true_text[0]) 
    train_trueLabels.append(pred_true_text[1]) 
    train_data.append(pred_true_text[2])	
  
  vectorizer = TfidfVectorizer( max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
  X = vectorizer.fit_transform(train_data)
  
  
  svd = TruncatedSVD(2)
  normalizer = Normalizer(copy=False)
  lsa = make_pipeline(svd, normalizer)
  X_svd = lsa.fit_transform(X)
  
  min_cluster_size_in_a_batch=int(math.ceil(avgItemsInCluster_in_a_batch))
  
  min_cluster_size_in_a_batch=2
  
  clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_in_a_batch)
  clusterer.fit(X)
  X_hdbscan_labels=clusterer.labels_
  
  print("X-total-clusters="+str(X_hdbscan_labels.max()))
  print("Homogeneity: %0.4f" % metrics.homogeneity_score(train_trueLabels, X_hdbscan_labels))
  print("Completeness: %0.4f" % metrics.completeness_score(train_trueLabels, X_hdbscan_labels))
  print("V-measure: %0.4f" % metrics.v_measure_score(train_trueLabels, X_hdbscan_labels))
  print("Adjusted Rand-Index: %.4f" % metrics.adjusted_rand_score(train_trueLabels, X_hdbscan_labels))
  print ("nmi_score-whole-data:   %0.4f" % metrics.normalized_mutual_info_score(train_trueLabels, X_hdbscan_labels, average_method='arithmetic' ) )
  
  clusterer_svd = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_in_a_batch)
  clusterer_svd.fit(X_svd)
  X_svd_hdbscan_labels=clusterer_svd.labels_
  
  db = DBSCAN().fit(X_svd)
  X_svd_dbscan_labels=db.labels_
  
  print("X-svd-total-clusters="+str(X_svd_hdbscan_labels.max()))
  print("Homogeneity: %0.4f" % metrics.homogeneity_score(train_trueLabels, X_svd_hdbscan_labels))
  print("Completeness: %0.4f" % metrics.completeness_score(train_trueLabels, X_svd_hdbscan_labels))
  print("V-measure: %0.4f" % metrics.v_measure_score(train_trueLabels, X_svd_hdbscan_labels))
  print("Adjusted Rand-Index: %.4f" % metrics.adjusted_rand_score(train_trueLabels, X_svd_hdbscan_labels))
  print ("nmi_score-whole-data:   %0.4f" % metrics.normalized_mutual_info_score(train_trueLabels, X_svd_hdbscan_labels, average_method='arithmetic' ) )
  
  print("X-svd-dbscan-total-clusters="+str(X_svd_dbscan_labels.max()))
  print("Homogeneity: %0.4f" % metrics.homogeneity_score(train_trueLabels, X_svd_dbscan_labels))
  print("Completeness: %0.4f" % metrics.completeness_score(train_trueLabels, X_svd_dbscan_labels))
  print("V-measure: %0.4f" % metrics.v_measure_score(train_trueLabels, X_svd_dbscan_labels))
  print("Adjusted Rand-Index: %.4f" % metrics.adjusted_rand_score(train_trueLabels, X_svd_dbscan_labels))
  print ("nmi_score-whole-data:   %0.4f" % metrics.normalized_mutual_info_score(train_trueLabels, X_svd_dbscan_labels, average_method='arithmetic' ) )
  
  
 