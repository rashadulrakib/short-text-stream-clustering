from sklearn.cluster import KMeans
from groupTxt_ByClass import groupTxtByClass
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture

def clusterByTfIdfFeature(list_pred_true_text):
    print("pred_mstreams")
    printClusterEvaluation_list(list_pred_true_text)
    dic_tupple_class=groupTxtByClass(list_pred_true_text, False)
    pred_clusters=len(dic_tupple_class)
    print("pred_clusters for k-means="+str(pred_clusters))

    preds, trues, texts	= split_pred_true_txt_from_list(list_pred_true_text)
    skStopWords=getScikitLearn_StopWords()
    texts= processTextsRemoveStopWordTokenized(texts, skStopWords)	
    vectorizer = TfidfVectorizer(tokenizer=stem_text,max_df=0.5,min_df=2)
    #vectorizer = TfidfVectorizer(max_df=0.5,min_df=2, stop_words='english')	
    X = vectorizer.fit_transform(texts)
	
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
	