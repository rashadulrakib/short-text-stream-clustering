from read_clust_label import readClustLabel
from combine_predtruetext import combinePredTrueText
from groupTxt_ByClass import groupTxtByClass
from word_vec_extractor import populateTermVecs
from nltk.tokenize import word_tokenize
from sent_vecgenerator import generate_sent_vecs_toktextdata
from sklearn.ensemble import IsolationForest
from generate_TrainTestTxtsTfIdf import comPrehensive_GenerateTrainTestTxtsByOutliersTfIDf
from generate_TrainTestVectorsTfIdf import generateTrainTestVectorsTfIDf
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn import metrics
from nltk.corpus import stopwords
from txt_process_util import processTxtRemoveStopWordTokenized
from nltk.tokenize import word_tokenize

extClustFile = "/home/owner/PhD/dr.norbert/dataset/shorttext/biomedical/2n-biomedical-w2vec-sparse-alpha-20000-0-labels"
clustlabels=readClustLabel(extClustFile)

dataFileTxtTrue = "/home/owner/PhD/dr.norbert/dataset/shorttext/biomedical/biomedicalraw"
listtuple_pred_true_text, uniqueTerms=combinePredTrueText(clustlabels, dataFileTxtTrue)

for itr in range(30):
 print("itr="+str(itr))
 for items in range(700,1050,50):
  trainTup_pred_true_txt, testTup_pred_true_txt= comPrehensive_GenerateTrainTestTxtsByOutliersTfIDf(listtuple_pred_true_text,items) 
  perct_train_inst = len(trainTup_pred_true_txt)/len(listtuple_pred_true_text)
  print("perct_train_inst="+str(perct_train_inst))
  if perct_train_inst > 0.85:
   #del trainTup_pred_true_txt
   #del testTup_pred_true_txt
   break

  X_train, train_labels, X_test, test_labels= generateTrainTestVectorsTfIDf(trainTup_pred_true_txt, testTup_pred_true_txt)
  clf = LogisticRegression() 
  t0 = time()
  clf.fit(X_train, train_labels)
  train_time = time() - t0
  print ("train time: %0.3fs" % train_time)

  t0 = time()
  preds = clf.predict(X_test)
  test_time = time() - t0
  print ("test time:  %0.3fs" % test_time)
  #change pred_labels of testTup_pred_true_txt
  for i in range(len(testTup_pred_true_txt)):
   testTup_pred_true_txt[i][0]=preds[i]    
    
  #merge trainTup_pred_true_txt and testTup_pred_true_txt, and create listtuple_pred_true_text
  #del listtuple_pred_true_text
  listtuple_pred_true_text = trainTup_pred_true_txt+ testTup_pred_true_txt
  print("listtuple_pred_true_text="+str(len(listtuple_pred_true_text)))
  #del trainTup_pred_true_txt
  #del testTup_pred_true_txt
  y_test = [int(i) for i in test_labels]
  pred_test = [int(i) for i in preds]
  score = metrics.homogeneity_score(y_test, pred_test)
  print ("homogeneity_score:   %0.6f" % score)
  score = metrics.completeness_score(y_test, pred_test)
  print ("completeness_score:   %0.6f" % score)
  score = metrics.v_measure_score(y_test, pred_test)
  print ("v_measure_score:   %0.6f" % score)
  score = metrics.accuracy_score(y_test, pred_test)
  print ("acc_score:   %0.6f" % score)
  score = metrics.normalized_mutual_info_score(y_test, pred_test)  
  print ("nmi_score:   %0.6f" % score) 
  outFileName= "/home/owner/PhD/dr.norbert/dataset/shorttext/biomedical/semisupervised/tfidf/"+str(itr)+"_"+str(items)
  file2=open(outFileName,"w")
  for i in range(len(listtuple_pred_true_text)):
   file2.write(listtuple_pred_true_text[i][0]+"\t"+listtuple_pred_true_text[i][1]+"\t"+listtuple_pred_true_text[i][2]+"\n")

  file2.close() 
     
 


