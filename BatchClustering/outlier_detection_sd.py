from sklearn.feature_extraction.text import TfidfVectorizer
from compute_util import compute_sim_matrix
#from compute_util import compute_mean_sd
from compute_util import compute_row_sim_I
from compute_util import compute_row_lexsimCommonCount_I
import statistics

def detect_outlier_sd_txt_tfidf(txts, percentoutlier):
 vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english', use_idf=True, smooth_idf=True, norm='l2')
 txtVecs = vectorizer.fit_transform(txts)
 outlierLabels = detect_outlier_sd_vec(txtVecs.toarray(), percentoutlier)
 return outlierLabels


def detect_outlier_sd_vec(txtVecs, percentoutlier):
 rows = len(txtVecs) 
 outlierLabels = [1 for i in range(rows)]
 #simMatrix = compute_sim_matrix(txtVecs)
 mostSimLists= {}
 for i in range(rows):
  rowSimsToI,rowSimsToI_ExcpetSelf = compute_row_sim_I(txtVecs[i], txtVecs, i)
  #meanVal,sdVal = compute_mean_sd(rowSimsToI)
  meanVal=statistics.mean(rowSimsToI_ExcpetSelf)
  sdVal = statistics.stdev(rowSimsToI_ExcpetSelf)  
  for j in range(len(rowSimsToI)):
   if i!=j and rowSimsToI[j] > meanVal + sdVal:
    if j in mostSimLists:
     mostSimLists[j]= mostSimLists[j]+1
    else:
     mostSimLists[j]=1
 
 numOutliers = int(percentoutlier*rows)
 items = [(v, k) for k, v in mostSimLists.items()]
 items.sort()
 items.reverse()
 sortedKeys = [k for v, k in items] #key is index of a text
 #print(mostSimLists)
 #print(sortedKeys)
 #print(rows, numOutliers)
 sortedKeys = sortedKeys[rows-numOutliers: rows] #outliers index
 for outlierInd in sortedKeys:
  outlierLabels[outlierInd]=-1
 
 #print("SD based outliers="+str(outlierLabels))
 return outlierLabels

def detect_outlier_sd_lexSemanticSim_auto(texts,txtVecs):
  rows = len(txtVecs) 
  outlierLabels = [1 for i in range(rows)]
  mostSimLists= {}

  if rows<2:
    return outlierLabels  
  
  for i in range(rows):
    rowSimsToI,rowSimsToI_ExcpetSelf = compute_row_sim_I(txtVecs[i], txtVecs, i)
    if len(rowSimsToI_ExcpetSelf)<2:
      return outlierLabels	
    rowLexSims_CommonCountsToI,rowLexSims_CommonCountsToI_ExcpetSelf=compute_row_lexsimCommonCount_I(texts[i], texts, i) 
    	
    meanVal=statistics.mean(rowSimsToI_ExcpetSelf)
    sdVal = statistics.stdev(rowSimsToI_ExcpetSelf)  
    for j in range(len(rowSimsToI)):
      if i!=j and rowSimsToI[j] > meanVal + sdVal:
        if j in mostSimLists:
          mostSimLists[j]= mostSimLists[j]+1
        else:
          mostSimLists[j]=1
  
  mostSimCounts=list(mostSimLists.values())
  if len(mostSimCounts)<3:
    return outlierLabels    
  #print("detect_outlier_sd_vec_auto-rows="+str(rows)+", mostSimCounts="+str(mostSimCounts))
  mostSimCount_mean=statistics.mean(mostSimCounts)
  mostSimCount_sd=statistics.stdev(mostSimCounts)  
 
  for i, mostSimCount in mostSimLists.items():
    if float(mostSimCount)< float(mostSimCount_mean-1.5*mostSimCount_sd):
      outlierLabels[i]=-1	
 
  return outlierLabels

def detect_outlier_sd_vec_auto(txtVecs):
  rows = len(txtVecs) 
  outlierLabels = [1 for i in range(rows)]
  mostSimLists= {}

  if rows<2:
    return outlierLabels  
  
  for i in range(rows):
    rowSimsToI,rowSimsToI_ExcpetSelf = compute_row_sim_I(txtVecs[i], txtVecs, i)
    meanVal=statistics.mean(rowSimsToI_ExcpetSelf)
    sdVal = statistics.stdev(rowSimsToI_ExcpetSelf)  
    for j in range(len(rowSimsToI)):
      if i!=j and rowSimsToI[j] > meanVal + sdVal:
        if j in mostSimLists:
          mostSimLists[j]= mostSimLists[j]+1
        else:
          mostSimLists[j]=1
  
  mostSimCounts=list(mostSimLists.values())
  if len(mostSimCounts)<2:
    return outlierLabels    
  #print("detect_outlier_sd_vec_auto-rows="+str(rows)+", mostSimCounts="+str(mostSimCounts))
  mostSimCount_mean=statistics.mean(mostSimCounts)
  mostSimCount_sd=statistics.stdev(mostSimCounts)  
 
  for i, mostSimCount in mostSimLists.items():
    if float(mostSimCount)< float(mostSimCount_mean-1.5*mostSimCount_sd):
      outlierLabels[i]=-1	
 
  return outlierLabels

"""def detect_outlier_sd_vec(txtVecs, percentoutlier): #working
 rows = len(txtVecs) 
 outlierLabels = [1 for i in range(rows)]
 simMatrix = compute_sim_matrix(txtVecs)
 mostSimLists= {}
 for i in range(len(simMatrix)):
  meanVal,sdVal = compute_mean_sd(simMatrix[i])
  for j in range(len(simMatrix[i])):
   if i!=j and simMatrix[i][j] > meanVal + sdVal:
    #mostSimLists[j].append(i)
    if j in mostSimLists:
     mostSimLists[j]= mostSimLists[j]+1
    else:
     mostSimLists[j]=1
 
 numOutliers = int(percentoutlier*rows)
 items = [(v, k) for k, v in mostSimLists.items()]
 items.sort()
 items.reverse()
 sortedKeys = [k for v, k in items] #key is index of a text
 #print(mostSimLists)
 #print(sortedKeys)
 #print(rows, numOutliers)
 sortedKeys = sortedKeys[rows-numOutliers: rows] #outliers index
 for outlierInd in sortedKeys:
  outlierLabels[outlierInd]=-1
 
 #print("SD based outliers="+str(outlierLabels))
 return outlierLabels"""
