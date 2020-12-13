from Chains import Chains
import sys

def detect_outlier_XStream(X):
  rows = len(X) 
  outlierLabels = [1 for i in range(rows)]
  
  k = 50
  nchains = 50
  depth = 10

  cf = Chains(k=k, nchains=nchains, depth=depth)
  cf.fit(X)
  anomalyscores = -cf.score(X)
  
  return outlierLabels
