from pyod.models.iforest import IForest

def detect_outlier_other(X):
  rows = len(X) 
  outlierLabels = [1 for i in range(rows)]
  cf=IForest(contamination=0.05,random_state=0) 
  
  
  
  return outlierLabels