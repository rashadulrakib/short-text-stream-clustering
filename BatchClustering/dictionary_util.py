def combineTwoDictionary(dictri_keys_selectedClusters_currentBatch, dicbi_keys_selectedClusters_currentBatch, replace=True):

  new_dic_combined_keys_selectedClusters={}

  for key, txtInds in dictri_keys_selectedClusters_currentBatch.items():
    if len(txtInds)==0:
      continue	 
    new_dic_combined_keys_selectedClusters[key]=txtInds	
	  
  for key, txtInds in dicbi_keys_selectedClusters_currentBatch.items():
    if len(txtInds)==0:
      continue
    new_dic_combined_keys_selectedClusters[key]=txtInds		 	  
    if replace==False:	  
      if key in dictri_keys_selectedClusters_currentBatch:
        new_dic_combined_keys_selectedClusters[key]=txtInds+dictri_keys_selectedClusters_currentBatch[key]     	  
       	  
  return new_dic_combined_keys_selectedClusters	  
    