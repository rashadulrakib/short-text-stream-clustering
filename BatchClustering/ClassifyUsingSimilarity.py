from scipy.spatial import distance

def computeL2Dist(docVec, wordsVec):
    l2dist = distance.euclidean(docVec, wordsVec)	
    return l2dist
	
def computeClosestL2Distance(docVec, wordsVectorPerClusterForTraining):
    l2MinDistance=10000000
    predLabel=1	

    for clusterid, wordsVec in wordsVectorPerClusterForTraining.items():        
        #print("computeClosestL2Distance clusterid="+str(clusterid)+", wordsVec="+str(wordsVec))
        l2dist = computeL2Dist(docVec, wordsVec)
        if l2MinDistance >= l2dist:
            l2MinDistance = l2dist
            predLabel = clusterid			
	
    return [l2MinDistance, predLabel]
    

def classifyBySimilarity(wordsVectorPerClusterForTraining, docsVectorPerClusterToBeReclassified, docsPerClusterToBeReclassified):
    docsLabelPerClusterForToBeReclassified={}
    
    #for clusterid, docs in docsPerClusterToBeReclassified.items():
    #    for doc in docs:
    #        print("classifyBySimilarity clusterid="+str(clusterid)+", doc="+doc.text)		
	
    for clusterid, docVectors in docsVectorPerClusterToBeReclassified.items():
        docs = docsPerClusterToBeReclassified[clusterid]
        for i in range(len(docs)):
            docVec = docVectors[i]		
		#for docVec in docVectors:
            #print("classifyBySimilarity clusterid="+str(clusterid)+", doc Vec="+str(docVec))	
            l2MinDistance, predLabel=computeClosestL2Distance(docVec, wordsVectorPerClusterForTraining)
            print("classifyBySimilarity l2MinDistance="+str(l2MinDistance)+", doc predLabel="+str(predLabel))			
            docsLabelPerClusterForToBeReclassified.setdefault(predLabel,[]).append(docs[i])			
      
    return docsLabelPerClusterForToBeReclassified   

def classifyBySimilaritySingle(singleDocVector, wordsVectorPerClusterForTraining):
   l2MinDistance, predLabel=computeClosestL2Distance(singleDocVector, wordsVectorPerClusterForTraining)
   print("classifyBySimilarity l2MinDistance="+str(l2MinDistance)+", doc new predLabel="+str(predLabel))
   #doc.predLabel=predLabel
   return predLabel	