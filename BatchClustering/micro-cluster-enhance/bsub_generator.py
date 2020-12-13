import numpy as np

fixedMinP=0.5
fraction=0.05
startMaxP=fixedMinP+fraction
fixedMaxP=0.95
dataset="tweet-2472"

maxp_probs=np.arange(startMaxP, fixedMaxP+fraction, fraction)
print(maxp_probs)
for maxP in maxp_probs:
 maxP=round(maxP, 2)
 for i in range(20):
  cluster_file_id=i+1
  job_folder=dataset+"_"+str(fixedMinP)+"_"+str(maxP)+"_"+str(cluster_file_id)
  print("bsub -oo tweet-2472-kmeans-we-lr-tfidf-"+str(fixedMinP)+"_"+str(maxP)+"_"+str(cluster_file_id)+" -n 1 -M 4GB python itr_clustering_multipass_external_arg_classification.py "+str(cluster_file_id)+" " +str(fixedMinP)+" "+str(maxP)+" "+job_folder)

#bsub -oo o1 -n 1 -M 4GB python itr_clustering_multipass_external_arg_classification.py 1 0.6 0.8 0.05  
#bsub -oo websnippet-2280-kmeans-we-lr-tfidf-out$i-$maxP-$fixedMinP-$minDelta -n 1 -M 4GB python itr_clustering_multipass_external_arg_classification.py $i $maxP $fixedMinP $minDelta