

Title: Efficient clustering of short text streams using online-offline clustering
https://dl.acm.org/doi/abs/10.1145/3469096.3469866

ABSTRACT
Short text stream clustering is an important but challenging task since massive amount of text is generated from different sources such as micro-blogging, question-answering, and social news aggregation websites. The two major challenges of clustering such massive amount of text is to cluster them within a reasonable amount of time and to achieve better clustering result. To overcome these two challenges, we propose an efficient short text stream clustering algorithm (called EStream) consisting of two modules: online and offline. The online module of EStream algorithm assigns a text to a cluster one by one as it arrives. To assign a text to a cluster it computes similarity between a text and a selected number of clusters instead of all clusters and thus significantly reduces the running time of the clustering of short text streams. EStream assigns a text to a cluster (new or existing) using the dynamically computed similarity thresholds. Thus EStream efficiently deals with the concept drift problem. The offline module of EStream algorithm enhances the distributions of texts in the clusters obtained by the online module so that the upcoming short texts can be assigned to the appropriate clusters.

Experimental results demonstrate that EStream outperforms the state-of-the-art short text stream clustering methods (in terms of clustering result) by a statistically significant margin on several short text datasets. Moreover, the running time of EStream is several orders of magnitude faster than that of the state-of-the-art methods.
