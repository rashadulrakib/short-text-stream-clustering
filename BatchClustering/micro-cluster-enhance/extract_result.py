import glob
import numpy as np

filelist=glob.glob("/dshome/grad/rakib/parameter_tuning/2.2.fixed-MinP/2.2.tweet-2472-kmeans-we-lr-tfidf/*")
print(len(filelist))

nmiDic={}
purityDic={}

for fileName in filelist:
 #arr=fileName.split('-')
 #print(fileName) 
 #fileKey=arr[len(arr)-3]+"-"+arr[len(arr)-2]+"-"+arr[len(arr)-1]
 
 fileKey=fileName.replace("/dshome/grad/rakib/parameter_tuning/2.2.fixed-MinP/2.2.tweet-2472-kmeans-we-lr-tfidf/tweet-2472-kmeans-we-lr-tfidf-", "")
 #print(fileKey)
 
 arr=fileKey.split("_")
 fileKey=arr[1]
 
 file=open(fileName,"r")
 lines=file.readlines()
 file.close()
 
 tlines=len(lines)
 nmi=str(lines[tlines-3]).strip().replace("nmi_score-whole-data:   ",'')  
 purity=str(lines[tlines-1]).strip().replace("purity majority whole data=",'')
 #print(fileKey+"\t"+purity+"\t"+nmi)
 nmiDic.setdefault(fileKey, []).append(float(nmi))
 purityDic.setdefault(fileKey, []).append(float(purity))
 
#print(nmiDic) 
#print(purityDic)
for key in purityDic.keys(): 
 nmis= nmiDic[key] 
 nmis=np.array(nmis)
 nmi_mean=np.mean(nmis)
 nmi_std=np.std(nmis)
 
 purities=purityDic[key]
 purities=np.array(purities)
 purity_mean=np.mean(purities)
 purity_std=np.std(purities)
 
 print("minP-maxP="+key+",purity_mean="+str(purity_mean)+",purity_std="+str(purity_std)+",nmi_mean="+str(nmi_mean)+",nmi_std="+str(nmi_std))


 
