import glob
import numpy as np
from statistics import mean 

folderName="/home/owner/PhD/MStream-master/MStream/eval-result/mstream-enhance/Tweets-T/*"

#folderName="/home/owner/PhD/MStream-master/MStream/eval-result/mstream-kdd/Tweets-T/*"

print(folderName)

filelist=glob.glob(folderName)
print(len(filelist))

homos=[]
nmis=[]
purities=[]

for fileName in filelist:
  #print(fileName)
  
  file=open(fileName,"r")
  lines=file.readlines()
  file.close()
  
  #print(lines)
  homo=float(lines[2].strip().replace("homogeneity_score-whole-data:   ",""))
  nmi=float(lines[3].strip().replace("nmi_score-whole-data:   ",""))
  purity=float(lines[5].strip().replace("purity majority whole data=",""))
  homos.append(homo)
  nmis.append(nmi)
  purities.append(purity)

print("homoginity ",mean(homos))
print("nmi ",mean(nmis))
print("purity ",mean(purities))   
  
  
    

