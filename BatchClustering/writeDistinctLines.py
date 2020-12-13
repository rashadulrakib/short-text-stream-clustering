file=open("/home/owner/PhD/MStream-master/MStream/result/NewsPredTueTextMStreamSemantic.txt","r")
lines = file.readlines()
file.close()

setLines= []

for line in lines:
    line=line.strip()
    if len(line)==0:
        continue
    setLines.append(line)		

file=open("/home/owner/PhD/MStream-master/MStream/result/NewsPredTueTextMStreamSemantic.txt","w")
setLines=set(setLines)
for setLine in setLines:
    print(setLine)
    file.write(setLine+"\n")    
file.close()
