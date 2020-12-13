import random

def SampleData(dataset, percent):
  file1=open(dataset,"r")
  linesNews = file1.readlines()
  file1.close()
  length=len(linesNews)
  total=int(length*percent)
  randIndecies = [random.randint(0,length) for i in range(total)]
  sublist = [linesNews[index] for index in randIndecies]
  return sublist
  
sublist=SampleData("NT-mstream-long",1.0)
for item in sublist:
  item=item.strip()
  print(item)
  