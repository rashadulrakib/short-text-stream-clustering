class Singleton:
   __instance = None
   embeddings={}
   @staticmethod 
   def getInstance():
      """ Static access method. """
      if Singleton.__instance == None:
         Singleton()
      return Singleton.__instance
   def __init__(self):
      """ Virtually private constructor. """
      if Singleton.__instance != None:
         raise Exception("This class is a singleton!")
      else:
         Singleton.__instance = self
         embeddingFile="/home/owner/PhD/dr.norbert/dataset/shorttext/glove.42B.300d/glove.42B.300d.txt"
         print(embeddingFile)
         file=open(embeddingFile,"r")
         vectorlines = file.readlines()
         file.close()
         lineProgCount = 0
         for vecline in vectorlines:
            vecarr = vecline.strip().split()
            lineProgCount=lineProgCount+1
            if lineProgCount % 100000 ==0:
                print(lineProgCount)
            if len(vecarr) < 20:
                continue
            vecword = vecarr[0]
            vecnumbers = list(map(float, vecarr[1:]))
            Singleton.embeddings[vecword]=vecnumbers 
         print("Finish loading #word embeddings="+str(lineProgCount))
         
#def loadAndCacheWordEmbeddings():
#   Singleton()

def getWordEmbedding(word):
   s=Singleton.getInstance()
   if word in s.embeddings: 
      return s.embeddings[word]
   else:
      return [0]*300    



