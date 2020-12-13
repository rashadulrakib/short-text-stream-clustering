import json
from Document import Document
#from textblob import TextBlob
from nltk.stem import PorterStemmer 

isProposed=True

class DocumentSet:

    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0  # The number of documents
        # self.clusterNoArray = []
        self.documents = []
        ps = PorterStemmer()		
        with open(dataDir) as input:
            line = input.readline()
            while line:
                self.D += 1
                obj = json.loads(line)
                text = obj['textCleaned']
                if isProposed==True: 				
                  words=[ps.stem(w) for w in text.strip().split(' ')]
                else:
                  words=text.strip().split()
                text=' '.join(words)              				
                '''#process the text  				
                blob = TextBlob(text)
                np_phrases=blob.noun_phrases
                for np_ph in np_phrases:
                  np_ph_arr=np_ph.split(' ')
                  len_np_ph_arr=len(np_ph_arr)
                  if len_np_ph_arr==1:
                    continue				  
                  new_np=np_ph				  
                  if len_np_ph_arr>2:
                    new_np=np_ph_arr[0]+' '+np_ph_arr[1]
                  text=text.replace(new_np, new_np.replace(' ','')) 					
                #end process the text'''
                #bigram extract				
                #words=text.strip().split()
                #bi_grams=[]				
                '''for j in range(len(words)):
                   for k in range(j+1,len(words)): 				   
                       bi_grams.append(concatWordsSort([words[j], words[k]]))'''
                #for j in range(len(words)-1):
                  #bi_grams.append(concatWordsSort([words[j], words[j+1]]))  				
                #text=' '.join(bi_grams)     				   
                #end bigram extract				
                clusterNo = obj['clusterNo']
                #print("concat->", text)				
                document = Document(text, clusterNo, -1, wordToIdMap, wordList, int(obj['Id'])) #rakib
                self.documents.append(document)
                line = input.readline()
        print("number of documents is ", self.D)