from nltk.corpus import brown
import nltk
from nltk.probability import FreqDist

from pprint import pprint
nltk.download('brown')
from nltk.util import ngrams
from collections import Counter, defaultdict
import math

from nltk import word_tokenize


class LangModel():
    
    def __init__(self,corpus):
        self.corpus=corpus
        self.vocab=None
        self.fd=None
        self.numgrams=[]
    
#calc stats such as fd and vocab        
    def calcStats(self):
        textCorpus = nltk.Text(self.corpus.words(self.corpus.fileids()))
        textFD = FreqDist(t.lower() for t in textCorpus)
        vocab=list(textFD.keys())
        unkcount=0
        for key in vocab:
            if(textFD[key]==1):
                unkcount=unkcount+1
        vocab.append("unk")
        textFD["unk"]=unkcount
        self.fd=textFD
        self.vocab=vocab
        self.numNgrams()
        return
        
#to display collocations       
    def collocations(self,topn):
        #corpus is brown by default
        n = 2
        bigrams = ngrams(self.corpus.words(), n)
        brownNgramFD = nltk.FreqDist(token for token in bigrams)
        brownNgramFD.plot(topn)
        return

#calc the number of ngrams
    def numNgrams(self):
#         n = 2
        for i in range(2,6):
            n_grams = ngrams(self.corpus.words(), i)
            NgramFD = nltk.FreqDist(token for token in n_grams)
            self.numgrams.append(len(NgramFD))
        return

#make a ngram model with or without add 1
    def makeModel(self,n,add1=False):
        # Create a placeholder for model
        model = defaultdict(lambda: defaultdict(lambda: 0))
        # Count frequency of co-occurance 
        data=brown.sents()
        for sentence in data:
            for it in range(len(sentence)):
                sentence[it]=sentence[it].lower()
            for n_grams in ngrams(sentence,n):#pad_right=True, pad_left=True,):
                pre=n_grams[0:-1]
                post=n_grams[-1]
                model[pre][post] += 1
                model[pre]["unseen"]=0
        # Let's transform the counts to probabilities
        total=0
        unkcount=0
        extend=0
        if(add1):
            if(n==2):
                extend=len(self.vocab)
            if(n==3):
                extend=self.numgrams[0]
            if(n==4):
                extend=self.numgrams[1]
            if(n==5):
                extend=self.numgrams[2]
        print("extended value: ",extend)
        for wprev in model:
#             total_count = float(sum(model[wprev].values()))
            total_count=len(model[wprev])
#             total=total+total_count
            for wcurr in model[wprev]:
#                 if(model[wprev][wcurr]==1):
#                     unkcount+=1               
                if(add1):
                    model[wprev][wcurr]+=1
                    total_count+=extend
                    model[wprev]["unseen"]=1
                else:
                    model[wprev]["unseen"]=0
                model[wprev][wcurr] /= total_count

        return model
    
#make a language model with gte smoothing
    def makeGTmodel(self,n):
        # Create a placeholder for model
        model = defaultdict(lambda: defaultdict(lambda: 0))
        # Count frequency of co-occurance 
        data=brown.sents()
        for sentence in data:
            for it in range(len(sentence)):
                sentence[it]=sentence[it].lower()
            for n_grams in ngrams(sentence,n):#pad_right=True, pad_left=True,):
                pre=n_grams[0:-1]
                post=n_grams[-1]
                model[pre][post] += 1
                model[pre]["unseen"]=0
        # Let's transform the counts to probabilities
        
        for wprev in model:
#             total_count = float(sum(model[wprev].values()))
            total_count=len(model[wprev])
#             total=total+total_count
            counts=defaultdict(lambda: 0)
            for wcurr in model[wprev]:
                counts[model[wprev][wcurr]]+=1
            for wcurr in model[wprev]:
                if(counts[model[wprev][wcurr]]!=0):
                    model[wprev][wcurr]=((model[wprev][wcurr]+1)*counts[model[wprev][wcurr]+1]/counts[model[wprev][wcurr]])/total_count
                else:
                    model[wprev][wcurr]=counts[1]/total_count
#                 if(model[wprev][wcurr]==1):
#                     unkcount+=1               
#         
        return model

#work in progress KND   
    def KneserNeyDiscountingModel(self,n):
         # Create a placeholder for model
        model = defaultdict(lambda: defaultdict(lambda: 0))
        # Count frequency of co-occurance 
        data=brown.sents()
        for sentence in data:
            for it in range(len(sentence)):
                sentence[it]=sentence[it].lower()
            for n_grams in ngrams(sentence,n):#pad_right=True, pad_left=True,):
                pre=n_grams[0:-1]
                post=n_grams[-1]
                model[pre][post] += 1
                model[pre]["unseen"]=0
        # Let's transform the counts to probabilities
        
        for wprev in model:
#             total_count = float(sum(model[wprev].values()))
            total_count=len(model[wprev])
#             total=total+total_count
            counts=defaultdict(lambda: 0)
            for wcurr in model[wprev]:
                counts[model[wprev][wcurr]]+=1
            for wcurr in model[wprev]:
                if(counts[model[wprev][wcurr]]!=0):
                    model[wprev][wcurr]=((model[wprev][wcurr]+1)*counts[model[wprev][wcurr]+1]/counts[model[wprev][wcurr]])/total_count
                else:
                    model[wprev][wcurr]=counts[1]/total_count
#                 if(model[wprev][wcurr]==1):
#                     unkcount+=1               
#         
        return model
        

#generate sentences with the trained model
    def sentGen(self,start,n,model):
#         model=models[n-2]
        count=0
#         para=start.split()
        start=start.lower()
        para=nltk.word_tokenize(start)
        
        finalpara=start
        prob=1;
        MaxKey=" "
        while(MaxKey!="." and count<30):
            if(n==2):
                pos=dict(model[para[-1],])
            if(n==3):
                pos=dict(model[para[-2],para[-1]])
            if(n==4):
                pos=dict(model[para[-3],para[-2],para[-1]])
            if(n==5):
                pos=dict(model[para[-4],para[-3],para[-2],para[-1]])
            try:
        #        
                MaxKey = max(pos, key=pos.get)
                prob=prob*pos[MaxKey]
                para.append(MaxKey)
    #             if(n==2):
    #                 print(MaxKey)
            except:
                pass
            try:
                finalpara=finalpara+" "+MaxKey
            except TypeError:
                pass
            count=count+1

        print("\n"+finalpara)
        print("prob : ",prob)
        logprob=math.log(prob,2)
        print("logprob : ",logprob)
        perp=1/((1/count)*prob)
        perplog=math.pow(2,1/((1/count)*logprob))
        print("reg perp : ",perp)
        print("log perp : ",perplog)
        return
    
#find prob of a sentence given a trained model   
    def sentProb(self,sent,n,model,printprob=False):
#         para=sent.split()
#         print(para)
        sent=sent.lower()
        para=nltk.word_tokenize(sent)
        paralen=len(para)
        
        for i in range(paralen):
            if para[i] not in self.vocab:
                para[i]="unk"
        i=0
        prob=1
        prob1=0
        avgprob=0
        count=0
        sentprob=[]
        while(i<(paralen-(n-1))):
            if(n==2):
                prob1=model[para[i],][para[i+1]]
#                 print(para[i],para[i+1],prob1)
                if(prob1==0):
                    prob1=model[para[i],]["unseen"]
            if(n==3):
                prob1=model[para[i],para[i+1]][para[i+2]]
                if(prob1==0):
                    prob1=model[para[i],para[i+1]]["unseen"]
            if(n==4):
                prob1=model[para[i],para[i+1],para[i+2]][para[i+3]]
                if(prob1==0):
                    prob1=model[para[i],para[i+1],para[i+2]]["unseen"]
            if(n==5):
                prob1=model[para[i],para[i+1],para[i+2],para[i+3]][para[i+4]]
                if(prob1==0):
                    prob1=model[para[i],para[i+1],para[i+2],para[i+3]]["unseen"]

            prob*=prob1
            if(printprob):
                print("prob :",prob)

            if(para[i+n-1]=="."):
#             if(i%20==0):
                sentprob.append(prob)
#                 print("prob each sentence : ",prob)
#                 print("prob at "+str(i)+"th"+" ngram : ",prob)
                avgprob=avgprob+prob
                prob=1
                count=count+1
            
            i=i+1
        avgprob=avgprob/count
        print("sent probs : ",sentprob)
        if(avgprob==0):
            print("Average Prob of sent : 0")
        else:
            print("avgprob : ",avgprob)
            logprob=math.log(avgprob,2)
            print("logprob : ",logprob)
            perp=paralen/(avgprob)
            perplog=math.pow(2,(1/((1/paralen)*logprob)))
            print("reg perp : ",perp)
            print("log perp : ",perplog)
        return