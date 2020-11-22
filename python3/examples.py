from LangModel import LangModel 
import nltk
from nltk.corpus import brown
nltk.download('brown')


lm=LangModel(brown)

lm.calcStats()
lm.fd.plot(20)
lm.collocations(50)

bimodeladd1=lm.makeModel(2,add1=True)
bimodel=lm.makeModel(2)
bimodelgt=lm.makeGTmodel(2)

lm.sentGen("I",2,bimodel)

test="A day will come when we can live."
lm.sentProb(test,2,bimodel,printprob=True)
print("\n")
lm.sentProb(test,2,bimodeladd1,printprob=True)
print("\n")
lm.sentProb(test,2,bimodelgt,printprob=True)