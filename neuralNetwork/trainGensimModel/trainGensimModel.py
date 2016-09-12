import sys, os, cPickle, logging
from collections import defaultdict
import operator
import gensim
from nltk.corpus import stopwords
import function

class SentenceIn(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.stop = stopwords.words('english')
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            with open(os.path.join(self.dirname, fname), 'r') as fr:
                fline = cPickle.load(fr)
                newline = []
                for word in fline:
                    if word in self.stop or any(char.isdigit() for char in word):
                        continue
                    newline.append(word)
                yield newline

if __name__ == '__main__':

 
    logger = function.setLog("../Log/trainGensimModel.log")

    try:
        tokenDir = sys.argv[1]
    except:
        logger.info("not enough arguments!")

    #modelPath = "../gensimModel"
    #if os.path.exists(modelPath):
    #    model = gensim.models.Word2Vec.load(modelPath)
    #else:
    #    os.makedirs("../gensimModel")
    logger.info("start training")
    sentences = SentenceIn(tokenDir)
    model = gensim.models.Word2Vec(sentences, min_count = 25, size = 500)
    model.save("../gensimModel/gensimModel")
