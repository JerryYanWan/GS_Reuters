import function
from database import database
import sys, os, csv
import gensim
import cPickle
import cnn
import numpy as np
import json

def predict(samples, tag_lookup_tbl, n_largest, model_direct, tags, logger, outputFile):
    model_read = os.path.join(model_direct, "best_model.pkl")
    with open(model_read, 'r') as f:
        classifier = cPickle.load(f)
    n_label = len(tags)
    #function.hitRate(samples, classifier, n_largest, n_label, logger)
    #function.accuracy_upperbound(samples, classifier, n_largest, n_label, logger)
    #function.metric(samples, classifier, n_largest, n_label, logger)
    results = function.decoding(samples, classifier, n_largest, logger)
    with open(outputFile, "wb") as f:
        writer = csv.writer(f)
        for key, val in results.items():
            writer.writerow([key, tags[val]])
    logger.info("finish! write to %s" % outputFile)
    

if __name__ == '__main__':
    logger = function.setLog('../../Log/test.log')
    try:
        testTokenDir = sys.argv[1]
        gensimPath   = sys.argv[2]
        n_largest    = (int) (sys.argv[3])
        outputFile   = sys.argv[4]
    except:
        logger.info("Input Error! Lack arguments!")

    vector_size = 500
    id_tag = function.getIdTag()
    logger.info("# of files = %s" % len(id_tag))
    tags = function.getTags("../../../csvFiles/tagDistribution.csv")
    logger.info("tags = %s"%tags)
    GensimModel = gensim.models.Word2Vec.load(gensimPath)
    logger.info("# of words in the dictionary = %s" %len(GensimModel.vocab))
    tag_lookup_tbl = function.tagLookupTbl(GensimModel, tags, vector_size)
    test_length = len(os.listdir(testTokenDir))
    model_direct = "../model"
    samples = function.DecodeIn(testTokenDir, GensimModel, vector_size)
    #samples = function.SentenceIn(testTokenDir, id_tag, tags, GensimModel, vector_size)
    predict(samples, tag_lookup_tbl, n_largest, model_direct, tags, logger, outputFile)
    
