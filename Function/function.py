import logging
import numpy as np
import heapq
import theano
from nltk.corpus import stopwords
import os, sys, csv
import cPickle
from collections import defaultdict

from database import database

def setLog(path):
    """ setting the logging configuration """
    logger = logging.getLogger('current function')
    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig( level = logging.DEBUG,
                         format = FORMAT,
                         filename = path,
                         filemode = 'wa' )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(FORMAT)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

class DecodeIn(object):
    def __init__(self, dirname, GensimModel, vector_size):
        self.dirname = dirname
        self.model   = GensimModel
        self.vector_size = vector_size
        self.stop = stopwords.words('english')
    def getTokens(self, filename):
        tokens = []
        with open(filename, 'r') as fr:
            fline = cPickle.load(fr)
            for line in fline:
                for word in line:
                    if word in self.model.vocab:
                        tokens.append(self.model[word])
                    else:
                        tokens.append(np.random.randn(self.vector_size))
        return np.array(tokens, dtype='float32')
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            tokens = self.getTokens(os.path.join(self.dirname, fname))
            yield tokens, fname.split('.')[0]

class SentenceIn(object):
    def __init__(self, dirname, id_tags, tags, GensimModel, vector_size):
        self.dirname = dirname
        self.tags    = np.array(tags, dtype='|S20')
        self.id_tags = id_tags
        self.model   = GensimModel
        self.vector_size = vector_size
        self.stop = stopwords.words('english')
    def getTokens(self, filename):
        tokens = []
        with open(filename, 'r') as fr:
            fline = cPickle.load(fr)
            for line in fline:
                for word in line:
                    if word in self.model.vocab:
                        tokens.append(self.model[word])
                    else:
                        tokens.append(np.random.randn(self.vector_size))
        return np.array(tokens, dtype='float32')
    def getTags(self, idnum):
        labels = np.array([False]*len(self.tags), dtype=bool)
        for tag in self.id_tags[idnum]:
            labels |= (self.tags == tag)
        return labels
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            tokens = self.getTokens(os.path.join(self.dirname, fname))
            labels = self.getTags(fname.split('.')[0])
            yield tokens, labels

def hitRate(samples, classifier, n_largest, n_label, logger):
    predict_model = theano.function(
        inputs = [classifier.input],
        outputs = classifier.rank()
    )
    accuracy = 0.
    denominator = 0.
    for index, sample in enumerate(samples):
        true_label = np.where(sample[1] == True)[0]
        if len(true_label) == 0:
            continue
        logger.info("sample %s" % index)
        try:
            score = predict_model(sample[0])
            pred_nlarge = heapq.nlargest(n_largest, range(len(score)), score.take) 
            intersection = list(set(true_label).intersection(set(pred_nlarge)))
            accuracy += len(intersection)
            denominator += len(true_label)
        except:
            pass
    logger.info("hit rate = %s" % (accuracy / denominator))
    return accuracy / denominator

def accuracy_upperbound(samples, classifier, n_largest, n_label, logger):
    #predict_model = theano.function(
    #    inputs = [classifier.input],
    #    outputs = classifier.rank()
    #)
    accuracy = 0.
    denominator = 0.
    for index, sample in enumerate(samples):
        true_label = np.where(sample[1] == True)[0]
        if len(true_label) == 0:
            continue
        logger.info("sample %s" % index)
        try:
            #score = predict_model(sample[0])
            #pred_nlarge = heapq.nlargest(n_largest, range(len(score)), score.take) 
            #intersection = list(set(true_label).intersection(set(pred_nlarge)))
            accuracy += (float) (min(n_largest, len(true_label))) / (float) (len(true_label))
            denominator += 1. 
        except:
            pass
    logger.info("hit rate upperbound = %s" % (accuracy / denominator))
    return accuracy / denominator

def decoding(samples, classifier, n_largest, logger):
    predict_model = theano.function(
        inputs = [classifier.input],
        outputs = classifier.rank()
    )
    results = {}
    for index, sample in enumerate(samples):
        logger.info("decoding sample %s" % index)
        try:
            score = predict_model(sample[0])
            pred_nlarge = heapq.nlargest(n_largest, range(len(score)), score.take)
        except:
	    pred_n_large = []
        results[sample[1]] = pred_nlarge
    return results

def accuracy(samples, classifier, n_largest, n_label, logger):
    predict_model = theano.function(
        inputs = [classifier.input],
        outputs = classifier.rank()
    )
    accuracy = 0.
    denominator = 0.
    for index, sample in enumerate(samples):
        true_label = np.where(sample[1] == True)[0]
        if len(true_label) == 0:
            continue
        logger.info("sample %s" % index)
        try:
            score = predict_model(sample[0])
            pred_nlarge = heapq.nlargest(n_largest, range(len(score)), score.take) 
            intersection = list(set(true_label).intersection(set(pred_nlarge)))
            accuracy += (float) (len(intersection)) / (float) (len(true_label))
            denominator += 1. 
        except:
	    pass
    logger.info("accuracy = %s" % (accuracy / denominator))
    return accuracy / denominator

def metric(samples, classifier, n_largest, n_label, logger):
    predict_model = theano.function(
        inputs = [classifier.input],
        outputs = classifier.rank()
    )
    true_pos, pred_one, true_one = np.zeros(n_label), np.zeros(n_label), np.zeros(n_label)
    for index, sample in enumerate(samples):
        true_label = np.where(sample[1] == True)[0]
        if len(true_label) == 0:
            continue
        logger.info("sample %s" % index)
        try:
            score = predict_model(sample[0])
            pred_nlarge = heapq.nlargest(n_largest, range(len(score)), score.take) 
            intersection = list(set(true_label).intersection(set(pred_nlarge)))
            true_pos[intersection] += 1.
            pred_one[pred_nlarge]  += 1.
            true_one[true_label]   += 1.
        except:
	    pass
    macro_precision, macro_recall, macro_fscore = macro(true_pos, pred_one, true_one, logger)
    micro_precision, micro_recall, micro_fscore = micro(true_pos, pred_one, true_one, logger)
    return macro_precision, macro_recall, macro_fscore, \
           micro_precision, micro_recall, micro_fscore

def macro(true_pos, pred_one, true_one, logger):
    n_label = len(true_pos)
    omit = len(true_one[true_one == 0])
    pred_one[pred_one == 0] += .1
    true_one[true_one == 0] += .1
    precision = np.mean(true_pos / pred_one) * n_label / (n_label - omit)
    recall    = np.mean(true_pos / true_one) * n_label / (n_label - omit)
    fscore    = np.mean(2 * true_pos / (pred_one + true_one)) * n_label / (n_label - omit)
    logger.info("macro metric:")
    logger.info("precision = %s" % (true_pos/pred_one))
    logger.info("recall    = %s" % (true_pos/true_one))
    logger.info("fscore    = %s" % (2 * true_pos / (pred_one + true_one)))
    logger.info("Precision = %s, Recall = %s, Fscore = %s" % (precision, recall, fscore))
    return precision, recall, fscore

def micro(true_pos, pred_one, true_one, logger):
    label_index = true_one != 0
    precision = np.sum(true_pos[label_index]) / np.sum(pred_one[label_index])
    recall    = np.sum(true_pos[label_index]) / np.sum(true_one[label_index])
    fscore    = 2 * np.sum(true_pos[label_index]) / (np.sum(pred_one[label_index]) + np.sum(true_one[label_index]))
    logger.info("micro metric:")
    logger.info("Precision = %s, Recall = %s, Fscore = %s" % (precision, recall, fscore))
    return precision, recall, fscore

def getIdTag():
    mongoBase = database()
    cursor = mongoBase.getTags()
    id_tag = {}
    for item in cursor:
        uid = item["uid"]
        tags = item["tags"].split(',')
        tags_filter = []
        for tag in tags:
            tags_filter.append(tag.split('(')[0].strip())
        id_tag[uid] = list(set(tags_filter))
    return id_tag

def getTags(filepath):
    tags = []
    tagDistFilename = filepath#"/home/ywanad/Documents/YanWan/GS/reuters/tagDistribution.csv"
    with open(tagDistFilename, "rb") as fr:
        reader = csv.reader(fr)
        for row in reader:
            tags.append(row[0].split('(')[0].strip())
    tags = np.array(tags)
    return tags

def tagLookupTbl(GensimModel, tags, vector_size):
    tag_lookup_tbl = []
    for tag in tags:
        if tag in GensimModel.vocab:
            tag_lookup_tbl.append(GensimModel[tag])
        else:
            tag_lookup_tbl.append(np.random.randn(vector_size))
    tag_lookup_tbl = np.array(tag_lookup_tbl, dtype='float32')
    return tag_lookup_tbl
