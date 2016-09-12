import function
from database import database
import sys, os, csv
from collections import defaultdict
import operator
import nltk
import gensim
import cPickle
import cnn
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import heapq

def train(samples, tag, tag_lookup_tbl, vector_size, train_length, n_largest, model_direct):
    """ main cnn training model """
    logger.info("start building model ...")
    x = T.matrix('x')
    y = T.ivector('y')
    initial_learning_rate = 2e-3
    learning_rate = theano.shared(np.asarray(initial_learning_rate, \
                    dtype = theano.config.floatX), borrow=True)
    n_label = len(tag)
    word_vec_dim = vector_size
    conv_size = 5
    hidden_dim = 1000
    embed_dim = 200
    
    classifier = cnn.CNN(x, tag_lookup_tbl, \
           word_vec_dim, conv_size, hidden_dim, embed_dim)
    cost = (
       classifier.loss(y)
    )

    
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    train_model = theano.function(
        [x, y],
        cost,
        updates = updates
    )

    model_score = theano.function(
        inputs = [x], 
        outputs = classifier.rank()
    )
    model_lookup = theano.function(
        inputs = [x], 
        outputs = classifier.lookup(),
        on_unused_input = 'ignore'
    )
    
    decay_learning_rate = theano.function(inputs = [], outputs = learning_rate, \
              updates = {learning_rate : learning_rate * 0.65})

    if not os.path.exists(model_direct):
        os.makedirs(model_direct)
    best_valid_loss = np.inf
    epoch, n_epochs = 0, 5
    logger.info("tag_hidden_layer.shape = %s, %s" % (classifier.tag_hidden_layer.get_value(borrow=True).shape[0], classifier.tag_hidden_layer.get_value(borrow=True).shape[1]))
    logger.info("start training, n_epochs = %s" % n_epochs)
    while (epoch < n_epochs):
        epoch = epoch + 1
        minibatch_avg_cost = 0.
        tags_trainNum = np.zeros(len(tags))
        for index, sample in enumerate(samples):
            #logger.info(sample[1])
            true_label = np.where(sample[1] == True)[0]
            if len(true_label) == 0:
                logger.info("omit index %s" % index)
                continue
            tags_trainNum[true_label] += 1
            flag = False
            overflow_label_index = 0
            for label_index in true_label:
                if tags_trainNum[label_index] > 10000:
                    flag = True
                    overflow_label_index = label_index
                    break
            if flag == True:
                logger.info("# of label %s over 10000!" % (tags[overflow_label_index]))
                continue
            try:
                minibatch_avg_cost += train_model(sample[0], np.array(true_label, dtype='int32'))
            except:
                pass
            if (index % 40 ) == 0:
                try:
                    percentage = float(index) / float(train_length) * 100.
                    logger.info("percentage = %s" % percentage)
                    logger.info("index = %s, true_label = %s" % (index, true_label))
                    score = model_score(sample[0])
                    pred_nlarge = heapq.nlargest(n_largest, range(len(score)), score.take)
                    logger.info("index = %s, pred_label = %s" % (index, pred_nlarge))
                    #logger.info("index = %s, score = %s" % (index, score))
                    logger.info("index = %s, cost = %s" % (index, minibatch_avg_cost))
                    #logger.info("index = %s, conv_layer = %s" % (index, classifier.conv_layer.get_value(borrow=True)))
                    model_save = os.path.join(model_direct, "best_model.pkl")
                    with open(model_save, "w") as f:
                        cPickle.dump(classifier, f)
                except:
                    pass
        logger.info("loss = %s, learning rate = %s" % (minibatch_avg_cost, learning_rate.get_value(borrow=True)))
        logger.info("training evaluation:")
        macro_precision, macro_recall, macro_fscore, \
        micro_precision, micro_recall, micro_fscore = function.metric(samples, classifier, n_largest, n_label, logger)
        if epoch < 20:
            if epoch % 10 == 0:
                new_learning_rate = decay_learning_rate()
        else:
            if epoch % 5 == 0:
                new_learning_rate = decay_learning_rate()
        if macro_fscore < best_valid_loss:
            best_valid_loss = macro_fscore
            model_save = os.path.join(model_direct, "best_model.pkl")
            with open(model_save, "w") as f:
                cPickle.dump(classifier, f)


def predict(samples, tag_lookup_tbl, n_largest, model_direct, n_label, logger):
    model_read = os.path.join(model_direct, "best_model.pkl")
    with open(model_read, 'r') as f:
        classifier = cPickle.load(f)
    function.metric(samples, classifier, n_largest, n_label, logger)

if __name__ == '__main__':
    logger = function.setLog('../../Log/modelRun.log')
    try:
        trainTokenDir = sys.argv[1]
        gensimPath    = sys.argv[2]
        n_largest     = (int) (sys.argv[3])
    except:
        logger.info("Input Error! Lack arguments!")

    vector_size = 500
    id_tag = function.getIdTag()
    tags = function.getTags("../../../csvFiles/tagDistribution.csv")
    logger.info("# of tags = %s" %len(tags))
    GensimModel = gensim.models.Word2Vec.load(gensimPath)
    logger.info("# of words in the dictionary = %s" % len(GensimModel.vocab))
    tag_lookup_tbl = function.tagLookupTbl(GensimModel, tags, vector_size)
    train_length = len(os.listdir(trainTokenDir))
    model_direct = "../model"
    samples = function.SentenceIn(trainTokenDir, id_tag, tags, GensimModel, vector_size)
    train(samples, tags, tag_lookup_tbl, vector_size, train_length, n_largest, model_direct)
