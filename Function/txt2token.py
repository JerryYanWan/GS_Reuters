import sys, os, re
import nltk
import cPickle
import logging
import threading
import csv

import function


class myThread(threading.Thread):
    def __init__(self, threadName, start, end, filenames, txtDir, tokenDir, logger):
        threading.Thread.__init__(self)
        self.name = threadName
        self.file_begin = start
        self.file_end   = end
        self.filenames  = filenames
        self.txtDir     = txtDir
        self.tokenDir   = tokenDir
        self.logger     = logger
    def run(self):
        self.logger.info("start %s" % self.name)
        tokenizer(self.filenames[self.file_begin:self.file_end], self.txtDir, self.tokenDir, self.logger)
        self.logger.info("Existing %s" % self.name)

def tokenizer(files, txtDir, tokenDir, logger):
    for i, filename in enumerate(files):
        if os.path.exists(os.path.join(tokenDir, filename + ".pkl")):
            continue
        try:
            with open(os.path.join(txtDir, filename)) as fr:
                text = fr.readlines()
            #logger.info(text)
            if len(text) == 0:
                logger.info("file length zero: %s" % filename)
            else:
                token_file = []
                for line in text:
                    try:
                        tokens = list(nltk.word_tokenize(line.decode("utf-8")))
                        token_file += tokens
                    except:
                        logger.info("tokenizer unsuccessfully")
                        pass
                #logger.info(token_file)
                fw = open(os.path.join(tokenDir, filename+".pkl"), "w")
                cPickle.dump(token_file, fw)
                fw.close()
                logger.info("processing %s" % i)
        except:
            logger.info("exception %s" % filename)
            pass


if __name__ == '__main__':

    logger = function.setLog("log/txt2token.log")

    try:
        txtDir = sys.argv[1]
        tokenDir = sys.argv[2]
    except:
        logger.info("not enough argument!")

    files = os.listdir(txtDir)
    nonEnglish, numAll = [], len(files)
    numNonEnglish = 0

    numThread = 10
    step = (int) (numAll / numThread)
    thread_list = []

    for j in xrange(numThread):
        thread_list.append(myThread("Thread-%s"%j, j*step, (j+1)*step, files, txtDir, tokenDir, logger))
    for item in thread_list:
        item.start()

    logger.info("Exiting Main Thread")
