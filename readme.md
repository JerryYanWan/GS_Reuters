
WorkFlow to process the reuters data
====================================


1. data processing
------------------

*  downloading articles from website  (crawl.py) 
*  storing them into reuters/data directory
*  tokenize all the files into reuters/data_tokens directory (txt2token.py)
*  splitting files into ./train and ./test under the reuters/corpus directory (trainTestSetup.py)

2. model preparing
------------------

*  running the cnn model in the neuralNetwork directory, which are stored in the following structure:
```
neuralNetwork/
|-- Log
|-- cnn
|   |-- model
|   |   `-- best_model.pkl
|   |-- test
|   |   |-- cnn.py
|   |   |-- test.py
|   |   `-- test.sh
|   `-- train
|       |-- cnn.py
|       |-- run.sh
|       `-- train.py
|-- gensimModel
|   `-- gensimModel
`-- trainGensimModel
    `-- trainGensimModel.py
```

*   training a word2vec model with gensim library in neuralNetwork/trainGensimModel directory. The gensim model will be stored in the gensimModel directory
*   training and testing a cnn model in neuralNetwork/cnn directory, both will read necessary files from Functions and csvFiles directory


3. training the model
---------------------
*    please prepare all the tokenized files into ONE directory, and run the run.sh script (with path modified). The model will read in the files one by one and train the model

*    the run.sh has three variables, the training path, the word2vec model path, and the number of tags that model chooses to output. The model will be saved into ``neuralNetwork/cnn/model/`` directory.

4. testing the model
--------------------
*    please prepare all the tokenzie files into ONE directory, and run the test.sh script (with path modified). The model will read in the files one by one and give the probability over each tags.

*    The ``test.sh`` has three variables, the testing path, the word2vec model path, and the number of tags that model chooses to output


5. reutersMongoDB
-----------------
   
*   It is a backup file from the mongodb database, storing all the reuters date, uid, title, tags, ticker, and url.


6. sample code
--------------
*   enter the toyExample directory, modify this path into the test.sh under the testing directory the sampleInput contains the sample files that are loaded into the cnn, the output will be a csv file format, with the uid as the key and predicted tags as the values, example output will be as following part:

<div>
    <table border="0">
        <tr>
<th>98b93738da70a06a001404d49e92c7de</th>
<th>'Basic Materials' 'Iron and Steel' 'Metals and Mining' 'Food Processing' 'Consumer Electronics'</th>
</tr>
<tr>
<th>98cca45bc7aa697a8f6e701ae6b87b33</th>
<th>'Oil and Gas' 'Iron and Steel' 'Food Processing' 'Basic Materials' 'Food and Tobacco'</th>
</tr>
</table>
</div>
