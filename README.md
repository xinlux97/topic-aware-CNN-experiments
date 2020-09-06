# topic-aware-CNN-experiments
The implementation in Xin Lu's postgraduate project.

This project is based on the topic-aware convolutional sequence to sequence model, and the original implementation can be found [here](https://github.com/EdinburghNLP/XSum). (Note that we will use "original implementation" to refer to this repository in following sections.) All experiments in this project are based on that implementation. To be more specific, the implementation of topic-aware convolutional sequence to sequence model is in folder [XSum-Topic-ConvS2S](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Topic-ConvS2S). So basically you should follow the original implementation, except some changes specified by us in following sections.

## Experiments on topic modeling
In this section, we use [prodLDA](https://github.com/dallascard/scholar) and [contextualized topic model](https://github.com/MilaNLProc/contextualized-topic-models) to train topic models and get topic representations of words and documents. Note that for the prodLDA we use the implementation of [scholar](https://arxiv.org/abs/1705.09296), which is a topic model based on metadata. But according to the auther, scholar is the same as prodLDA when there is no metadata. So we train a scholar with no metadata, which is equivalent to training a prodLDA.  

In the original implementation of topic-aware CNN model, according to the readme file, they process data into files which are saved to directory named "data-topic-convs2s", and the files are:
```
train.document, train.summary, train.document-lemma and train.doc-topics
validation.document, validation.summary, validation.document-lemma and validation.doc-topics
test.document, test.summary, test.document-lemma and test.doc-topics
```
Here we give those processed files for prodLDA ([here](https://drive.google.com/uc?id=1enJpUe3nCtGMBZoy7oBdJC0t0NwIKb-2)) and contextualized topic model ([here](https://drive.google.com/file/d/1LC3kRxb7-tnqfz7UbP93G2IOAVdavI19/view)) respectively. And the remaining steps, such as training the model and generating the summaries, are the same as the steps in the original implementation. 

## Experiments on attention mechanism 
In this section, we modify a python script file in the original implementation. The file is named "fconv.py", and you can find it in the directory "attention" in this repository. 

In "fconv.py", there is a class named AttentionLayer. You can find several implementations for this class from line 181 to 451. You may choose one type of attention mechanism and comment out the others. You should use this script file to replace the one in the original implementation (i.e. in the directory XSum-Topic-ConvS2S/fairseq/models/). 

## Experiments on word embeddings
In this section, we modify some python script files in the original implementation. 
