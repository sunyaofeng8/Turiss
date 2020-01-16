# Comment Grading for Recommendation System

This is the website of our group project "comment grading for recommendation" for Google ML Camp 2020. This repository contains all the code we used for data cleaning, neural network architecture, and result analysis. You can easily download and deploy it on your local computer for testing ^_^

## Description

The folder './SingleModal' contains 'DataClean.ipynb', 'Simple_LSTM.py', and 'Simple_BERT.py'. These baseline models are only based on comments embedding.

The folder './Multi_Input' contains 'dic_XXX.pkl', which are used for dropping out the rare IDs(user, product, and time) and remapping the rest to numbers.

The folder './MultiModal' contains 'run.py'. You can use it to test the 'BERTMulMod' model and the 'BERTMulModMulTask' model. The usage is 'run.py [--epoch X] [--big 1] [--model choice=['SingleLSTM', 'SingleBERT', 'BERTMulMod', 'BERTMulModMulTask']]'. The logs and result files are also in this folder.

The folder './Stat' contains the statistic-related codes used for the analysis of recommendation system.

The folder './img_folder' contains the images that visualized the performance of our neural networks and the results analysis.

