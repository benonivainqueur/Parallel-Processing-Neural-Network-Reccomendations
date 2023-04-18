# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:49:06 2023

@author: Elifnur
"""

import sys
import argparse
import os
import findspark
findspark.init()
import pandas as pd
from time import time
import csv
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.feature import PCA
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, HashingTF, IDF
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
import matplotlib.pyplot as plt


def createtfidf(df):
    # Tokenize the "summary" column
    tokenizer = Tokenizer(inputCol="summary", outputCol="tokens") 
    df = tokenizer.transform(df) 
    # Generate term frequency vectors using hashing trick
    hashingTF = HashingTF(inputCol="tokens", outputCol="tf", numFeatures = 1000) 
    df = hashingTF.transform(df) 
    # Calculate IDF values
    idf = IDF(inputCol="tf", outputCol="tfidf") 
    idfModel = idf.fit(df) 
    df = idfModel.transform(df) 
    # Select only the "summary" and "tfidf" columns
    df = df.select(col("category"),col("summary"), col("tfidf")) 
    # Show the resulting DataFrame
    df.show()    
    return df

def RandomForest_tfidf(df):
    rf = RandomForestClassifier(featuresCol="tfidf", labelCol="category") 
    # Define the hyperparameter grid for cross-validation
    num_trees = [10, 20, 30]
    max_depths = [5, 10, 15]
    param_grid = ParamGridBuilder().addGrid(rf.numTrees, num_trees).addGrid(rf.maxDepth, max_depths).build()
    # Create the cross-validator
    evaluator = MulticlassClassificationEvaluator(labelCol="category", predictionCol="prediction", metricName="accuracy") 
    cross_validator = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5) 
    # Split the data into training and testing sets
    (train_df, test_df) = df.randomSplit([0.8, 0.2], seed=42) 
    # Fit the cross-validator to the training data
    cv_model = cross_validator.fit(train_df) 
    # Get the best model with the best hyperparameters
    best_model = cv_model.bestModel
    # Use the best model to make predictions on the test data
    predictions = best_model.transform(test_df) 
    # Evaluate the accuracy of the predictions
    accuracy = evaluator.evaluate(predictions) 
    print("Best Model Accuracy: ", accuracy) 
    # Extract the cross-validation results
    cv_results = cv_model.avgMetrics
    # Create a meshgrid for plotting
    num_trees, max_depths = np.meshgrid(num_trees, max_depths) 
    cv_results = np.array(cv_results).reshape(num_trees.shape) 
    # Plot the cross-validation results
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.plot_surface(num_trees, max_depths, cv_results) 
    ax.set_xlabel('Num Trees') 
    ax.set_ylabel('Max Depth') 
    ax.set_zlabel('Accuracy') 
    ax.set_title('Cross-Validation Results') 
    plt.show() 
    # You can also get the best hyperparameters from the best model
    best_num_trees = best_model.getNumTrees
    print(best_num_trees)
    
    
    
    

if __name__ == "__main__":   
    
    parser = argparse.ArgumentParser(description = 'Summaries Loading',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('sum_path', default = "./data/classification_dataset.csv", help="Youtube Transcripts")
    parser.add_argument('--N',type=int,default=20,help="Number of partitions to be used in RDDs containing word counts.")
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    args = parser.parse_args()

    spark = SparkSession.builder.appName('Spark DataFrame').getOrCreate()
    # Read the CSV file into a Spark DataFrame
    
    
    df = pd.read_csv(args.sum_path)
    df['category'] = pd.Categorical(df['category']).codes
    df = df[["title","summary", "category"]]
    # print(df.nunique(), "\n")
    rdd=spark.createDataFrame(df).repartition(args.N) 
    
    # Category string to numbers
    rdd = rdd.select("title","summary",F.col('category').cast(FloatType()).alias('category'))
    
    start = time()
    # create tfidf
    rdd_tfidf = createtfidf(rdd)
    cv_results = RandomForest_tfidf(rdd_tfidf)
    end = time()
    print('Total execution time:',str(end-start)+'sec')
    
