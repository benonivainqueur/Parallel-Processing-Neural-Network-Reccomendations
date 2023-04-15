# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:55:15 2023

@author: Elifnur
"""
import sys
import argparse
import os
import findspark
findspark.init()
import pandas as pd
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim



if __name__ == "__main__":   
    
    parser = argparse.ArgumentParser(description = 'Data Loading',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('data_path', default = "./data/ytranscripts.csv", help="Youtube Transcripts")
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    args = parser.parse_args()

    spark = SparkSession.builder.appName('Spark DataFrame').getOrCreate()
    rdd_input = spark.read.option('header',True).csv(args.data_path)

    # Select columns
    rdd_input_select = rdd_input.select('title','transcript', 'playlist_name')
    
    # Remove text inside  "[...]"
    rdd_input_remove = rdd_input_select.withColumn('transcript',(regexp_replace('transcript', "\\[.*\\]", "")).alias('transcript'))
    
    # Remove non alphabetic characters, convert lowercase and trim blank space beginning of the column
    rdd_input_clean = rdd_input_remove.withColumn('title',(trim(lower(regexp_replace('title', "[^a-zA-Z\\s]", ""))).alias('title')))
    rdd_input_clean = rdd_input_clean.withColumn('transcript',(trim(lower(regexp_replace('transcript', "[^a-zA-Z\\s]", ""))).alias('transcript')))
    rdd_input_clean = rdd_input_clean.withColumn('playlist_name',(trim(lower(regexp_replace('playlist_name', "[^a-zA-Z\\s]", ""))).alias('playlist_name')))
    
    
    # # Save dataframe
    # output_csv = rdd_input_clean.toPandas()
    # output_csv.to_csv(args.data_path.split('.csv')[0] + '_preprocessed.csv') 
    