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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim



if __name__ == "__main__":   
    
    parser = argparse.ArgumentParser(description = 'Data Loading',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('data_path', default = "./data/YouTube_transcripts_Kaggle.csv", help="Youtube Transcripts")
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    args = parser.parse_args()

    spark = SparkSession.builder.appName('Spark DataFrame').getOrCreate()
    
    
    df = pd.read_csv(args.data_path)
    print(df.nunique(), "\n")
    rdd_input=spark.createDataFrame(df) 
    rdd_input.show(5)

    # Select columns
    rdd_input_select = rdd_input.select('title','transcript', 'playlist_name')
    rdd_input.show(5)
    
    # # Remove text inside  "[...]"
    # rdd_input_remove = rdd_input_select.withColumn('transcript',(regexp_replace('transcript', "\\[.*\\]", "")).alias('transcript'))
    
    # Remove non alphabetic characters, convert lowercase and trim blank space beginning of the column
    rdd_input_clean = rdd_input_select.withColumn('transcript',(trim(lower(regexp_replace('transcript', "[^a-zA-Z\\s]", ""))).alias('transcript')))
    rdd_input_clean.show(5)
    
    # Save dataframe
    output_csv = rdd_input_clean.toPandas()
    print(output_csv.nunique(), "\n")
    output_csv.to_csv(args.data_path.split('.csv')[0] + '_preprocessed.csv') 
    
    
    # https://www.kaggle.com/code/tientd95/advanced-pyspark-for-exploratory-data-analysis#3.-Detect-missing-values-and-abnormal-zeroes-
    # is used for data analysis
    
    
    # Visualize Preprocessed data
    print('Input Data overview \n')
    rdd_input_clean.printSchema()
    print('Columns overview \n')
    print(pd.DataFrame(rdd_input_clean.dtypes, columns = ['Column Name','Data type']))
    
    # Check missing values
    string_columns = ['title','transcript', 'playlist_name']
    numeric_columns = []
    array_columns = []
    missing_values = {} 
    for index, column in enumerate(rdd_input_clean.columns):
        if column in string_columns:    # check string columns with None and Null values

            missing_count = rdd_input_clean.filter(col(column).isNull()).count()
            missing_values.update({column:missing_count})
    missing_df = pd.DataFrame.from_dict([missing_values])
    print("\nCheck if there is any missing values: \n",missing_df)
     
    playlist_name_df = rdd_input.select( rdd_input.playlist_name,  rdd_input.title) \
    .distinct() \
    .groupBy(rdd_input.playlist_name) \
    .count() \
    .orderBy("count", ascending=False)
    
    # print(playlist_name_df.collect()[:5])

    # Top 5 playlist names
    highest_playlist_name_df = playlist_name_df.limit(5).toPandas()
    # Rename column name : 'count' --> Title count
    highest_playlist_name_df.rename(columns = {'count':'Title count'}, inplace = True)
    # Caculate the total unique titles, we will this result to compute percentage later
    total_titles = playlist_name_df.groupBy().sum().collect()[0][0]    
    print("\nPlaylist count (5 top): \n", highest_playlist_name_df)  
    
    
    # Plotting
    
    highest_playlist_name_df_renamed = highest_playlist_name_df
    # Compute the percentage of top 5 playlist names / total titles
    highest_playlist_name_df_renamed['percentage'] = highest_playlist_name_df_renamed['Title count'] \
        / total_titles * 100
    
    # We assign the rest of users belong to another specific group that we call 'others'
    others = {
          'playlist_name': 'others'
        , 'Title count': total_titles - sum(highest_playlist_name_df_renamed['Title count'])
        , 'percentage': 100 - sum(highest_playlist_name_df_renamed['percentage'])
    }
    
    highest_playlist_name_df_renamed = highest_playlist_name_df_renamed.append(
        others, ignore_index=True
    )
    print('Top 5 playlist names:')
    highest_playlist_name_df_renamed
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=plt.figaspect(0.5))
    plt.xticks(rotation=90)
    plot0 =   axs.bar(x=highest_playlist_name_df_renamed['playlist_name']
                         , height=highest_playlist_name_df_renamed['Title count'])
    title0 =  axs.set_title('Title count', fontsize = 'small')
    ylabel0 = axs.set_ylabel('Title count', fontsize = 'small') 
    
    text = fig.text(0.5, 1.02, 'Top 5 playlist names', ha='center', va='top', transform=fig.transFigure)

    

    

    