from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors
import numpy as np
import time

# Define function to calculate cosine similarity score
def cosine_similarity(query, embedding):
    dot_product = np.dot(query, embedding)
    norm_query = np.linalg.norm(query)
    norm_embedding = np.linalg.norm(embedding)
    return dot_product / (norm_query * norm_embedding)

# Define query and query embedding as RDDs
conf = SparkConf().setAppName("Cosine Similarity")
sc = SparkContext.getOrCreate(conf)
query = sc.parallelize([np.random.rand(786)])
embeddings = sc.parallelize(np.random.rand(1500, 786))

# Define list of number of partitions to try
num_partitions_list = [20]

# Loop over number of partitions and calculate cosine similarity scores
for num_partitions in num_partitions_list:
    # Repartition the embedding RDD with num_partitions partitions
    embeddings_partitioned = embeddings.repartition(num_partitions)

    # Calculate cosine similarity score using RDD zipWithIndex and map functions
    start_time = time.time()
    scores = embeddings_partitioned.zipWithIndex().map(lambda x: (x[1], cosine_similarity(query.collect()[0], x[0]))).collect()
    scores.sort(key=lambda x: x[0])
    scores = [x[1] for x in scores]
    end_time = time.time()

    print(f"Time taken with {num_partitions} partitions:", end_time - start_time, "seconds")
    print("---------------------------------------------------------------------------")

# Stop SparkContext
sc.stop()
