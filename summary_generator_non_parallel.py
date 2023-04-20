import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5Model,T5ForConditionalGeneration, pipeline
import math
import time
import random


def create_summaries(transcript,device,time_store,MAX_LENGTH=500):
    full_tokenize = tokenizer(transcript).input_ids
    length_sample = len(full_tokenize)
    chunks =math.ceil(length_sample/MAX_LENGTH)
    chunked_sample = np.array_split(full_tokenize,chunks)
    sample_text=""
    srz_min_length = 1    # to be considered
    srz_max_length = int(MAX_LENGTH/chunks)
    if (device==0):
        summarizer = summarizer_gpu
    else:
        summarizer = summarizer_cpu
    start = time.time()
    for j in chunked_sample:
        input_text = tokenizer.decode(j)
        sample_text+=summarizer(input_text, min_length=srz_min_length, max_length=srz_max_length)[0]['summary_text']
    end = time.time()
    time_store.append(end-start)
    return sample_text


def create_summary(df, device):
    store=[]
    df['summary'] = df['transcript'].apply(lambda x:create_summaries(x,device,store))
    df['time_taken'] = store
    return df


if __name__ == '__main__':
    df = pd.read_csv('ytdataset_preprocessed.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    df = df[df['transcript'].notna()]
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    global summarizer_gpu = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=0)
    global summarizer_cpu = pipeline("summarization", model="t5-base", tokenizer="t5-base")
    # set device. turn this to None for CPU 
    device = 0
    summarized_df = create_summary(df,device)
    summarized_df.to_csv('summarized.csv')
    