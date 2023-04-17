import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5Model,T5ForConditionalGeneration, pipeline
import math
import multiprocessing as mp
import time
from tqdm import tqdm
import random
mp.set_start_method('spawn', force=True)

def create_summaries(transcript,summarizer,tokenizer,time_store,MAX_LENGTH=500):
    full_tokenize = tokenizer(transcript).input_ids
    length_sample = len(full_tokenize)
    chunks =math.ceil(length_sample/MAX_LENGTH)
    chunked_sample = np.array_split(full_tokenize,chunks)
    sample_text=""
    srz_min_length = 1    # to be considered
    srz_max_length = int(MAX_LENGTH/chunks)
    start = time.time()
    summary = summarizer(tokenizer.batch_decode(chunked_sample), min_length=srz_min_length, max_length=srz_max_length, batch_size=len(chunked_sample))
    for s in summary:
        sample_text+=s['summary_text']
    end = time.time()
    time_store.append(end-start)
    return sample_text


def create_summary(df,summarizer, tokenizer,q):
    
    store=[]
    summary = []
    df = df.reset_index()
    for i in tqdm(range(df.shape[0])):
        t = df['transcript'][i]
        s = create_summaries(t,summarizer,tokenizer, store)
        summary.append(s)
    df['summary'] = summary
    df['time_taken'] = store
    q.put(df)


if __name__ == '__main__':
    df = pd.read_csv('./data/ytdataset_preprocessed.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    df = df[df['transcript'].notna()]
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    summarizer_gpu1 = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=0)
    summarizer_gpu2 = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=1)
    
    import time
    start = time.time()
    q1 = mp.Queue()
    p1 = mp.Process(target=create_summary, args=(df.iloc[:2500,:],summarizer_gpu1, tokenizer,q1))
    p1.start()

    q2 = mp.Queue()
    p2 = mp.Process(target=create_summary, args=(df.iloc[2500:,:],summarizer_gpu2, tokenizer, q2))
    p2.start()

    summaries1 = q1.get()
    summaries2 = q2.get()

    p1.join()
    p2.join()
    print("time taken: ", time.time()-start)
    print("store data")
    summaries1.to_csv('summar1.csv')
    summaries2.to_csv('summar2.csv')
   
    