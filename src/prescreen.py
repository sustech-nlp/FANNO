# sys
import sys
import os
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import torch
from typing import List, Dict, Any
import numpy as np
from utils import load_jsonlines, save_jsonlines
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sentence_transformers import SentenceTransformer, util
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from tqdm import trange, tqdm
from datetime import datetime
from scipy.spatial.distance import pdist
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import gc
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import time
from loguru import logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,6,7"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Clustering and Visualization Script')
    parser.add_argument('--data_pth', type=str, default="/home/admin/advance-pipeline/data/unlabel/unlabel_data_noadv_good.jsonl", help='The path to the data file')
    parser.add_argument('--model_name', type=str, default='paraphrase-MiniLM-L6-v2', help='The name of the model')
    parser.add_argument('--embedding_pth', type=str, default='data/unlabel_data2.pt', help='The path to the embedding file')
    parser.add_argument('--output_pth', type=str, default='/home/admin/advance-pipeline/data/diversity_unlabel_data_v8.jsonl', help='The path to the output file')
    parser.add_argument('--plot', default=False, action='store_true', help='Whether to plot the clusters')
    parser.add_argument("--input_file", default="/home/admin/ppl_output/llama-2-7b-use_good_select_raw_v1-ft/data/seed.jsonl", help="Path to the input JSON file.")
    parser.add_argument("--result_file", default="/home/admin/advance-pipeline/data_selection/fliter_data", help="Path to the result JSON file.")
    return parser.parse_args()


# method:
# 1. kmeans: this is very slow
def kmeans_cluster(corpus_embeddings: np.ndarray, corpus: List[str], num_clusters: int, type: str) -> List[List[str]]:
    if type == 'mini_batch':
        clustering_model = MiniBatchKMeans(n_clusters=num_clusters)
    else:
        clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
    return clustered_sentences


# 2. agglomerative_clustering: this is very slow
def agglomerative_clustering(corpus_embeddings: np.ndarray, corpus: List[str], distance_threshold: float) -> List[List[str]]:
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(corpus[sentence_id])
    return clustered_sentences


# 3. community_detection: very fast
def community_detection(docs: List[str], corpus_embeddings: np.ndarray, min_community_size: int = 2, threshold: float = 0.7) -> List[Dict[int, str]]:
    import time
    start_time = time.time()
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)
    print("Find clusters in {:.2f} sec".format(time.time() - start_time))
    data_list = []
    for i, cluster in enumerate(clusters):
        for sentence_id in cluster:
            data_list.append({"cluster": i, "sentence": docs[sentence_id]})
    return data_list


# sample from data_list to ensure the diversity of the data, every cluster has at least one sample
def sample_data(data_list: List[Dict[int, str]], sample_num: int) -> List[Dict[int, str]]:
    cluster_num = len(set([d['cluster'] for d in data_list]))
    sample_data = []
    for i in trange(cluster_num):
        cluster_data = [d for d in data_list if d['cluster'] == i]
        if len(cluster_data) > sample_num:
            sample_data.extend(cluster_data[:sample_num])
        else:
            sample_data.extend(cluster_data)
    return sample_data


# plot use umap -> 2D , show different clusters with different colors, and tag the topic of the cluster
def plot_clusters(show_corpus: List[str], corpus_embeddings: np.ndarray, cluster_assignment: List[int], cluster_method: str, reduce_method: str):
    umap_embeddings = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine').fit_transform(corpus_embeddings)
    umap_data = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
    umap_data['cluster'] = cluster_assignment
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x='x', y='y', hue='cluster', palette=sns.color_palette("hls", len(set(cluster_assignment))), data=umap_data, legend="full", alpha=0.3)
    plt.title(f'{cluster_method} clustering of {len(show_corpus)} sentences using {reduce_method} embeddings')
    plt.savefig(f'./{cluster_method}_{reduce_method}.png')
    plt.show()


def encode_documents(model: SentenceTransformer, docs: List[str]) -> torch.Tensor:
    return model.encode(docs, convert_to_tensor=True, show_progress_bar=True, device='cuda', batch_size=256)


class Mistral_Scorer(object):
    
    def __init__(self, model_name_or_path: str = "/data1/zhe/huggingface_model/Mistral-7B-Instruct-v0.2", **kwargs):
        
        self.llm = LLM(model_name_or_path, tokenizer_mode="auto", trust_remote_code=True,max_model_len=25000)
        self.sampling_params = SamplingParams(max_tokens = 2, logprobs = 1000)
        

    def infer_score_batch(self, user_input: List[str]):
            outputs = self.llm.generate(user_input, self.sampling_params)
            try:
                logprobs_list = [outputs[i].outputs[0].logprobs[0] for i in range(len(user_input))]
            except IndexError:
                return [0 for i in range(len(user_input))]
            score_batch = []
            for i in range(len(user_input)):
                score_logits = []
                for k in self.id2score:
                    try:
                        score_logits.append(logprobs_list[i][k])
                    except KeyError:
                        return [0 for i in range(len(user_input))]
                score_logits = np.array(score_logits)
                score_npy = softmax(score_logits, axis=0)
                score_batch.append(np.argmax(score_npy))
            return score_batch
            
    def infer_doc_quality_batch(self, raw_doc: List[str]):
            
            user_input = [self.doc_quality_template.format(doc=i) for i in raw_doc]
            return self.infer_score_batch(user_input)
    
    def infer_advertisement_batch(self, raw_text: List[str]):
            
            user_input = [self.is_advertisement_template.format(text=i) for i in raw_text]
            return self.infer_score_batch(user_input)
        
    def infer_all_batch(self, raw_doc: List[str]):
        doc_result = self.infer_doc_quality_batch(raw_doc)
        
        good_doc = [d for d, r in zip(raw_doc, doc_result) if r == 1]
        
        ad_result = self.infer_advertisement_batch(good_doc)
        

        good_doc = [d for d, r in zip(good_doc, ad_result) if r == 0]
        

        return good_doc


    @property
    def id2score(self):
        
        id2score = {
                28734: "0",
                28740: "1",
                }
        
        return id2score
    
    # Note: You can add more criteria to the prompt if needed.
    # 1 is good, 0 is bad 
    @property
    def doc_quality_template(self):
        DOC_EVALUATE_PROMPT = ("""
        I want you act as an document evaluator. Let's think step by step.
        The goal is to carefully assess the document based on specific criteria and decide whether to accept (1) or reject (0) it.
        Your response should be '0' (reject) if the document meets the criteria or '1' (accept) if it does not, without providing any reasoning and explanation.
        
        Evaluate the document considering the following criteria:

        1. Repetition and Parallelism: Check for excessive repetition or parallel structures that add no new information or insight. Documents should not waste the reader's time with redundant content or overly similar sentence structures that do not contribute to the main theme or understanding.

        2. Relevance of Information: Ensure that the document does not include irrelevant details such as unnecessary mentions of time, locations, names, etc., that do not support or enhance the main theme. The presence of such information can make the text confusing and detract from its overall quality.
        
        <Answer Format>: 1 or 0

        ###Document:
        {doc}
        
        ###Answer: 
        """)
        return DOC_EVALUATE_PROMPT
    


    # this is very important; 0 is good but 1 is not good
    @property
    def is_advertisement_template(self):
        AD_EVALUATE_PROMPT = ("""
        I want you to act as an advertisement evaluator. Let's think step by step.
        The objective is to meticulously inspect the text based on certain characteristics and decide whether it is an advertisement or not.
        Your response should be '1' (yes) if the text is an advertisement, or '0' (no) if it is not, without providing any reasoning and explanation.
        
        Evaluate the text considering these characteristics:

        - Promotional language or sales pitch
        - Mention of product or service benefits
        - Call to action (e.g., "Buy now", "Subscribe")
        - Pricing information or special offers
        - Contact information or links for more details

        <Answer Format>: 1 or 0

        ###Text:
        {text}
        
        ###Answer: 
        """)
        return AD_EVALUATE_PROMPT
    

def main():
    args = parse_arguments()
    
    # step1. 
    scorer = Mistral_Scorer()
    data = load_jsonlines(args.input_file)
    good_doc = scorer.infer_all_batch([record['doc'] for record in data])
    
    # Step2. diversify unlabeled data 
    model = SentenceTransformer(args.model_name)
    embeddings = encode_documents(model, good_doc)
    torch.save(embeddings, args.embedding_pth)
    logging.info(f"Saved embeddings to {args.embedding_pth}")
    clustered_sentences = community_detection(good_doc, embeddings.cpu().numpy(), min_community_size=2, threshold=0.7)
    sampled_data = sample_data(clustered_sentences, sample_num=1)
    save_jsonlines(sampled_data, args.output_pth)
    # logger.info(f"Saved sampled data to {args.output_pth}")

    # if args.plot:
    #     plot_clusters([record['document'] for record in data], embeddings.cpu().numpy(), [record['cluster'] for record in sampled_data], cluster_method='community_detection', reduce_method='UMAP')
    #     logger.info("Plotting complete")


if __name__ == '__main__':
    main()

