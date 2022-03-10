import os
import json
import glob
import csv
import time

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering


# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 12


model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):

    print("Sentence:", len(sentence))
    print(sentence[0])
    print("Embedding:", len(embedding))
    print("")

def init_sentence_transformer_model(pretrained_model_name='all-MiniLM-L6-v2', max_seq_length=256):
    model = SentenceTransformer(pretrained_model_name)
    model.max_seq_length = max_seq_length
    print("Max Sequence Length:", model.max_seq_length)
    return model

def clustering(corpus_sentences, model):
    start = time.time()
    corpus_embeddings = model.encode(corpus_sentences, batch_size=32, show_progress_bar=True, convert_to_tensor=True)

    corpus_embeddings = corpus_embeddings.cpu().detach().numpy()
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    print(corpus_embeddings.shape)

    # Perform kmean clustering without cluster number
    # following https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/agglomerative.py
    start = time.start() 
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus_sentences[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        print(cluster)
        print("")
    print("Running time: {:.2f}s".format(time.time()-start))

def read_so_sents_for_tpl():
    with open("/app/additional_data/tpl_list.json", "r") as fp:
        tpls = json.load(fp)

    # for tpl in tpls:
    #     pass
    
    model = init_sentence_transformer_model()

    for file_path in glob.glob("/app/search_result/output/new_or/*.json"):
        file_tpl_index = int(file_path.split(os.sep)[-1].split(".")[0])
        tpl_name = tpls[file_tpl_index]
        with open(file_path, "r") as sents_fp:
            loaded_json = json.load(sents_fp)['threads']
            tpl_name = loaded_json['tpl_name']
            tpl_threads = loaded_json['threads']
        all_sents = []
        for sents in tpl_threads.values():
            all_sents += sents

        clustering(all_sents, model)
        


def main():
    read_so_sents_for_tpl()

if __name__ == "__main__":
    main()