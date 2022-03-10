"""
    Search SO threads mentioning the TPLs in their sentences.
    Step 1 in collecting so sents
"""
import os
import gc
import time
# pip install nltk
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

from multiprocessing import Pool
from pprint import pformat

import sys
sys.path.append("/app/scripts/")
from utils.es_client import ESClient
from config.config import config

def remove_stop_word_from_query(query):
    text_tokens = word_tokenize(query)
    tokens_without_sw = [word for word in text_tokens if (word.lower() not in stopwords.words()) and (word.lower() not in ['sdk', 'plugin'])]
    return " ".join(tokens_without_sw)

def main():
    pass
    import json
    # searching
    with open("/app/additional_data/tpl_list.json", "r") as fp:
        libs = json.load(fp)
    INDEX_NAME = config.es.index_name
    es_client = ESClient()
    print(len(libs))
    print(INDEX_NAME)
    for lib_i, lib in enumerate(libs):    
        query = remove_stop_word_from_query(lib)
        # print(f"{lib_i+1}: {lib}")
        # print(query)
        # thread_results, results_len = es_client.query(index_name=INDEX_NAME, search_string=query, op="and", field="search")
        # # print(thread_results)
        # predicted_ids = []
        # for res in thread_results:
        #     if res['_id'] not in predicted_ids:
        #         predicted_ids.append(res['_id'])


        # with open(f"/app/search_result/and/{lib_i+1}.json", "w+") as fp:
        #     json.dump(predicted_ids, fp, indent=2)
        # time.sleep(1)

        thread_results, results_len = es_client.query(index_name=INDEX_NAME, search_string=query, op="or", field="search")
        predicted_ids = []
        for res in thread_results:
            if res['_id'] not in predicted_ids:
                predicted_ids.append(res['_id'])
        os.makedirs("/app/search_result/or/", exist_ok=True)
        with open(f"/app/search_result/or/{lib_i+1}.json", "w+") as fp:
            json.dump(predicted_ids, fp, indent=2)
        time.sleep(1)
    

if __name__ == "__main__":
    main()