"""
Step 12 in collecting SO sents
"""
import os
import sys
import glob
import json
import time
sys.path.append('/app/scripts/')

from utils.es_client import ESClient
from config.config import config

def get_threads():
    thread_dir = "/app/search_result/"
    big_thread_list = []
    for file in glob.glob(f"{thread_dir}/and/*"):
        with open(file, "r") as fp:
            small_list_threads = json.load(fp)
        big_thread_list += small_list_threads
    for file in glob.glob(f"{thread_dir}/or/*"):
        with open(file, "r") as fp:
            small_list_threads = json.load(fp)
        big_thread_list += small_list_threads
    
    big_thread_list = list(set(big_thread_list))
    return big_thread_list
    

def main():
    es_client = ESClient()
    list_threads = get_threads()
    print(len(list_threads))
    for thread_id in list_threads:
        while True:
            try:
                path = str(thread_id)[:2]
                sub_folder = f"/app/additional_data/tpl_threads/{path}/"
                os.makedirs(sub_folder, exist_ok=True)
                thread_content_json = es_client.get(thread_id, config.es.index_name)
                with open(f"{sub_folder}/{thread_id}.json", "w+") as fp:
                    json.dump(thread_content_json, fp, indent=2)
                break
            except Exception as e:
                print(e)
                time.sleep(3)


    # es_client.get()
    pass

if __name__ == "__main__":
    main()