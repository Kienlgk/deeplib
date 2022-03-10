"""
Step 3 in collecting SO sents
"""
import os
import sys
import glob
import json
import re
import time
import nltk
# pip install nltk
import shutil
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from functools import reduce

sys.path.append('/app/scripts/')

from utils.es_client import ESClient
from config.config import config

def make_parent_dirs_recursively(_dir):
    from functools import reduce
    def concat_and_make(path_1, path_2):
        concatnated_path = path_1+os.sep+path_2
        os.makedirs(concatnated_path, exist_ok=True)
        return concatnated_path

    broken_dir = _dir.split(os.sep)[:-1]

    reduce(concat_and_make, broken_dir)

def get_threads():
    tpl_list = get_tpl_name()
    thread_dir = "/app/search_result/"
    out_dir = "/app/search_result/output/"
    os.makedirs(out_dir, exist_ok=True)
    big_thread_list = []
    for file in glob.glob(f"{thread_dir}/or/*")[1:]:
        saved_output_sents_for_tpl = {}
        print(file)
        tpl_id = int(file.split(os.sep)[-1][:-5])
        tpl_name = tpl_list[tpl_id-1] # this was bug as it was tpl_list[tpl_id]
        out_file = out_dir+"new_"+os.sep.join(file.split(os.sep)[-3:-1])[1:] + os.sep + str(int(file.split(os.sep)[-1].split(".")[0])-1)+".json"
        if os.path.exists(out_file):
            continue
        make_parent_dirs_recursively(out_file)

        with open(file, "r") as fp, open(f"{out_file}", "w+") as wp:
            small_list_threads = json.load(fp)
            for thread_id in small_list_threads:
                sents = extract_sentences(tpl_name, thread_id)
                if len(sents) > 0:
                    saved_output_sents_for_tpl[thread_id] = sents
                saving_file = {'tpl_name': tpl_name, 'threads': saved_output_sents_for_tpl}
            json.dump(saving_file, wp, indent=2)

        # big_thread_list += small_list_threads
    # for file in glob.glob(f"{thread_dir}/or/*"):
    #     with open(file, "r") as fp:
    #         small_list_threads = json.load(fp)
    #     big_thread_list += small_list_threads
    
    # big_thread_list = list(set(big_thread_list))

def get_threads_2(): # when /app/search_result/or/ is deleted by accident
    tpl_list = get_tpl_name()
    thread_dir = "/app/search_result/"
    out_dir_old = "/app/search_result/output/"
    out_dir_new = "/app/search_result/output_new/"

    os.makedirs(out_dir_new, exist_ok=True)
    big_thread_list = []

    for file in glob.glob(f"{out_dir_old}/or/*"):
        saved_output_sents_for_tpl = {}
        print(file)
        tpl_id = int(file.split(os.sep)[-1][:-5])
        tpl_name = tpl_list[tpl_id-1] 
        out_file = out_dir_new+os.sep.join(file.split(os.sep)[-3:])
        make_parent_dirs_recursively(out_file)

        with open(file, "r") as fp, open(f"{out_file}", "w+") as wp:
            small_list_threads = json.load(fp)
            for thread_id in small_list_threads.keys():
                sents = extract_sentences(tpl_name, thread_id)
                saved_output_sents_for_tpl[thread_id] = sents
            json.dump(saved_output_sents_for_tpl, wp, indent=2)

def get_tpl_name():
    def remove_stop_word_from_query(query):
        text_tokens = word_tokenize(query)
        tokens_without_sw = [word for word in text_tokens if (word.lower() not in stopwords.words()) and (word.lower() not in ['sdk', 'plugin'])]
        return " ".join(tokens_without_sw)

    with open("/app/additional_data/tpl_list.json", "r") as fp:
        libs = json.load(fp)
    list_tpls = []
    for lib_i, lib in enumerate(libs):    
        tpl_name = remove_stop_word_from_query(lib)
        list_tpls.append(tpl_name)
    return list_tpls
    
def extract_sentences(tpl_name, thread_id):
    thread_subpath = str(thread_id)[:2]
    tpl_parts = tpl_name.split(" ")
    return_sentences = []
    with open(f"/app/additional_data/tpl_threads/{thread_subpath}/{thread_id}.json", "r") as fp:
        thread_json = json.load(fp)
        thread_json['text'].replace("#CODE", "")
        thread_json['text'].replace("\\n", " ")
        sentences = re.split("\.\s+", thread_json['text'])
        for sent in sentences:
            # for split_sent in sent.split("\\\\n\\\\n"):
            for split_sent in sent.split("\\n\\n"):
                for tpart in tpl_parts:
                    if tpart.isdecimal():
                        continue
                    # sent_words = sent.split()
                    sent_words = re.split('[^a-zA-Z]', split_sent)
                    sent_words = [word.lower() for word in sent_words]
                    if tpart.lower() in sent_words:
                        return_sentences.append(split_sent.strip())
                        break
    return return_sentences

def filter_received_threads():
    # need_filtering_dir = "/app/search_result/output/or/"
    # filtered_dir = "/app/search_result/output/or_filtered/"
    need_filtering_dir = "/app/search_result/output/new_or/"
    filtered_dir = "/app/search_result/output/new_or_filtered/"
    paths = glob.glob(f"{need_filtering_dir}/**.json")
    for path in paths:
        file_name = path.split(os.sep)[-1]
        with open(path, "r") as fp:
            content = json.load(fp)
        content_keys = list(content.keys())
        for key in content_keys:
            if len(content[key]) == 0:
                content.pop(key)
        out_file_name = f"{filtered_dir}/{file_name}"
        if len(list(content.keys())) == 0:
            if os.path.isfile(out_file_name):
                os.remove(out_file_name)
            continue
        
        make_parent_dirs_recursively(out_file_name)

        with open(f"{out_file_name}", "w+") as fp:
            json.dump(content, fp, indent=2)


def main():
    get_threads() # Get the SO sentences from the thread id list of each TPL
    # filter_received_threads() # Filter empty files and empty threads

if __name__ == "__main__":
    main()