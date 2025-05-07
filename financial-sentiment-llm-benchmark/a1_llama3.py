import hashlib
from itertools import accumulate
import json
import os
import time
from datetime import datetime
from typing import List
from urllib import response

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from a1_utils import *


def check_requst_result(user_id: str, task_id: str):
    try:
        res = requests.get(f"http://csc401:8000/check_request_result?user_id={user_id}&task_id={task_id}", headers={"Content-Type": "application/json"})
        res_json = res.json()
    except Exception as e:
        print("{0:=^80}".format("ERROR"))
        print(e)
        return None
    
    if "error" in res_json:
        print("{0:=^80}".format("ERROR"))
        print(json.dumps(res_json, indent=True))
        return None
    
    return res_json

def send_request(user_id: str, input_text: str):
    '''
    When the request is successfully processed, you will receive a JSON object in the following format:
    {
        "model": "registry.ollama.ai/library/llama3:latest",
        "created_at": "2023-12-12T14:13:43.416799Z",
        "message": {
            "role": "assistant",
            "content": response
        },
        "done": true,
        "total_duration": time spent in nanoseconds generating the response,
        "load_duration": time spent in nanoseconds loading the model,
        "prompt_eval_count": number of tokens in the prompt,
        "prompt_eval_duration": time spent in nanoseconds evaluating the prompt,
        "eval_count": number of tokens in the response,
        "eval_duration": time in nanoseconds spent generating the response,
        "status": "done"
    }
    '''
    messages = [
        {"role": "system", "content": "You will compute the sentiment score of articles. Start with the classification, capitalized. Then explain the reason within 100 words."},
        {"role": "user", "content": input_text},
    ]
    req_body = {
        "user_id": user_id,
        "messages": messages,
        "max_new_tokens": 256,
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 10,
    }
    try:
        res = requests.post("http://csc401:8000/submit_request_llama3", json=req_body, headers={"Content-Type": "application/json"})
        res_json = res.json()
    except Exception as e:
        print("{0:=^80}".format("ERROR"))
        print(e)
        return

    if "error" in res_json:
        print("{0:=^80}".format("ERROR"))
        print(json.dumps(res_json, indent=True))
        return
    
    task_id = res_json["task_id"]
    print(f"You have submitted the request to the server: {task_id}")
    time.sleep(2)
    start_time = time.time()
    while True:
        print("checking request")
        res = check_requst_result(user_id, task_id)
        print(res)
        if res is None:
            print("An error occured.")
            return None
        curr_stat = res["status"]
        if curr_stat in ["queued", "processing"]:
            print(f"current status: {curr_stat}")
            time.sleep(5)
        elif curr_stat in ["does not exist", "cancelled", "failed"]:
            print(curr_stat)
            return None
        elif curr_stat == "done":
            return res
        else:
            raise Exception("Unknown status.")
        # time out for 10 min
        if time.time() - start_time > 10 * 60:
            print(f"Time out for task: {task_id}. You can use check_requst_result for checking the result later.")
            return None

def parse_response(response):
    '''
    Input: 
    - json object returned by send_request,
    
    Returns: 
    - a parsed json object to use in process_df
    {   
        "label": extracted text label (POSTIVE, NEUTRAL, NEGATIVE) from the response,
        "label_numeric": 1/0/-1 for positive, neutral, negative,
        "raw_result": ["message"]["content"],
        "compute_time": total_duration in second
    }
    '''
    ###########################
    ##  Your code goes here  ##
    ###########################
    # TODO: retrieve the reponse content and total duration
    # TODO: extract sentiment label from the response content

    raw_result = response["message"]["content"]
    compute_time = response["total_duration"]/1e9
    label = ""
    if "POSITIVE" in raw_result:
        label = "POSITIVE"
        label_numeric = 1
    elif "NEUTRAL" in raw_result:
        label = "NEUTRAL"
        label_numeric = 0
    elif "NEGATIVE" in raw_result:
        label = "NEGATIVE"
        label_numeric = -1
    else:
        label_numeric = -100

    ###################################
    ##  Do not change the following  ##
    ###################################
    return {
        "label": label,
        "label_numeric": label_numeric,
        "raw_result": raw_result,
        "compute_time": compute_time
    }



def process_df(df: pd.DataFrame, output_dir: str, UTORid: str, sample_num = 25, chosen_idx=[]):
    '''
    Randomly sample articles from df, 
    '''
    # Set the random seed using hashed UTORid
    hh = int(hashlib.sha1(UTORid.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    np.random.seed(hh)

    ###########################
    ##  Your code goes here  ##
    ###########################
    if len(chosen_idx) == 0:
    # TODO: use np.random.choice, select sample_num indices from df
        chosen_idx = np.random.choice(df.index, size=sample_num, replace=False)

    # Save the chosen indices to a file for later reference
    indices_file = os.path.join(output_dir, "sampled_indices.txt")
    with open(indices_file, "w") as f:
        f.write(f"Randomly sampled indices: {chosen_idx.tolist()}\n")

    ## Do NOT change the output df definition 
    df_out = {
        "idx": [],
        "text": [],
        "label": [],
        "llama_pred": [],
        "llama_explanation": [],
        "compute_time": []
    }

    # TODO: Iterate through chosen index
    true_labels = []
    predicted_labels = []

    for idx in chosen_idx:
        # TODO: get the text, and label, submit the request using send_request
        # TODO: then parse the response and
        # save the values to df_out, llama_explanation should get the "raw_result" value of the response
        user_id = "wangka76"
        text = df.loc[idx, "text"]
        true_label = df.loc[idx, "label_numeric"]

        response = send_request(user_id, text)

        if response:
            parsed_response = parse_response(response)
            predicted_label = parsed_response['label_numeric']
            
            df_out["idx"].append(idx)
            df_out["text"].append(text)
            df_out["label"].append(true_label)
            df_out["llama_pred"].append(predicted_label)
            df_out["llama_explanation"].append(parsed_response['raw_result'])
            df_out["compute_time"].append(parsed_response['compute_time'])

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        else:
            print(f"Request failed for index {idx}")
    
    # TODO: get the confusion matrix using df_out
    # Compute, accuracy, recall, precision
    C = confusion_matrix(true_labels, predicted_labels, labels=[1, 0, -1])
    # acc, rec, pre = -100, -100, -100
    acc = accuracy(C)
    rec = recall(C)
    pre = precision(C)

    ###################################
    ##  Do not change the following  ##
    ###################################
    # Save the dataframe,
    # Write acc, rec, pre to result_summary with specific format
    pd.DataFrame(df_out).to_csv(f"{output_dir}/processed.csv")
    outf = open(os.path.join(output_dir, f"result_summary.txt"), "w")
    outf.write(f'\tAccuracy: {acc:.4f}\n')
    outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
    outf.write(f'\tPrecision: {[round(item, 4) for item in pre]}\n')
    outf.write(f'\tConfusion Matrix: \n{C}\n\n')
    outf.flush()
    outf.close()

if __name__ == "__main__":
    #################################################
    ##  Your code goes here                        ##
    ##  NOTE: This part of the code is not graded  ##
    #################################################

    # TODO: Read a specific df, drop na of columns: "text", "label_numeric"
    # Call process df on your df (raw text)
    # For Part 1 of the assignment, randomly sample 25 indices from the entire dataset using random choice in process_df
    
    
    path = "/u/cs401/A1/data/fpb_dataset.parquet"
    df = pd.read_parquet(path)
    # #print(df.head())

    process_df(df, "/h/u6/c4/05/wangka76/A1-starter-files-wangka76/results", "wangka76", sample_num = 25, chosen_idx=[])

    # For Part 2 of the assignment, randomly sample 25 indices from test set as defined by train_test_split in classify
    # path = "/h/u6/c4/05/wangka76/A1-starter-files-wangka76/output/wsj89_dataset.parquet"
    # df = pd.read_parquet(path)

    
    # X_train, X_test, y_train, y_test = train_test_split(
    #     df["text"], df["label_numeric"], test_size=0.2, random_state=401
    # )

    # test_df = pd.DataFrame({"text": X_test, "label_numeric": y_test})

    # process_df(test_df, "/h/u6/c4/05/wangka76/A1-starter-files-wangka76/results", "wangka76", sample_num=25, chosen_idx=[])
