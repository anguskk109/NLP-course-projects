#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2024 Frank Rudzicz, Gerald Penn

import argparse
import gzip
import html
import json
import os
import re
import shutil
import sys
import tempfile
import time

import pandas as pd
import spacy


def extract_text_from_gz(gz_file_path):
    with gzip.open(gz_file_path, "rt", errors="ignore") as f:
        return f.read()

def generate_wsj_dataframe(input_wsj_gz_file, input_label_parquet_file, output_parquet_file):
    '''
    Inputs:
    - input_wsj_gz_file: the file path to the raw wsj89.gz file, should be hardwired to "/u/cs401/A1/wsj89.tar" in main
    - input_label_parquet_file: the file path to the label file, should be hardwired to "/u/cs401/A1/wsj89_label.parquet" in main
    - output_parquet_file: the file path to the generated parquet file, you decide the name, but the file type should be .parquet
    '''
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.unpack_archive(input_wsj_gz_file, temp_dir)
        # This will create a folder structure of ``temp_dir/wsj/xx/wsj_xxxx.gz''
        all_texts = []
        fns = []
        second_level = os.listdir(os.path.join(temp_dir, "wsj"))
        second_level.sort()
        for sl in second_level:
            curr_level_root = os.path.join(temp_dir, "wsj", sl)
            files = os.listdir(curr_level_root)
            files.sort()
            for file in files:
                # TODO: when we find a gz file:
                # 1. extract the text from the file
                # 2. remove the initial prefix ".START \n\n" from the text
                # 3. append the text and file names to corresponding lists
                gz_file_path = os.path.join(curr_level_root, file)
                text = extract_text_from_gz(gz_file_path)

                fns.append(file)
                all_texts.append(text)

        # TODO: create a pandas dataframe of two columns:
        # "fn": fns,
        # "text": all_texts
        df = pd.DataFrame({
            'fn': fns,
            'text': all_texts
            })
        # TODO: load and merge with input_label_parquet_file, then save the merged df to output_parquet_file
        label_df = pd.read_parquet(input_label_parquet_file)
        df = df.merge(label_df, on='fn', how='left')
        df.to_parquet(output_parquet_file)

class A1Preprocess:
    def __init__(self,
                 in_df: pd.DataFrame,
                 output_dir="./output",
                 filename_prefix="wsj89") -> None:
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.nlp.add_pipe('sentencizer')
        self.df = in_df
        self.output_fn = filename_prefix + "_cleaned.parquet"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def preprocess_single(self, text: str):
        '''
        Preprocess a single text string.

        Inputs:                                                                      
        - text : string, the body of an article

        Returns:
        - modified_text : string, the modified comment 
        '''

        ###########################
        ##  Your code goes here  ##
        ###########################
        filtered_text = text

        # STEP 1
        # TODO: Replace newlines with spaces to handle other whitespace chars.
        filtered_text = re.sub(r'\s+', ' ', text)
        # STEP 2
        # TODO: Unescape HTML
        filtered_text = html.unescape(filtered_text)
        # STEP 3
        # TODO: Remove URLs.
        filtered_text = re.sub(r'http\S+|www\S+', '', filtered_text)
        # STEP 4
        # TODO: Remove all numerical values
        filtered_text = re.sub(r'\d+', '', filtered_text)
        # STEP 5
        # TODO: Remove multiple spaces, and leading/tailing spaces
        filtered_text = filtered_text.strip()
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        # STEP 6
        # TODO: Tokenize text (convert to lower case first)
        doc = self.nlp(filtered_text.lower())
        # TODO: Remove digits, stop words and punctuations, also lemmatize the text
        filtered_text = " ".join([token.lemma_ for token in doc if not (token.is_digit or token.is_stop or token.is_punct)])
        ###################################
        ##  Do not change the following  ##
        ###################################
        # more filtering on bigrams and trigrams
        phrases_to_filter = ["year year", "month month", "week week", "day day", 
                             "wall street journal", "new york times", "new york", "dow jones newswires"
                            ]
        filtered_text = re.sub("|".join(phrases_to_filter), "", filtered_text)
        filtered_text = re.sub(r"\s+", " ", filtered_text) # clean up multiple spaces
        # Return the cleaned text
        return filtered_text
    
    def process(self):
        df = self.df
        output_df_fn = os.path.join(self.output_dir, self.output_fn)

        start_time = time.time()

        ###########################
        ##  Your code goes here  ##
        ###########################
        
        # TODO: 
        # 1. iterate through the dataframe
        # 2. call process_single on each text
        # 3. save to a new column: cleaned_text
        df['cleaned_text'] = df['text'].apply(self.preprocess_single)
        
        ###################################
        ##  Do not change the following  ##
        ###################################
        df.to_parquet(output_df_fn, index=False)
        print(f"Done. Time elapsed: {time.time() - start_time:.4f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A1 Preprocessing')
    parser.add_argument("--output_dir", help="Directs the output to folder of your choice", default="./output")
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default="/u/cs401/A1/data")
    parser.add_argument("--filename_prefix", help="The prefix of filename for the raw file to process. One of \"fpb\"/\"wsj89\". Defaults to \"wsj89\".", default="wsj89")
    args = parser.parse_args()

    #################################################
    ##  Your code goes here                        ##
    ##  NOTE: This part of the code is not graded  ##
    #################################################
    print("{0:=^80}".format("A1 Preprocess"))
    print(args)

    if args.filename_prefix == "wsj89":
        os.makedirs(args.output_dir, exist_ok=True)
        input_wsj_gz_file = os.path.join(args.a1_dir, "wsj89.tar")
        input_label_parquet_file = os.path.join(args.a1_dir, "wsj89_labels.parquet")
        output_parquet_file = os.path.join(args.output_dir, "wsj89_dataset.parquet")
        # TODO: Generate WSJ dataframe from gz file if a preprocessed version doesn't exist, and save it to your output_dir.
        # Then read in the raw dataframe you just generated
        if not os.path.exists(output_parquet_file):
            generate_wsj_dataframe(input_wsj_gz_file, input_label_parquet_file, output_parquet_file)
    
        df = pd.read_parquet(output_parquet_file)
    else:
        # TODO: Read in the fpb dataframe from a1 dir     
        path = "/u/cs401/A1/data/fpb_dataset.parquet"
        df = pd.read_parquet(path)

    # TODO: Instantiate an A1Preprocess object, with required information, and call process.
    preprocessor = A1Preprocess(in_df=df, output_dir=args.output_dir, filename_prefix=args.filename_prefix)
    preprocessor.process()




