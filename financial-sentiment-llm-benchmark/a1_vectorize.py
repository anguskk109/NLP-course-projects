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
import json
import os
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


class A1Vectorize:
    def __init__(self, 
                 in_df: pd.DataFrame,
                 data_column="cleaned_text",
                 target_column="label_numeric",
                 output_dir="./output",
                 filename_prefix="wsj89",
                 vectorizer_type="count",
                 vectorizer_max_features=10000,
                 vectorizer_ngram_range=(1,3)) -> None:
        self.df = in_df
        self.data_column = data_column
        self.target_column = target_column
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        os.makedirs(self.output_dir, exist_ok=True)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.nlp.Defaults.stop_words.add("ve")
        self.nlp.Defaults.stop_words.add("ll")
        self.nlp.max_length = 2000000

        self.vectorizer_type = vectorizer_type

        ###########################
        ##  Your code goes here  ##
        ###########################
        
        # TODO: instantiate vectorizers based on vectorizer type "count", "tfidf", "mpnet"
        # For count vectorizer, use dtype=np.int32
        # For tf-idf vectorizer, use dtype=np.float32
        # For both count and tf-idf, make sure stop_words argument is properly set with self.nlp.Defaults.stop_words
        # For mpnet, set self.device with "cuda" or "cpu" based on availability of CUDA device
        # Use AutoTokenizer and AutoModel to instantiate "sentence-transformers/all-mpnet-base-v2". 
        # Make sure that you set the cache_dir to "/u/cs401/A1/model_weights" and there should be no download of models.

        if vectorizer_type == "count":
            self.vectorizer = CountVectorizer(max_features=vectorizer_max_features,
                                              ngram_range=vectorizer_ngram_range,
                                              stop_words=list(self.nlp.Defaults.stop_words),
                                              dtype=np.int32)
        elif vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=vectorizer_max_features,
                                              ngram_range=vectorizer_ngram_range,
                                              stop_words=list(self.nlp.Defaults.stop_words),
                                              dtype=np.float32)
        elif vectorizer_type == "mpnet":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir="/u/cs401/A1/model_weights")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir="/u/cs401/A1/model_weights").to(self.device)
        
        
        
    def mean_pooling(self, model_output, attention_mask):
        '''
        Mean Pooling - Take attention mask into account for correct averaging
        This is only used for mpnet
        '''
        ###########################
        ##  Your code goes here  ##
        ###########################
        
        # TODO: Get the first element of model_output which contains all token embeddings
        token_embeddings = model_output[0]
        # TODO: Compute weighted average based on attention mask, only positions with attention_mask=1 should be included in the weighted average
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def vectorize(self, df: pd.DataFrame, data_column: str):
        '''
        Inputs:
        - df: the dataframe to vectorize
        - data_column: the column in the dataframe to vectorize 

        Returns:
        - vocab_freq_df: When using count vectorizer, we normalize the count of each gram by the total count to convert to frequencies.
        When using Tf-idf vectorizer, return the tf-idf dataframe.
        When using MPNET, convert the sentence embedding into a dataframe, with the provided column names
        - feature_names: the extracted n-grams. convert it to list.
        '''

        if self.vectorizer_type == "count":
            '''
            When using count vectorizer, we normalize the count of each gram by the total count to convert to frequencies
            '''
            ###########################
            ##  Your code goes here  ##
            ###########################

            # TODO: fit the vectorizer and transform the input
            X = self.vectorizer.fit_transform(df[data_column])
            # TODO: use vectorizer.get_feature_names_out to get the columns of the dataframe (feature names)
            feature_names = self.vectorizer.get_feature_names_out().tolist()
            # Construct the frequency data frame by dividing the total count
            vocab_count_df = pd.DataFrame(X.toarray(), columns=feature_names)
            vocab_freq_df = vocab_count_df.div(vocab_count_df.sum(axis=1), axis=0)
            
            return vocab_freq_df, feature_names
        elif self.vectorizer_type == "tfidf":
            ###########################
            ##  Your code goes here  ##
            ###########################
            # TODO: fit the vectorizer and transform the input
            X = self.vectorizer.fit_transform(df[data_column])
            # TODO: use vectorizer.get_feature_names_out to get the columns of the dataframe (feature names)
            feature_names = self.vectorizer.get_feature_names_out().tolist()
            vocab_tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
            
            return vocab_tfidf_df, feature_names
        elif self.vectorizer_type == "mpnet":
            res_arr = np.zeros((len(df), 768))

            # TODO: Iterate through the dataframe, and embed each data_column item
            for i in range(len(df)):
                if (i + 1) % 500 == 0:
                    print(f"Finished {i+1} items")
                text = [df[data_column].values[i]]
                
                # TODO: Tokenize the text, and move to appropriate device
                tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
                # TODO: Compute token embeddings
                
                with torch.no_grad(): # only for inferencing
                    model_output = self.model(**tokens)
                # TODO: Perform mean pooling
                
                sentence_embeddings = self.mean_pooling(model_output, tokens['attention_mask'])
                # TODO: Normalize embeddings with F.normalize, use default p=2
                
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                res_arr[i, :] = sentence_embeddings.detach().cpu().numpy()

            ###################################
            ##  Do not change the following  ##
            ###################################
            columns = [f"feat_{i}" for i in range(768)]
            embed_df = pd.DataFrame(res_arr, columns=columns)
            return embed_df, columns
        
    def run(self):
        start_time = time.time()
        vocab_df, feature_names = self.vectorize(self.df, self.data_column)

        if "date" in list(self.df.columns):
            df_y_values = self.df[["date", self.target_column]].copy()
            df_y_values = df_y_values.rename(columns={"date": "DATA_DATE_"})
        else:
            df_y_values = self.df[[self.target_column]].copy()

        ###########################
        ##  Your code goes here  ##
        ###########################
        # TODO: properly merge the dataframe, based on indices
        vocab_df = pd.concat([df_y_values.reset_index(drop=True), vocab_df.reset_index(drop=True)], axis=1)

        ###################################
        ##  Do not change the following  ##
        ###################################
        vocab_df_name = f"{self.output_dir}/{self.filename_prefix}_vectorized_{self.vectorizer_type}.csv"
        feature_file_name = f"{self.output_dir}/{self.filename_prefix}_feature_names_{self.vectorizer_type}.json"
        vocab_df.to_csv(vocab_df_name, index=False)
        with open(feature_file_name, "w") as f:
            f.write(json.dumps(feature_names, indent=True))
        print(f"Done. Time elapsed: {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A1 Vectorize')
    parser.add_argument("--filename_prefix", type=str, help="The prefix of filename for file to process. One of \"fpb\"/\"wsj89\". Defaults to \"wsj89\".", default="wsj89")
    parser.add_argument("--data_column", type=str, help="Column in dataframe to vectorize. Default: cleaned_text. You should not have to change this for count and tf-idf vectorizer. For mpnet, you should use text.", default="cleaned_text")
    parser.add_argument("--target_column", type=str, help="Column in dataframe as target variable. Default: label_numeric. You should not have to change this.", default="label_numeric")
    parser.add_argument("--output_dir", type=str, help="Directs the output to folder of your choice", default="./output")
    parser.add_argument("--vectorizer_type", type=str, help="The type of vectorizer to use. \"count\" for Count Vectorizer, \"tfidf\" for Tf-iDF vectorizer, and \"mpnet\" for the MPNET language model.", default="count")
    parser.add_argument("--max_features", type=int, help="The maximum features (vocabs) to extract across the corpus.", default=10000)
    parser.add_argument("--min_ngrams", type=int, help="The lower boundary of the range of n-values for different word n-grams or char n-grams to be extracted.", default=1)
    parser.add_argument("--max_ngrams", type=int, help="The upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.", default=3)
    args = parser.parse_args()

    #################################################
    ##  Your code goes here                        ##
    ##  NOTE: This part of the code is not graded  ##
    #################################################

    # TODO: Read in the dataframe processed by a1_preprocess, with specific filename prefix (wsj89 or fpb), based on the argument
    input_parquet_file = os.path.join(args.output_dir, f"{args.filename_prefix}_cleaned.parquet")
    df = pd.read_parquet(input_parquet_file)
    # TODO: Instantiate an A1Vectorize object, with required information, and call run.
    vectorizer = A1Vectorize(
        in_df=df,
        data_column=args.data_column,
        target_column=args.target_column,
        output_dir=args.output_dir,
        filename_prefix=args.filename_prefix,
        vectorizer_type=args.vectorizer_type,
        vectorizer_max_features=args.max_features,
        vectorizer_ngram_range=(args.min_ngrams, args.max_ngrams)
    )
    vectorizer.run()

