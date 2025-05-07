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
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from a1_utils import *

warnings.filterwarnings("ignore")

    
class A1Classify:
    def __init__(self, 
                 vectorized_df: pd.DataFrame,
                 feature_names: List[str],
                 output_dir: str="./output",
                 filename_prefix: str="wsj89",
                 vectorizer_type: str="count",
                 target_column: str="label_numeric") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.vectorized_df = vectorized_df.dropna()
        self.feature_names = feature_names
        self.target_column = target_column

        self.filename_prefix = filename_prefix
        self.vectorizer_type = vectorizer_type
        assert self.vectorizer_type in ["count", "tfidf", "mpnet"], "Unknown vectorizer type"

        self.X = self.vectorized_df.loc[:, self.feature_names].values
        self.y = self.vectorized_df.loc[:, target_column].values
    
    def classify(self, train_size=0.8):
        '''
        This function performs experiment 4.1
        '''
        outf = open(os.path.join(self.output_dir, f"a1_classify_{self.filename_prefix}_{self.vectorizer_type}.txt"), "w")
        ###########################
        ##  Your code goes here  ##
        ###########################
        # TODO: perform train_test_split on self.X and self.y, with random_state=401, train_size as provided.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=401)
        
        # TODO: Fit the classifiers, compute confusion matrix, accuracy, precision and recall,
        # TODO: write the results in the provided format
        best_idx, accBest = None, -1

        classifiers = {
            "GaussianNB": GaussianNB(),
            "MLPClassifier": MLPClassifier(alpha=0.05, max_iter=1000)
            }

        for idx, (clf_name, clf) in enumerate(classifiers.items()):

            clf.fit(X_train, y_train)  
            y_pred = clf.predict(X_test)
            
            C = confusion_matrix(y_test, y_pred)
            acc = accuracy(C)
            rec = recall(C)
            pre = precision(C)
            
            # Uncomment the following for writing to the file
            outf.write(f'Results for {clf_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in pre]}\n')
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')
            outf.flush()

            if acc > accBest:
                accBest = acc
                best_idx = idx
        
        outf.close()
        return best_idx
    
    def classify_top_feats(self, best_idx, train_size=0.8):
        ''' This function performs experiment 4.2

        Parameters:
        best_idx: int, the index of the supposed best classifier (from task 4.1)
        '''
        outf = open(os.path.join(self.output_dir, f"a1_classify_{self.filename_prefix}_top_feats_{self.vectorizer_type}.txt"), "w")

        ###########################
        ##  Your code goes here  ##
        ###########################
        # TODO: perform train_test_split on self.X and self.y, with random_state=401, train_size as provided.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=401)

        for k in (5, 50):
            # TODO: Select the top k features using SelectKBest, compute p values, and write to output file with the provided format
            selector = SelectKBest(f_classif, k=k)
            X_new = selector.fit_transform(X_train, y_train)
            pvals = selector.pvalues_
            
            outf.write(f'{k} p-values: {[format(pval) for pval in pvals]}\n')

        # TODO: Select the top 5 features, train the best classifier, compute the accuracy
        selector_5 = SelectKBest(f_classif, k=5)
        X_train_5 = selector_5.fit_transform(X_train, y_train)
        X_test_5 = selector_5.transform(X_test)

        classifiers = {
            0: GaussianNB(),
            1: MLPClassifier(alpha=0.05, max_iter=1000)
            }
    
        clf = classifiers[best_idx]
        clf.fit(X_train_5, y_train)

        y_pred = clf.predict(X_test_5)  
        C = confusion_matrix(y_test, y_pred)
        acc = accuracy(C)
        top_idx = selector_5.get_support(indices=True)
        
        ###################################
        ##  Do not change the following  ##
        ###################################
        outf.write(f'Accuracy for full dataset: {acc}\n')
        outf.write(f'Top-5: {top_idx}\n')
        outf.flush()


    def classify_cross_validation(self, best_idx):
        print("Generating 5 fold train/test split")

        ###########################
        ##  Your code goes here  ##
        ###########################

        # TODO: Generate KFold, with shuffle
        kf = KFold(n_splits=5, shuffle=True, random_state=401)
        folds = []

        # TODO: train/test each model on each fold and record the accuracy
        classifiers = {
            0: GaussianNB(),
            1: MLPClassifier(alpha=0.05, max_iter=1000)
            }
        
        classifier_accuracies = {0: [], 1: []}

        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            for idx, clf in classifiers.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                C = confusion_matrix(y_test, y_pred)
                acc = accuracy(C)
                classifier_accuracies[idx].append(acc)

        folds = list(zip(classifier_accuracies[0], classifier_accuracies[1]))

        # TODO: compute the p-values of accuracy using ttest_ind, as compared to the best model
        other_clf_idx = 1 if best_idx == 0 else 0
        p_value = ttest_ind(classifier_accuracies[best_idx], classifier_accuracies[other_clf_idx]).pvalue
        p_values = [p_value]

        with open(f"{self.output_dir}/a1_classify_{self.filename_prefix}_cross_valid_{self.vectorizer_type}.txt", "w") as outf:
            # Prepare kfold_accuracies, then modify this, so it writes them to outf.
            for i, (acc_gnb, acc_mlp) in enumerate(folds):
                kfold_accuracies = [acc_gnb, acc_mlp]
                outf.write(f'Fold {i} Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directs the output to folder of your choice.", default="./output")
    parser.add_argument("--filename_prefix", type=str, help="The prefix of filename for file to process. One of \"fpb\"/\"wsj89\". Defaults to \"wsj89\".", default="wsj89")
    parser.add_argument("--vectorizer_type", type=str, help="The type of vectorizer to use. \"count\" for Count Vectorizer, \"tfidf\" for Tf-iDF vectorizer, and \"mpnet\" for the MPNET language model.", default="count")
    parser.add_argument("--target_column", type=str, help="Column in dataframe as target variable. Default: label_numeric. You should not have to change this.", default="label_numeric")
    args = parser.parse_args()

    # set the random state for reproducibility 
    np.random.seed(401)

    #################################################
    ##  Your code goes here                        ##
    ##  NOTE: This part of the code is not graded  ##
    #################################################

    # TODO: Read in the dataframe and feature names generated by a1_vectorize, with specific filename prefix (wsj89 or fpb), and vectorizer type (count, tfidf, mpnet), based on the argument
    # TODO: Instantiate an A1Classify object, with required information, and call classify, classify_top_feats, and classify_cross_validation.
    # For classify_top_feats and classify_cross_validation, use the output best_idx from classify.

    vectorized_file = os.path.join(args.output_dir, f"{args.filename_prefix}_vectorized_{args.vectorizer_type}.csv")
    feature_names_file = os.path.join(args.output_dir, f"{args.filename_prefix}_feature_names_{args.vectorizer_type}.json")

    vectorized_df = pd.read_csv(vectorized_file)
    with open(feature_names_file, "r") as f:
        feature_names = json.load(f)

    classifier = A1Classify(
        vectorized_df=vectorized_df,
        feature_names=feature_names,
        output_dir=args.output_dir,
        filename_prefix=args.filename_prefix,
        vectorizer_type=args.vectorizer_type,
        target_column=args.target_column
    )

    best_idx = classifier.classify(train_size=0.8)

    classifier.classify_top_feats(best_idx)

    classifier.classify_cross_validation(best_idx)




# Written Answers

# Name the top 5 features chosen using feature names extracted in Step 2
# For FPB data, Top-5 features: [ 59  64 291 349 766] which are [address, administration, annual general meeting, approve, biotie north]
# For wsj89, Top-5 features: [3246 5308 5550 7068 9507] which are [fantasy, mark yen, miller, pro democracy, urban development]

#  Hypothesize as to why those particular features (n-grams) might differentiate the classes

#  For FPB, Annual General Meeting is a significant event for publicly traded companies, 
#  where shareholders gather to discuss company performance and vote on key issues,
#  which very possibly affects the bussiness

# For wsj89, politics ("pro democracy") and currency ("mark yen") are always the key areas
# that a bussiness should pay attention to and affects bussiness

