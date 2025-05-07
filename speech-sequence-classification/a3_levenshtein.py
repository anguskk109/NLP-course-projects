import os
import re
import string
from typing import List, Literal
import numpy as np

dataDir = '/u/cs401/A3/data/'

# Class definition for an individual table cell. DO NOT MODIFY
class TableCell(object):
    EditTypes = Literal['del', 'ins', 'match', 'sub', 'first']
    def __init__(self, cost:int = 0, trace:EditTypes = 'first'):
        """
        initialize: defines an individual table cell

        Parameters
        ----------
        cost : minimal number of insertions, deletions, and substitutions to get to this cell
        trace : The last edit that was made to enter this cell.

        Returns
        -------
        N/A

        Examples
        --------
        Creating a cell
        >>> cell = TableCell(1, "ins")

        Checking the cost of a cell
        >>> if cell.cost == 1:
        ...     ...

        Modifying the edit type
        >>> cell.trace = "match"
        """
        super().__init__()
        self.cost, self.trace = cost, trace

# Preprocess text for best result. DO NOT MODIFY!
def preprocess(raw_line):
    # remove label
    line = raw_line.split(" ", 2)[-1]

    # remove tags (official)
    line = re.sub(r'<[A-Z]+>', '', line)

    # remove tags (Kaldi)
    line = re.sub(r'\[[a-z]+\]', '', line)

    # remove punctuations
    line = line.translate(str.maketrans("", "", string.punctuation))

    return line

# Complete the following component for Levenshtein
def initialize(U : int, T : int) -> List[List[TableCell]]:
    """
    initialize: allocates the table and initializes the first row and column, 
    representing an alignment with an empty sequence. 

    Parameters
    ----------
    U : int
        Number of elements in the reference sequence (e.g., ground truth).
    T : int
        Number of elements in the hypothesis sequence (e.g., predicted sequence).

    Returns
    -------
    table : List[List[TableCell]]
        A list of lists of `TableCell` instances, representing a dynamic programming table
        of size [U + 1, T + 1]. Each cell in the table will store the minimal cost and 
        the type of edit operation (trace) leading to that cell.

    Initialization Details
    ----------------------
    - The `[0,0]` cell (top-left corner) is initialized with a cost of `0` and a trace of `"first"`, 
      indicating the starting point with no edits required.
    - The first row represents the scenario where we align the hypothesis with an empty reference:
        - Each cell in this row, from `[0,1]` to `[0,T]`, is initialized with increasing costs 
          based on the index (i.e., `cost = col`) and trace set to `"ins"` for each column.
    - The first column represents the scenario where we align the reference with an empty hypothesis:
        - Each cell in this column, from `[1,0]` to `[U,0]`, is initialized with increasing costs 
          based on the index (i.e., `cost = row`) and trace set to `"del"` for each row.
    """

    ############################################
    table = [[TableCell() for _ in range(T + 1)] for _ in range(U + 1)]
    # Initialize the first row
    for t in range(1, T + 1):
        table[0][t] = TableCell(cost=t, trace="ins")
    # Initialize the first column
    for u in range(1, U + 1):
        table[u][0] = TableCell(cost=u, trace="del")
    ############################################

    # Make sure the type of values are correct
    assert isinstance(table, list) and all(isinstance(row, list) and all(isinstance(cell, TableCell) for cell in row) for row in table)
    return table

def step(u : int, t : int, table : List[List[TableCell]], r : List[str], h : List[str]) -> None:
    """
    step: computes the value of the current cell

    **NOTE** :  in case of tie, use the following priority: "match" > "sub" > "ins" > "del"

    Parameters
    ----------
    u : int, row index of the current cell
    t : int, col index of the current cell
    table : list of list of TableCell of size [U + 1][T + 1]
    r : list of strings, representing the reference sentence
    h : list of strings, representing the hypothesis sentence

    Returns
    -------
    N/A
    """
    ############################################
    if r[u - 1] == h[t - 1]:  # Match
        match_cost = table[u - 1][t - 1].cost
        match_trace = "match"
    else:  # Substitution
        match_cost = table[u - 1][t - 1].cost + 1
        match_trace = "sub"

    # insertion cost
    ins_cost = table[u][t - 1].cost + 1
    ins_trace = "ins"

    # deletion cost
    del_cost = table[u - 1][t].cost + 1
    del_trace = "del"

    # priority: match > sub > ins > del
    min_cost, min_trace = min(
        (match_cost, match_trace),
        (ins_cost, ins_trace),
        (del_cost, del_trace),
        key=lambda x: (x[0], ["match", "sub", "ins", "del"].index(x[1]))
    )

    # Update the table cell
    table[u][t] = TableCell(cost=min_cost, trace=min_trace)
    ############################################
    return

def finalize(table):
    """
    finalize: computes the final results, including WER, number of all operations

    NOTE: If the reference sequence is of length 0, WER should be `float("inf")`

    Parameters
    ----------
    table : list of list of TableCell of size [U + 1][T + 1]

    Returns
    -------
    (WER, nD, nS, nI): (float, int, int, int) WER, number of deletions, substitutions, and insertions respectively

    Order of Return (This Year's Convention)
    ----------------------------------------
    - This year's convention is to return the operations in the following order:
      - Deletions (`nD`)
      - Substitutions (`nS`)
      - Insertions (`nI`)
      
    Previous Orders
    ---------------
    - Last year's order was: Insertions, Deletions, Substitutions
    - The order two years ago was: Substitutions, Insertions, Deletions
    """

    # Define results to be returned:
    wer = 0.0
    deletions = 0
    substitutions = 0
    insertions = 0

    ############################################
    U, T = len(table) - 1, len(table[0]) - 1
    reference_length = U

    # If reference length is 0, return WER as infinity
    if reference_length == 0:
        return (float("inf"), deletions, substitutions, insertions)

    while U > 0 or T > 0:
        current_cell = table[U][T]

        if current_cell.trace == "match" or current_cell.trace == "sub":
            if current_cell.trace == "sub":
                substitutions += 1
            U -= 1
            T -= 1
        elif current_cell.trace == "ins":
            insertions += 1
            T -= 1
        elif current_cell.trace == "del":
            deletions += 1
            U -= 1


    wer = (deletions + substitutions + insertions) / reference_length
    ############################################

    return (wer, deletions, substitutions, insertions)

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    You should complete the core component of this function.
    DO NOT MODIFY ANYTHING IN HERE

    Parameters
    ----------
    r : list of strings, representing the reference sentence
    h : list of strings, representing the hypothesis sentence

    Returns
    -------
    (WER, nD, nS, nI): (float, int, int, int) WER, number of deletions, substitutions, and insertions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 1 0 0
    >>> wer("who is there".split(), "".split())
    1.0 3 0 0
    >>> wer("".split(), "who is there".split())
    Inf 0 0 3
    """

    # U: length of reference;
    # T: length of hypothesis
    U, T = len(r), len(h)

    ############################################
    ############### Levenshtein: ###############
    ############################################

    # Call initialize() to create table
    table = initialize(U, T)

    # Iterate over the remaining cols and rows
    # Use values of words and the previously computed cells to compute current cell
    for u in range(1, U + 1, 1):
        for t in range(1, T + 1, 1):
            step(u, t, table, r, h)

    # Use table to compute final results
    # A.K.A WER, deletions, substitutions and insertions
    return finalize(table)

if __name__ == "__main__":
    """
    Main Function: Generates a file that has all the result. DO NOT MODIFY! 

    Output Format: ([] contains argument)
    --------------
    [Speaker Name] [System Name] [Line Number i] [WER] D:[# Deletions], S:[# Substitutions], I:[# Insertions]
    [Speaker Name] [System Name] [Line Number i] [WER] D:[# Deletions], S:[# Substitutions], I:[# Insertions]
    ...
    FINAL KALDI AVG = [Some Number] +- [Some Number]
    FINAL GOOGLE WER = [Some Number] +- [Some Number]
    """

    GTF = "transcripts.txt"
    GOOGLE = "transcripts.Google.txt"
    KALDI = "transcripts.Kaldi.txt"

    # Make sure the data is clean and usable
    def check_valid(sdir):
        def exists(name):
            path = os.path.join(sdir, name)
            return os.path.exists(path)

        return exists(GTF) or exists(GOOGLE) or exists(KALDI)

    # Load speakers names
    speakers = os.listdir(dataDir)

    wer_google = []
    wer_kaldi = []

    # Actual Process
    print("a3_levenshtein process is running...")
    with open('a3_levenshtein.out', 'w') as file:
        for s in speakers:
            # Form the full path
            speaker_dir = os.path.join(dataDir, s)
            if not check_valid(speaker_dir):
                continue

            # Read all three transcripts
            google = open(os.path.join(speaker_dir, GOOGLE)).readlines()
            kaldi = open(os.path.join(speaker_dir, KALDI)).readlines()
            groundtruth = open(os.path.join(speaker_dir, GTF)).readlines()

            print(f"Processing Speaker {s}...")

            for i, (g, k, r) in enumerate(zip(google, kaldi, groundtruth)):
                # Preprocess the lines
                google_sample = preprocess(g)
                kaldi_sample = preprocess(k)
                gt_sample = preprocess(r)

                # Caluculate WER for Google and Kaldi
                google_lev = Levenshtein(gt_sample.split(), google_sample.split())
                kaldi_lev = Levenshtein(gt_sample.split(), kaldi_sample.split())

                # Append the result for final output
                wer_google.append(google_lev[0])
                wer_kaldi.append(kaldi_lev[0])

                print(f'{s} Google {i} {google_lev[0]:.6f} D:{google_lev[1]} S:{google_lev[2]} I:{google_lev[3]}', file=file)
                print(f'{s} Kaldi {i} {kaldi_lev[0]:.6f} D:{kaldi_lev[1]} S:{kaldi_lev[2]} I:{kaldi_lev[3]}', file=file)

        wer_google = np.array(wer_google)
        wer_kaldi = np.array(wer_kaldi)

        # Print out the final result
        print(f"FINAL KALDI AVG = {np.mean(wer_kaldi) :.4f} +- {np.std(wer_kaldi):.4f}", file=file)
        print(f"FINAL GOOGLE WER = {np.mean(wer_google):.4f} +- {np.std(wer_google):.4f}", file=file)
    print("a3_levenshtein process has completed!")
