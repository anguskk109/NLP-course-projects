""""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Arvie Frydenlund, Raeid Saqur and Jingcheng Niu

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

"""
Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
"""

from math import exp  # exp(x) gives e^x
from collections.abc import Sequence


def grouper(seq: Sequence[str], n: int) -> list:
    """
    Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    # assert False, "Fill me"
    if n > len(seq):
        return []

    ngrams = [tuple(seq[i : i+n]) for i in range(len(seq) -n+1)]

    return ngrams


def n_gram_precision(
    reference: Sequence[str], candidate: Sequence[str], n: int
) -> float:
    """
    Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    # assert False, "Fill me"
    ref_ngrams = grouper(reference, n)
    can_ngrams = grouper(candidate, n)

    if len(can_ngrams) ==0:
        return 0.0

    ref_counts = {}
    for ngram in ref_ngrams:
        if ngram in ref_counts:
            ref_counts[ngram] +=1
        else:
            ref_counts[ngram] = 1

    match_count = 0
    for ngram in can_ngrams:
        if ngram in ref_counts:
            match_count += min(can_ngrams.count(ngram), ref_counts[ngram])

    p_n = match_count / len(can_ngrams)
    return p_n

def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """
    Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """
    # assert False, "Fill me"
    len_ref = len(reference)
    len_can = len(candidate)

    if len_can ==0:
        return 0.0
    
    if len_can > len_ref:
        return 1.0
    else:
        return exp(1- (len_ref/len_can))


def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n) -> float:
    """
    Calculate the BLEU score.  Please scale the BLEU score by 100.0
    In the case that the candidate has length less than n, p_n is 0.

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    """
    # assert False, "Fill me"

    if len(candidate) < n:
        return 0.0

    precisions = [n_gram_precision(reference, candidate, i) for i in range(1, n+1)]

    pop = 1.0 # product of precisions

    for p in precisions:
        pop = pop*p

    if pop == 0:
        geo_mean = 0.0
    else:
        geo_mean = pop **(1/n)

    BP = brevity_penalty(reference, candidate)

    BLEU = BP * geo_mean *100.0

    return BLEU
