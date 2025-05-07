from a1_utils import *
from a1_classify import *
from a1_preprocess import *
from a1_vectorize import *

'''
You can run each function in the __main__ block.

This setup serves as a basic sanity check and does not ensure your correctness in grading.
'''

def check_accuracy():
    y_true = [1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 0]
    expected = 0.6
    C = confusion_matrix(y_true, y_pred)
    res = accuracy(C)
    assert np.allclose(expected, res), f"Expected: {expected}, result: {res}"

def check_precision():
    y_true = [1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 0]
    expected = [2/3, 0.5]
    C = confusion_matrix(y_true, y_pred)
    res = precision(C)
    assert np.allclose(np.array(expected), np.array(res)), f"Expected: {expected}, result: {res}"

def check_recall():
    y_true = [1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    expected = [2/3, 1]
    C = confusion_matrix(y_true, y_pred)
    res = recall(C)
    assert np.allclose(np.array(expected), np.array(res)), f"Expected: {expected}, result: {res}"


def check_preprocess():
    df = {
        "text": [
            "According to Gran , the company  has no plans to move all production to Russia , although that is where the company is growing .\n",
            "\tFinnish-Swedish Stora Enso does not understand the decision issued by a federal judge in Brazil concerning Stora Enso 's associated pulp company Veracel ."
        ],
        "label_numeric": [0, -1],
    }
    df = pd.DataFrame(df)
    preproc = A1Preprocess(in_df=df)
    res1 = preproc.preprocess_single(df["text"].values[0])
    expected1 = "accord gran company plan production russia company grow"
    res2 = preproc.preprocess_single(df["text"].values[1])
    expected2 = "finnish swedish stora enso understand decision issue federal judge brazil concern stora enso associate pulp company veracel"
    assert res1 == expected1, f"Process 1:\n Expected: {expected1}\n Result: {res1}"
    assert res2 == expected2, f"Process 2:\n Expected: {expected2}\n Result: {res2}"

def check_vectorize_count():
    df = {
        "cleaned_text": ["accord accord accord gran gran", "accord company accord accord gran"],
        "label_numeric": [0, -1],
    }
    df = pd.DataFrame(df)
    vec = A1Vectorize(df, vectorizer_max_features=3)
    vocab_df, feat_names = vec.vectorize(df, "cleaned_text")
    assert len(feat_names) == 3
    for feat in feat_names:
        assert feat in ["accord", "accord accord", "gran"]
    expected = np.array([[3/7, 2/7, 2/7],
                        [0.6, 0.2, 0.2]])
    vocab_freq_res = vocab_df[feat_names].values
    assert np.allclose(vocab_freq_res, expected), f"Expected:\n {expected}\n Result: {vocab_freq_res}"

def check_vectorize_tfidf():
    df = {
        "cleaned_text": ["accord accord accord gran gran", "accord company accord accord gran"],
        "label_numeric": [0, -1],
    }
    df = pd.DataFrame(df)
    vec = A1Vectorize(df, vectorizer_type="tfidf", vectorizer_max_features=3)
    vocab_df, feat_names = vec.vectorize(df, "cleaned_text")
    assert len(feat_names) == 3
    for feat in feat_names:
        assert feat in ["accord", "accord accord", "gran"]
    expected = np.array([[0.727607, 0.485071, 0.485071],
                        [0.904534, 0.301511, 0.301511]])
    vocab_freq_res = vocab_df[feat_names].values
    assert np.allclose(vocab_freq_res, expected), f"Expected:\n {expected}\n Result: {vocab_freq_res}"

def check_mean_pooling():
    df = {
        "cleaned_text": ["accord accord accord gran gran", "accord company accord accord gran"],
        "label_numeric": [0, -1],
    }
    df = pd.DataFrame(df)
    vec = A1Vectorize(df, vectorizer_type="tfidf", vectorizer_max_features=3)
    hidden_units = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4]]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    res = vec.mean_pooling((hidden_units, None), attention_mask)
    expected = torch.tensor([1.5, 1.5])
    assert torch.allclose(res, expected), f"Expected:\n {expected.item()}\n Result: {res.item()}"


if __name__ == "__main__":
    check_accuracy()
    check_precision()
    check_recall()
    check_preprocess()
    check_vectorize_count()
    check_vectorize_tfidf()
    check_mean_pooling()