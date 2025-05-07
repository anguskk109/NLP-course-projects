import numpy as np

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    total_correct = np.trace(C)
    total_instances = np.sum(C)
    return total_correct / total_instances if total_instances > 0 else -100


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recalls = np.zeros(C.shape[1])
    for k in range(C.shape[0]):
        actual_class_total = np.sum(C[k, :])
        if actual_class_total > 0:
            recalls[k] = C[k, k] / actual_class_total
        else:
            recalls[k] = -100
    return recalls


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precisions = np.zeros(C.shape[0])
    for k in range(C.shape[0]):
        predicted_class_total = np.sum(C[:, k])

        if predicted_class_total > 0:
            precisions[k] = C[k, k] / predicted_class_total
        else:
            precisions[k] = -100
    return precisions