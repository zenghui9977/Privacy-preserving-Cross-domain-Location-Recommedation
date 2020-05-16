import math

def precision(ground_truth, prediction):
    return 1.0 * len(set(ground_truth) & set(prediction)) / len(prediction)

def recall(ground_truth, prediction):
    return 1.0 * len(set(ground_truth) & set(prediction)) / len(ground_truth)

def f_measure(ground_truth, prediction):
    pre = precision(ground_truth, prediction)
    rec = recall(ground_truth, prediction)
    if pre + rec == 0:
        return 0
    else:
        return 2 * pre * rec / (pre + rec)

def hit_num_k(ground_truth, prediction):
    return len(set(ground_truth) & set(prediction))


def getNDCG(ground_truth, predction):
    for i in range(len(predction)):
        item = predction[i]
        if item == ground_truth:
            return math.log(2) / math.log(i+2)
    return 0