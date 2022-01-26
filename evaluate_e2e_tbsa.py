### Adapted from
# https://github.com/lixin4ever/E2E-TBSA/blob/8cb7182d890d2b36766b23c788bb82c265ca4242/evals.py#L98

import numpy as np
import csv
import ast
import sys

SMALL_POSITIVE_CONST = 1e-4

# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/exp/other/transformed-targets.csv'
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'models/exp/other/transformed-sentiments.csv'

TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/commongen_evaluation_old_prompt_dataset2_early_stopping/transformed-targets.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'models/commongen_evaluation_old_prompt_dataset2_early_stopping/transformed-sentiments.csv'

# Dhruv's e.g. evaluate_ote("O O O S O B I E", "O O O S O O O O") ->
# (0.9999000099990001, 0.9999000099990001, 0.9998500124991251)
# print(evaluate_ote(["O O O S O B I E"], ["O O O S O O O O"]))
# -> (0.9999000099990001, 0.49997500124993743, 0.6665777829629381)
# Updated if not tag conversion: print(evaluate_ote("[(6, 6), (10, 14)]", "[(6, 6), (10, 14)]"))
def evaluate_ote(gold_ot, pred_ot):
    """
    evaluate the model performance for the ote task
    :param gold_ot: gold standard ote tags
    :param pred_ot: predicted ote tags
    :return:
    """
    # assert len(gold_ot) == len(pred_ot)
    n_samples = len(gold_ot)
    # number of true positive, gold standard, predicted opinion targets
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    for i in range(n_samples):
        g_ot = gold_ot[i]
        p_ot = pred_ot[i]
        # g_ot_sequence, p_ot_sequence = tag2ot(ote_tag_sequence=g_ot), tag2ot(ote_tag_sequence=p_ot)
        g_ot_sequence, p_ot_sequence = g_ot, p_ot
        # hit number
        n_hit_ot = match_ot(gold_ote_sequence=g_ot_sequence, pred_ote_sequence=p_ot_sequence)
        n_tp_ot += n_hit_ot
        n_gold_ot += len(g_ot_sequence)
        n_pred_ot += len(p_ot_sequence)
    # add 0.001 for smoothing
    # calculate precision, recall and f1 for ote task
    ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)
    ote_scores = (ot_precision, ot_recall, ot_f1)
    return ote_scores

def match_ot(gold_ote_sequence, pred_ote_sequence):
    """
    calculate the number of correctly predicted opinion target
    :param gold_ote_sequence: gold standard opinion target sequence
    :param pred_ote_sequence: predicted opinion target sequence
    :return: matched number
    """
    n_hit = 0
    for t in pred_ote_sequence:
        if t in gold_ote_sequence:
            n_hit += 1
    return n_hit

# Dhruv's e.g. Updated: print(evaluate_ts([[(1, 3, 'POS'), (4, 4, 'NEG')]], [[(1, 3, 'POS'), (4, 4, 'NEG')]]))
def evaluate_ts(gold_ts, pred_ts):
    """
    evaluate the model performance for the ts task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return:
    """
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)
    # number of true positive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
        # g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts), tag2ts(ts_tag_sequence=p_ts)
        g_ts_sequence, p_ts_sequence = g_ts, p_ts
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence, idx=i)

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        # calculate macro-average scores for ts task
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    ts_macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    n_tp_total = sum(n_tp_ts)
    # total sum of TP and FN
    n_g_total = sum(n_gold_ts)
    # total sum of TP and FP
    n_p_total = sum(n_pred_ts)

    ts_micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    ts_micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    ts_micro_f1 = 2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + SMALL_POSITIVE_CONST)
    ts_scores = (ts_micro_p, ts_micro_r, ts_micro_f1)

    return ts_scores


def match_ts(gold_ts_sequence, pred_ts_sequence, idx):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        # print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    # if not np.all(hit_count == gold_count) or not np.all(hit_count == pred_count):
    #     print("{}".format(idx))
    return hit_count, gold_count, pred_count


def read_transformed_targets(transformed_targets_predictions_file):
    predicted_data = []
    gold_data = []
    with open(transformed_targets_predictions_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            predicted_idx = ast.literal_eval(line[0])
            gold_idx = ast.literal_eval(line[1])
            predicted_data.append(predicted_idx)
            gold_data.append(gold_idx)
    return predicted_data, gold_data


def read_transformed_sentiments(transformed_sentiments_predictions_file):
    predicted_data = []
    gold_data = []
    with open(transformed_sentiments_predictions_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            predicted_idx = ast.literal_eval(line[0])
            gold_idx = ast.literal_eval(line[1])
            predicted_data.append(predicted_idx)
            gold_data.append(gold_idx)
    return predicted_data, gold_data


def run_from_generative_script(file_to_write):

    predicted_data_targets, gold_data_targets = read_transformed_targets(TRANSFORMED_TARGETS_PREDICTIONS_FILE)
    output_targets = evaluate_ote(gold_data_targets, predicted_data_targets)
    print(output_targets)

    predicted_data_sentiment, gold_data_sentiment = read_transformed_sentiments(TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE)
    output_sentiment = evaluate_ts(gold_data_sentiment, predicted_data_sentiment)
    print(output_sentiment)

    if file_to_write is not None:
        print("Writing to : {}".format(file_to_write))
        with open(file_to_write, 'a') as file:
            file.write("{}, {}\n".format(output_targets[2], output_sentiment[2]))


if __name__ == '__main__':
    run_from_generative_script(None)

