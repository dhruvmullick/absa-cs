### Adapted from
# https://github.com/lixin4ever/E2E-TBSA/blob/8cb7182d890d2b36766b23c788bb82c265ca4242/evals.py#L98

import numpy as np
import csv
import ast
import sys
import pandas as pd
import sklearn.metrics

SMALL_POSITIVE_CONST = 1e-4

# # TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'Results/AmbiguousDataset5/Predictions/evaluation_commongen_predictions_0.0_1_0_transformed_targets.csv'
# # TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'Results/AmbiguousDataset5/Predictions/evaluation_commongen_predictions_0.0_1_0_transformed_sentiments.csv'
# TRANSFORMED_TARGETS_PREDICTIONS_FILE = ''
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = ''
# TRANSFORMED_TARGETS_PREDICTIONS_FILE_PATH = 'Results/AmbiguousDataset6/Predictions/temp_0.2_8_transformed_targets.csv'
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_PATH = 'Results/AmbiguousDataset6/Predictions/temp_0.2_8_transformed_sentiments.csv'
TRANSFORMED_TARGETS_PREDICTIONS_FILE_PATH = 'Results/AmbiguousDataset6/commongen/Predictions/temp_baseline_target_idx.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_PATH = 'Results/AmbiguousDataset6/commongen/Predictions/temp_baseline_sentiments_idx.csv'

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
        n_hit_ot = match_ot(gold_ote_sequence=g_ot_sequence, pred_ote_sequence=p_ot_sequence, idx=i)
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

def match_ot(gold_ote_sequence, pred_ote_sequence, idx):
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

    # if n_hit != len(pred_ote_sequence) or n_hit != len(gold_ote_sequence):
    #     print("{}".format(idx))

    return n_hit

# Dhruv's e.g. Updated: print(evaluate_ts([[(1, 3, 'POS'), (4, 4, 'NEG')]], [[(1, 3, 'POS'), (4, 4, 'NEG')]]))
def evaluate_ts(gold_ts, pred_ts, neutral_ignore = True):
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
                                                              pred_ts_sequence=p_ts_sequence, idx=i,
                                                              neutral_ignore=neutral_ignore)

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


def match_ts(gold_ts_sequence, pred_ts_sequence, idx, neutral_ignore):
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

    if neutral_ignore:
        hit_count[2] = 0
        gold_count[2] = 0
        pred_count[2] = 0

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


def get_stats_alc(prediction_file_to_evaluate, polarity):
    df = pd.read_csv(prediction_file_to_evaluate)
    tp = len(df[(df['Generated Text'] == polarity) & (df['Actual Text'] == polarity)])
    fp = len(df[(df['Generated Text'] == polarity) & ~(df['Actual Text'] == polarity)])
    fn = len(df[~(df['Generated Text'] == polarity) & (df['Actual Text'] == polarity)])
    return tp, fp, fn


def run_from_generative_script(target_file_to_evaluate, sentiments_file_to_evaluate, neutral_ignore=True):

    print("Evaluating target file: {}, Evaluating sentiments file: {}\n"
          .format(target_file_to_evaluate, sentiments_file_to_evaluate))

    predicted_data_targets, gold_data_targets = read_transformed_targets(target_file_to_evaluate)
    output_targets = evaluate_ote(gold_data_targets, predicted_data_targets)
    print(output_targets)

    print("--------")

    predicted_data_sentiment, gold_data_sentiment = read_transformed_sentiments(sentiments_file_to_evaluate)
    output_sentiment = evaluate_ts(gold_data_sentiment, predicted_data_sentiment, neutral_ignore)
    print(output_sentiment)

    return {'te': output_targets[2], 'tse': output_sentiment[2]}


def run_from_generative_script_alsc(prediction_file_to_evaluate):

    print("Evaluating ALSC: {}\n".format(prediction_file_to_evaluate))

    df = pd.read_csv(prediction_file_to_evaluate)
    y_true = df['Actual Text'].to_numpy()
    y_pred = df['Generated Text'].to_numpy()
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    print(f1)


    # #### FP = FN in a multi class setting. Therefore Micro P = Micro R = Micro F = Accuracy
    #
    # TP, FP, FN = [], [], []
    # for polarity in ["positive", "negative", "neutral"]:
    #     tp, fp, fn = get_stats_alc(prediction_file_to_evaluate, polarity)
    #     TP.append(tp)
    #     FP.append(fp)
    #     FN.append(fn)
    #
    # micro_precision = np.sum(TP)/(SMALL_POSITIVE_CONST + np.sum(TP) + np.sum(FP))
    # micro_recall = np.sum(TP)/(SMALL_POSITIVE_CONST + np.sum(TP) + np.sum(FN))
    # micro_f1 = 2 * micro_precision * micro_recall / (SMALL_POSITIVE_CONST + micro_precision + micro_recall)
    #
    # print(micro_precision, micro_recall, micro_f1)
    #
    # return micro_f1

def evaluate_exact_match_for_columns(predictions_filepath):
    predictions_df = pd.read_csv(predictions_filepath)
    correct = predictions_df["Generated Text"].apply(str) == predictions_df["Actual Text"].apply(str)
    acc = 100*correct.sum()/len(predictions_df)
    return acc


if __name__ == '__main__':
    print(evaluate_exact_match_for_columns('alc_prediction.csv'))
    # run_from_generative_script(TRANSFORMED_TARGETS_PREDICTIONS_FILE_PATH, TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE_PATH)