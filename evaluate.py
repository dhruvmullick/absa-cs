import csv
from utils import *

PREDICTIONS_FILE = 'models/combined/predictions.csv'
TRUE_POS, TRUE_NEG, FALSE_POS, FALSE_NEG = 'tp', 'tn', 'fp', 'fn'


def add_to_dest(dest, src):
    for k in src.keys():
        dest[k] += src[k]


def get_sentiment(polarity):
    return polarity.split()[-1]


def get_other_sentiments(polarity):
    all_sentiments = [POSITIVE, NEUTRAL, NEUTRAL]
    all_sentiments.remove(get_sentiment(polarity))
    return all_sentiments


def get_stats_dict(dictionary, polarity_phrase):
    return dictionary[polarity_phrase.split()[-1]]


def is_negative(polarity_phrase):
    return polarity_phrase.endswith(NEGATIVE)


def is_positive(polarity_phrase):
    return polarity_phrase.endswith(POSITIVE)


def is_neutral(polarity_phrase):
    return polarity_phrase.endswith(NEUTRAL)


# def get_true_polarities(sentence):
#     polarities = sentence.split(SEPARATOR)
#     polarities = [normalise_sentence(p).strip() for p in polarities]
#     return polarities


def check_for_unequal_word_overlap(phrase_to_check, original_phrase):
    phrase_to_check_tokens = phrase_to_check.split()
    phrases_tokens = original_phrase.split()
    if len(phrases_tokens) == len(phrase_to_check_tokens):
        return False
    else:
        for word in phrases_tokens:
            if word in phrase_to_check_tokens:
                return True
    return False


def check_for_unequal_word_overlap_for_many_phrases(phrase_to_check, original_phrases):
    for original_phrase in original_phrases:
        if check_for_unequal_word_overlap(phrase_to_check, original_phrase):
            return True, original_phrase
    else:
        return False, None


# def get_reverse_accuracy_for_line(line):
#     generated_polarities, true_polarities, original_sentence_length = get_polarities_and_len(line)
#     pos = 0
#     neg = 0
#     for polarity in true_polarities:
#         if polarity in generated_polarities:
#             # Loop hole - if generated once, but there are multiple in true polarities then we'll still treat it as
#             # correct.
#             pos += 1
#         else:
#             neg += 1
#     return pos, neg
#
#
# def get_reverse_accuracy():
#     # ACCURACY FOR CHECKING FROM TRUE TO GEN - 26%
#     pos_total, neg_total = 0, 0
#     with open(PREDICTIONS_FILE, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         for line in reader:
#             pos, neg = get_reverse_accuracy_for_line(line)
#             pos_total += pos
#             neg_total += neg
#     accuracy = pos_total / (pos_total + neg_total)
#     print(accuracy)


# Basically words are being classified as Aspect or Not Aspect.
# Consider words that are not classified by Generator as being Not Aspect.
def get_metrics_for_line_target_word(line):
    generated_polarities, true_polarities = get_polarities_for_line(line)
    generated_aspects_targets = get_aspect_targets(generated_polarities)
    true_aspects_targets = get_aspect_targets(true_polarities)

    tp = 0
    for g in generated_aspects_targets:
        if g in true_aspects_targets:
            tp += 1

    return tp, len(generated_aspects_targets), len(true_aspects_targets)


# def get_precision_recall_for_aspect_words():
#     tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0
#     with open(PREDICTIONS_FILE, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         for line in reader:
#             tp, fp, tn, fn = get_metrics_for_line_target_word(line)
#             tp_total += tp
#             fp_total += fp
#             tn_total += tn
#             fn_total += fn
#     accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)
#     precision = tp_total / (tp_total + fp_total)
#     recall = tp_total / (tp_total + fn_total)
#     return accuracy, precision, recall


def get_precision_recall_for_aspect_words():
    tp_total, gen_len_total, true_len_total = 0, 0, 0
    with open(PREDICTIONS_FILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            tp, gen_len, true_len = get_metrics_for_line_target_word(line)
            tp_total += tp
            gen_len_total += gen_len
            true_len_total += true_len
    precision = tp_total / gen_len_total
    recall = tp_total / true_len_total
    return precision, recall, 2*precision*recall/(precision + recall)


# def get_metrics_for_line_aspect_polarity(line):
#     generated_polarities, true_polarities, _ = get_polarities_and_len(line)
#     generated_aspects_targets = get_aspect_targets(generated_polarities)
#     true_aspects_targets = get_aspect_targets(true_polarities)
#     copy_true_aspect_targets = true_aspects_targets[:]
#     copy_true_polarities = true_polarities[:]
#
#     stats = {POSITIVE: {}, NEGATIVE: {}, NEUTRAL: {}}
#     # Note that size of generated aspects and true aspects could be different
#     for g_target, g_polarity in zip(generated_aspects_targets, generated_polarities):
#         g_sentiment = get_sentiment(g_polarity)
#         other_g_sentiments = get_other_sentiments(g_polarity)
#         if g_target in copy_true_aspect_targets:
#             if g_polarity in copy_true_polarities:
#                 stats[g_sentiment][TRUE_POS] += 1
#                 stats[other_g_sentiments][TRUE_NEG] += 1
#                 copy_true_aspect_targets.remove(g_target)
#                 copy_true_polarities.remove(g_polarity)
#             else:
#                 stats[g_sentiment][FALSE_POS] += 1
#
#         else:
#             stats[g_sentiment][FALSE_POS] += 1
#
#     for t_target, t_polarity in zip(copy_true_aspect_targets, copy_true_polarities):
#         sentiment = get_sentiment(t_polarity)


    # fn = len(copy_true_aspect_targets)
    # tn = len(true_aspects_targets) - len(copy_true_aspect_targets)
    # return tp, fp, tn, fn


# def get_precision_recall_for_sentiments():
#     stats = {POSITIVE: {}, NEGATIVE: {}, NEUTRAL: {}}
#     with open(PREDICTIONS_FILE, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         for line in reader:
#             pos_stats, neg_stats, neutral_stats = get_metrics_for_line_aspect_polarity(line)
#             add_to_dest(stats[POSITIVE], pos_stats)
#             add_to_dest(stats[NEGATIVE], neg_stats)
#             add_to_dest(stats[NEUTRAL], neutral_stats)
#
#     # accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)
#     # precision = tp_total / (tp_total + fp_total)
#     # recall = tp_total / (tp_total + fn_total)
#     # return accuracy, precision, recall


print("Precision, Recall, F1 = " + str(get_precision_recall_for_aspect_words()))
