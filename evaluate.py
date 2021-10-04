import csv
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PREDICTIONS_FILE = '/home/mullick/scratch/ABSA_LM/models/combined-dhruv/predictions.csv'
SEPARATOR = '<sep>'
POSITIVE, NEGATIVE, NEUTRAL = 'positive', 'negative', 'neutral'
TRUE_POS, TRUE_NEG, FALSE_POS, FALSE_NEG = 'tp', 'tn', 'fp', 'fn'

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def add_to_dest(dest, src):
    for k in src.keys():
        dest[k] += src[k]


def get_aspect_targets(polarity_sentence):
    return [' '.join(polarity.split()[:-1]) for polarity in polarity_sentence]


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


def normalise_sentence(sentence):
    sentence = sentence.replace(',', '')
    sentence = sentence.replace('.', '')
    sentence = sentence.replace('\"', '')
    sentence = sentence.lower()
    tokenised_sentence = sentence.split(" ")
    return ' '.join([lemmatizer.lemmatize(w) for w in tokenised_sentence if w not in stop_words])


def get_generated_polarities(sentence):
    polarities = sentence.split(SEPARATOR)
    polarities = [normalise_sentence(p) for p in polarities]
    polarities = [p.strip() for p in polarities if not p.strip().startswith((NEGATIVE, POSITIVE, NEUTRAL))]
    return polarities


def get_true_polarities(sentence):
    polarities = sentence.split(SEPARATOR)
    polarities = [normalise_sentence(p).strip() for p in polarities]
    return polarities


def get_polarities_and_len(line):
    generated_sentence = line[1].strip()
    true_sentence = line[2].strip()
    original_sentence_length = len([normalise_sentence(p).strip() for p in line[3].strip()])
    generated_polarities = get_generated_polarities(generated_sentence)
    true_polarities = get_true_polarities(true_sentence)
    return generated_polarities, true_polarities, original_sentence_length


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
    generated_polarities, true_polarities, original_sentence_length = get_polarities_and_len(line)
    generated_aspects_targets = get_aspect_targets(generated_polarities)
    true_aspects_targets = get_aspect_targets(true_polarities)
    copy_true_aspect_targets = true_aspects_targets[:]

    tp, fp, tn, fn = 0, 0, 0, 0
    # Note that size of generated aspects and true aspects could be different
    for g in generated_aspects_targets:
        if g in copy_true_aspect_targets:
            tp += 1
            copy_true_aspect_targets.remove(g)
        else:
            fp += 1

    fn = len(copy_true_aspect_targets)
    tn = len(true_aspects_targets) - len(copy_true_aspect_targets)
    return tp, fp, tn, fn


def get_precision_recall_for_aspect_words():
    tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0
    with open(PREDICTIONS_FILE, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            tp, fp, tn, fn = get_metrics_for_line_target_word(line)
            tp_total += tp
            fp_total += fp
            tn_total += tn
            fn_total += fn
    accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)
    precision = tp_total / (tp_total + fp_total)
    recall = tp_total / (tp_total + fn_total)
    return accuracy, precision, recall


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


# get_reverse_accuracy()
print("Accuracy, Precision, Recall = " + str(get_precision_recall_for_aspect_words()))
