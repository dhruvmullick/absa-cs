import csv
import utils
import sys

PREDICTIONS_FILE = 'models/combined/predictions.csv'
# PREDICTIONS_FILE = 'models/combined/predictions_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/combined/transformed-targets.csv'
TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'models/combined/transformed-sentiments.csv'
# TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/combined/transformed-targets_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
# TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE = 'models/combined/transformed-sentiments_{}_{}_{}.csv'.format(sys.argv[1], sys.argv[2], sys.argv[3])
SEPARATOR = '<sep>'

sentiment_to_identifier = {
    'positive': 'POS',
    'negative': 'NEG',
    'neutral': 'NEU'
}


def get_sentiment(polarity_list):
    return [polarity.split()[-1] for polarity in polarity_list]


def get_sentence_matched_with_targets(aspects_targets, real_sentence):
    real_target_idx = []
    for tg in aspects_targets:
        idx_start = real_sentence.find(tg)
        if idx_start == -1:
            pass
        idx_end = idx_start + len(tg) - 1
        real_target_idx.append((idx_start, idx_end))
    return real_target_idx


def get_sentence_matched_with_targets_and_sentiments(aspects_targets, aspects_sentiments, real_sentence):
    real_idx = []
    for i, tg in enumerate(aspects_targets):
        if aspects_sentiments[i] not in sentiment_to_identifier.keys():
            # print("no sentiment found in {}".format(real_sentence))
            continue
        idx_start = real_sentence.find(tg)
        if idx_start == -1:
            continue
        idx_end = idx_start + len(tg) - 1
        real_idx.append((idx_start, idx_end, sentiment_to_identifier[aspects_sentiments[i]]))

    return real_idx


def transform_line_for_target_extraction(line):
    generated_polarities, true_polarities = utils.get_polarities_for_line(line)
    real_sentence = utils.normalise_sentence(line[3].strip())

    generated_aspects_targets = utils.get_aspect_targets(generated_polarities)
    true_aspects_targets = utils.get_aspect_targets(true_polarities)

    generated_target_idx_list = list(set(get_sentence_matched_with_targets(generated_aspects_targets, real_sentence)))
    true_target_idx_list = list(set(get_sentence_matched_with_targets(true_aspects_targets, real_sentence)))

    return generated_target_idx_list, true_target_idx_list


def transform_line_for_sentiment_extraction(line):
    generated_polarities, true_polarities = utils.get_polarities_for_line(line)
    real_sentence = utils.normalise_sentence(line[3].strip())

    generated_aspects_targets = utils.get_aspect_targets(generated_polarities)
    true_aspects_targets = utils.get_aspect_targets(true_polarities)

    generated_aspects_sentiments = get_sentiment(generated_polarities)
    true_aspects_sentiments = get_sentiment(true_polarities)

    generated_sentiment_idx_list = get_sentence_matched_with_targets_and_sentiments(generated_aspects_targets,
                                                                                    generated_aspects_sentiments,
                                                                                    real_sentence)
    true_sentiment_idx_list = get_sentence_matched_with_targets_and_sentiments(true_aspects_targets,
                                                                               true_aspects_sentiments, real_sentence)

    return generated_sentiment_idx_list, true_sentiment_idx_list


def transform_gold_and_truth():
    with open(PREDICTIONS_FILE, 'r') as csvfile:
        with open(TRANSFORMED_TARGETS_PREDICTIONS_FILE, 'w') as newfile_targets:
            with open(TRANSFORMED_SENTIMENTS_PREDICTIONS_FILE, 'w') as newfile_sentiments:
                reader = csv.reader(csvfile)
                next(reader, None)  # skip the headers
                writer_targets = csv.writer(newfile_targets)
                writer_targets.writerow(["Predicted idx", "Gold idx"])
                writer_sentiments = csv.writer(newfile_sentiments)
                writer_sentiments.writerow(["Predicted sentiment tags", "Gold sentiment tags"])
                for line in reader:
                    pred_transformed, gold_transformed = transform_line_for_target_extraction(line)
                    writer_targets.writerow([pred_transformed, gold_transformed])
                    pred_sentiment_transformed, gold_sentiment_transformed = transform_line_for_sentiment_extraction(
                        line)
                    writer_sentiments.writerow([pred_sentiment_transformed, gold_sentiment_transformed])


transform_gold_and_truth()

