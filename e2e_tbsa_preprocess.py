import csv
import utils

PREDICTIONS_FILE = 'models/combined/predictions.csv'
TRANSFORMED_TARGETS_PREDICTIONS_FILE = 'models/combined/transformed-targets.csv'
SEPARATOR = '<sep>'


def get_sentence_matched_with_targets(aspects_targets, real_sentence):
    real_target_idx = []
    for tg in aspects_targets:
        idx_start = real_sentence.find(tg)
        if idx_start == -1:
            pass
        idx_end = idx_start + len(tg) - 1
        real_target_idx.append((idx_start, idx_end))
    return real_target_idx


def transform_line_for_target_extraction(line):
    generated_polarities, true_polarities = utils.get_polarities_for_line(line)
    real_sentence = utils.normalise_sentence(line[3].strip())

    generated_aspects_targets = utils.get_aspect_targets(generated_polarities)
    true_aspects_targets = utils.get_aspect_targets(true_polarities)

    generated_target_idx_list = get_sentence_matched_with_targets(generated_aspects_targets, real_sentence)
    true_target_idx_list = get_sentence_matched_with_targets(true_aspects_targets, real_sentence)

    return generated_target_idx_list, true_target_idx_list


def transform_gold_and_truth():
    with open(PREDICTIONS_FILE, 'r') as csvfile:
        with open(TRANSFORMED_TARGETS_PREDICTIONS_FILE, 'w') as newfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip the headers
            # writer = csv.writer(newfile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
            writer = csv.writer(newfile)
            writer.writerow(["Predicted idx", "Gold idx"])
            for line in reader:
                pred_transformed, gold_transformed = transform_line_for_target_extraction(line)
                writer.writerow([pred_transformed, gold_transformed])


transform_gold_and_truth()
