import pandas as pd

BASELINE_PREDICTIONS_FILE = 'Results/AmbiguousDataset6/squad/Predictions/0.01_8_predictions.csv'
COMMONSENSE_PREDICTIONS_FILE = 'Results/AmbiguousDataset6/commongen/Predictions/0.05_8_predictions.csv'
BASELINE_IDX_FILE = 'Results/AmbiguousDataset6/squad/Predictions/0.01_8_sentiments_idx.csv'
COMMONSENSE_IDX_FILE = 'Results/AmbiguousDataset6/commongen/Predictions/0.05_8_sentiments_idx.csv'

COMPARISON_OUTPUT_FILE = 'Results/AmbiguousDataset6/squad/ComparisonCommonGen/comparisons.csv'


def read_idx_as_dict(file_name):
    dict = {}
    with open(file_name, 'r') as file:
        for line in file:
            dict[line.strip()] = True
    return dict


if __name__ == '__main__':
    baseline_idx_dict = read_idx_as_dict(BASELINE_IDX_FILE)
    commonsense_idx_dict = read_idx_as_dict(COMMONSENSE_IDX_FILE)
    merged_idx_list = {**baseline_idx_dict, **commonsense_idx_dict}
    merged_idx_list = [int(x) for x in merged_idx_list]
    merged_idx_list.sort()

    baseline_predictions_df = pd.read_csv(BASELINE_PREDICTIONS_FILE)
    commonsense_predictions_df = pd.read_csv(COMMONSENSE_PREDICTIONS_FILE)

    comparison_list = []

    for idx in merged_idx_list:

        if str(idx) in baseline_idx_dict:
            original_sentence = baseline_predictions_df[baseline_predictions_df['idx'] == idx]['Original Sentence'].values[0]
            actual_answer = baseline_predictions_df[baseline_predictions_df['idx'] == idx]['Actual Text'].values[0]
            baseline_sentence_incorrect \
                = baseline_predictions_df[baseline_predictions_df['idx'] == idx]['Generated Text'].values[0]
        else:
            baseline_sentence_incorrect = ''

        if str(idx) in commonsense_idx_dict:
            original_sentence = commonsense_predictions_df[commonsense_predictions_df['idx'] == idx]['Original Sentence'].values[0]
            actual_answer = commonsense_predictions_df[commonsense_predictions_df['idx'] == idx]['Actual Text'].values[0]
            commonsense_sentence_incorrect \
                = commonsense_predictions_df[commonsense_predictions_df['idx'] == idx]['Generated Text'].values[0]
        else:
            commonsense_sentence_incorrect = ''

        comparison_list.append([idx, original_sentence, actual_answer,
                                baseline_sentence_incorrect, commonsense_sentence_incorrect])

    comparison_df = pd.DataFrame(comparison_list,
                                 columns=['idx', 'Original Sentence', 'Actual Answer', 'Baseline Sentence Incorrect',
                                          'Commonsense Sentence Incorrect'])

    comparison_df.to_csv(COMPARISON_OUTPUT_FILE, index=False)




