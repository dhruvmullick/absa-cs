import pandas as pd

# PREDICTIONS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/baseline.csv'
# PREDICTIONS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/squad_0.5.csv'
PREDICTIONS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/t5-large/baseline_seed5.csv'
PRONOUNS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/data/merged_test_ambiguous_alsc_manual_w_pronouns.csv'

predictions_df = pd.read_csv(PREDICTIONS_FILE_PATH)
pronouns_df = pd.read_csv(PRONOUNS_FILE_PATH)

AMBIGUOUS_CASES = ["it", "its", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "there", "which",
                   "who"]
incorrect_dict = {}
correct_dict = {}

for pronoun in AMBIGUOUS_CASES:
    correct_dict[pronoun] = 0
    incorrect_dict[pronoun] = 0

for i, row in predictions_df.iterrows():
    sentence = row["Original Sentence"]
    sentence = sentence.replace("get sentiment: ", "")
    correct = row["Generated Text"] == row["Actual Text"]
    pronouns = pronouns_df[pronouns_df["sentences_texts"] == sentence]["pronouns"].tolist()[0].split('/')

    if correct:
        for pronoun in pronouns:
            correct_dict[pronoun] += 1
    else:
        for pronoun in pronouns:
            incorrect_dict[pronoun] += 1

accuracy_dict = {}
t=0
total = {}
for k in correct_dict.keys():
    accuracy_dict[k] = round(100 * correct_dict[k] / (correct_dict[k] + incorrect_dict[k] + 0.000001), 2)
    total[k] = correct_dict[k] + incorrect_dict[k]
    t+=total[k]

print(f"Correct: {correct_dict}")
print(f"Incorrect: {incorrect_dict}")
print(f"Accuracy: {accuracy_dict}")
print(f"Total: {total}")
