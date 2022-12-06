import pandas as pd

QQP_PREDICTIONS_FILE = '../Results/AmbiguousDataset8_ALSC/t5-large/qqp_0.5_seed9.csv'
COMMONGEN_PREDICTIONS_FILE = '../Results/AmbiguousDataset8_ALSC/t5-large/commongen_0.1_seed7.csv'
TEST_DATASET = '/Users/dhruvmullick/Projects/absa-cs/data/Ambiguous8/merged_test_ambiguous_alsc_manual.csv'

test_df = pd.read_csv(TEST_DATASET)
mams_data = []
rest_data = []

for i, row in test_df.iterrows():
    if row['review_id'] == '0':
        mams_data.append('get sentiment: ' + row['sentences_texts'])
    else:
        rest_data.append('get sentiment: ' + row['sentences_texts'])

qqp_results_df = pd.read_csv(QQP_PREDICTIONS_FILE)
cg_results_df = pd.read_csv(COMMONGEN_PREDICTIONS_FILE)

mams_correct = 0
rest_correct = 0
for i, row in qqp_results_df.iterrows():
    if row['Original Sentence'] in mams_data:
        if row['Generated Text'] == row['Actual Text']:
            mams_correct+=1
    elif row['Original Sentence'] in rest_data:
        if row['Generated Text'] == row['Actual Text']:
            rest_correct += 1

print(f"QQP Results -> MAMS: {mams_correct}, REST: {rest_correct}")

mams_correct = 0
rest_correct = 0
for i, row in cg_results_df.iterrows():
    if row['Original Sentence'] in mams_data:
        if row['Generated Text'] == row['Actual Text']:
            mams_correct+=1
    elif row['Original Sentence'] in rest_data:
        if row['Generated Text'] == row['Actual Text']:
            rest_correct += 1

print(f"CG Results -> MAMS: {mams_correct}, REST: {rest_correct}")
