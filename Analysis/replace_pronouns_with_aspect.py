import pandas as pd
import re

INPUT_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/data/merged_test_ambiguous_alsc_manual_w_pronouns.csv'
OUTPUT_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/data/merged_test_ambiguous_alsc_manual_w_pronouns_replaced.csv'
MISMATCH_FILE_PATH = '/tmp/multiple_matches.csv'

pronouns_df = pd.read_csv(INPUT_FILE_PATH)
rows = []
mismatch_rows = []

for i, row in pronouns_df.iterrows():
    sentence = row["sentences_texts"]
    pos = sentence.find("aspect: ")
    assert pos != -1
    aspect_phrase = sentence[pos:].strip()
    aspect = aspect_phrase.split()[-1]
    sentence = sentence[:pos].strip()
    pronoun_list = row["pronouns"].split('/')
    for pronoun in pronoun_list:
        regex_pattern = f' {pronoun}(\\.|,|;| )'
        regex_matches = re.findall(regex_pattern, sentence)
        matches = len(regex_matches)
        if matches != 1:
            mismatch_rows.append([row['review_id'], row['sentences_ids'], sentence + ' ' + aspect_phrase, row['sentences_opinions'], pronoun])
            break
        sentence_new = re.sub(regex_pattern, f' {aspect} ', sentence) + ' ' + aspect_phrase
        rows.append([row['review_id'], row['sentences_ids'], sentence_new, row['sentences_opinions']])

df = pd.DataFrame(data=rows, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions'])
df.to_csv(OUTPUT_FILE_PATH, index=False)

mismatch_df = pd.DataFrame(data=mismatch_rows, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions', 'pronoun'])
mismatch_df.to_csv(MISMATCH_FILE_PATH, index=False)
