import pandas as pd

INPUT_FILE_PATH = '/data/merged_test_ambiguous_alsc_manual.csv'
OUTPUT_FILE_PATH = '/data/merged_test_ambiguous_alsc_manual_w_pronouns.csv'
PRONOUNS_LIST = [" it ", " its ", " he ", " him ", " his ", " she ", " her ",  " hers ", " they ", " them ", " their ", " there ", " which ", " who "]

df = pd.read_csv(INPUT_FILE_PATH)
df['pronouns'] = ""
row_list = []

for i, row in df.iterrows():
    pronouns_found = []
    for pronoun in PRONOUNS_LIST:
        if pronoun in row['sentences_texts']:
            pronouns_found.append(pronoun)
    row['pronouns'] = pronouns_found
    row_list.append(row)

new_df = pd.DataFrame(data=row_list, columns=df.columns)
new_df.to_csv(OUTPUT_FILE_PATH, index=False)


