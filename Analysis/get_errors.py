import pandas as pd

INPUT_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/NonAmbiguousDataset9_ALSC/seed7_evaluation_predictions.csv'
PRONOUNS_LIST = [" it ", " its ", " he ", " him ", " his ", " she ", " her ",  " hers ", " they ", " them ", " their ", " there ", " which ", " who "]

df = pd.read_csv(INPUT_FILE_PATH)

for i, row in df.iterrows():
    if row['Generated Text']!=row['Actual Text']:
        sentence = row['Original Sentence']
        for pronoun in PRONOUNS_LIST:
            if pronoun in sentence:
                print(sentence)
                break