import random
from re import search
import pandas as pd

FILE = '../data/processed_train_Mams_en.csv'
FILE_OUTPUT_AMBI = '../data/MAMS_ATSA/Mams_ambi/processed_train_Mams_en_ambi_segregated.csv'
FILE_OUTPUT_NON_AMBI = '../data/MAMS_ATSA/Mams_ambi/processed_train_Mams_en_non_ambi_segregated.csv'

AMBIGUOUS_CASES = [" it ", " its ", " he ", " him ", " his ", " she ", " her ", " hers ",
                   " they ", " them "]
REGEX_PHRASE = '|'.join(AMBIGUOUS_CASES)
random.seed(0)

df = pd.read_csv(FILE)
df_ambi = df[df['sentences_texts'].str.contains(REGEX_PHRASE)]
df_not_ambi = df[~df['sentences_texts'].str.contains(REGEX_PHRASE)]

df_ambi.to_csv(FILE_OUTPUT_AMBI, index=False)
df_not_ambi.to_csv(FILE_OUTPUT_NON_AMBI, index=False)




