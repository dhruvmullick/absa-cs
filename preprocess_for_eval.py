import re
import pandas as pd


def clean_labels(txt):
    txt = re.sub(r'<extra.*\d>', '', txt).lower()
    return txt

def add_missed_sep(txt):
    txt = txt.split()
    missed_in = []
    for i in range(len(txt)):
        if txt[i].lower() in ['positive', 'negative', 'neutral'] and i < len(txt)-1:
            if txt[i+1] != '<sep>':
                missed_in.append(i+1)
    

    missed_in.reverse()
    for idx in missed_in:
        txt.insert(idx, '<sep>')
    
    # if len(missed_in):
    #     print(txt)
    #     print(missed_in)
    #     exit(1)
    return ' '.join(txt)

def split_SEP(txt):
    # clean unknown tags after sentiment (e.g., extra_id...)
    txt = clean_labels(txt)
    
    # if the word after pos/neg/neu not sep, then add it.
    txt = add_missed_sep(txt)
    
    # split on <sep>
    txt = txt.split('<sep>')

    return txt


df = pd.read_csv('models/combined/predictions.csv')[['pred', 'gold']]

# Prepare input
df['pred'] = df['pred'].map(lambda x: split_SEP(x))
df['gold'] = df['gold'].map(lambda x: split_SEP(x))

for i, row in df.iterrows():
    print(row['gold'])
    print(row['pred'])
    print('\n===============')

