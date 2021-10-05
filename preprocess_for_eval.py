import enum
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

    # normalize text
    txt = list(set([item.strip() for item in txt]))

    return txt

def aspect_scores(gold, pred):
    """

    I am not sure if this code is correct. 
    I wrote it quickly to test the results on the aspects
    
    """
    gold = [[' '.join(item.split()[:-1]) for item in row] for row in gold]
    pred = [[' '.join(item.split()[:-1]) for item in row] for row in pred]
    
    hit_ = 0
    for i, g in enumerate(gold):
        for p in pred[i]:
            if p in g:
                hit_ += 1
    prec = hit_ / len([item for sublist in pred for item in sublist])
    rec = hit_ / len([item for sublist in gold for item in sublist])
    f1 = 2 * prec * rec / (prec + rec)
    print('precision:  ', prec)
    print('recall:  ', rec)
    print('f1: ', f1)


df = pd.read_csv('models/combined/predictions.csv')[['pred', 'gold']]

# Prepare input
df['pred'] = df['pred'].map(lambda x: split_SEP(x))
df['gold'] = df['gold'].map(lambda x: split_SEP(x))

# printing some examples:
for i, row in df.sample(n=5).iterrows():
    print(row['gold'])
    print(row['pred'])
    print('\n===============')

# calculate scores
aspect_scores(df['gold'].tolist(), df['pred'].tolist())

