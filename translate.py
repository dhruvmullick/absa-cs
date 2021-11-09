import requests, time, random
import pandas as pd
from tqdm import tqdm
from fake_useragent import UserAgent

tqdm.pandas()
ua = UserAgent()

def run2(txt, from_, to_):
    time.sleep(random.random())
    headers = {'User-Agent': ua.random}
    try:
        tr = requests.get(f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl={from_}&tl={to_}&q={txt}", headers=headers).json()["sentences"][0]["trans"]
    except:
        return ''
    return tr


def trans(df, from_, to_):
    df['texts_trans'] = df['sentences_texts'].progress_map(lambda txt: run2(txt, from_, to_))
    df['opinions_trans'] = df['sentences_opinions'].progress_map(lambda txt: ' <sep> '.join([run2(' '.join(item.split()[:-1]), from_, to_) + ' ' + item.split()[-1] for item in txt.split(' <sep> ')]))
    return df
    
def filter_unmatched(df):
    data = []
    for i, row in df.iterrows():
        tmp_opinions = []
        tmp_pos = []
        for opn in row['opinions_trans'].split(' <sep> '):
            if ' '.join(opn.split()[:-1]).strip().lower() in row['texts_trans'].lower():
                pos = row['texts_trans'].lower().index(' '.join(opn.split()[:-1]).strip().lower())
                tmp_opinions.append(opn)
                opn_len = len(' '.join(opn.split()[:-1]).strip().lower())
                tmp_pos.append(f'{pos}-{pos+opn_len}')
        if len(tmp_opinions):
            data.append([row['review_id'], row['sentences_ids'], row['texts_trans'], ' <sep> '.join(tmp_opinions), ' <sep> '.join(tmp_pos), row['sentences_texts'], row['sentences_opinions']])
    data = pd.DataFrame(data, columns=['review_id', 'sentences_ids', 'sentences_texts', 'sentences_opinions', 'opinions_pos', 'sentences_texts_org', 'sentences_opinions_org'])
    return data


if __name__ == '__main__':
    for from_, to_ in [('en', 'es'), ('en', 'ru'), ('es', 'en'), ('es', 'ru'), ('ru', 'en'), ('ru', 'es')]:
        print(from_, ': ', to_)

        # # Translating
        # train = pd.read_csv(f'/remote/cirrus-home/bghanem/projects/ABSA_LM/data/processed_val_Rest16_{from_}.csv')
        # train_trans = trans(train, from_, to_)
        # train_trans.to_csv(f'/remote/cirrus-home/bghanem/projects/ABSA_LM/data/processed_val_Rest16_{from_}_to_{to_}.csv', index=False)

        # Process and filter unmatched ones
        train_trans = pd.read_csv(f'/remote/cirrus-home/bghanem/projects/ABSA_LM/data/processed_val_Rest16_{from_}_to_{to_}.csv')
        print(train_trans.shape)
        train_trans = filter_unmatched(train_trans)
        print(train_trans.shape)
        # .sample(n=500, random_state=0)
        # train_trans.to_csv(f'/remote/cirrus-home/bghanem/projects/ABSA_LM/data/processed_val_Rest16_{from_}_to_{to_}_processed.csv', index=False)
