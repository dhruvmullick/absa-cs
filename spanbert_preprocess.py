import csv
import json
import os

PREDICTIONS_FILE = 'spanbert-predictions/train_spanbert_{}.csv/test_spanbert_{}.csv/predictions.json'
TRANSFORMED_PREDICTIONS_FILE ='spanbert-predictions-transformed/train_spanbert_{}.csv/test_spanbert_{}.csv/predictions.json'
GEN_DATA_FILE = 'data/processed_test_{}.csv'

def get_language(data_file):
    return data_file.split("_")[-1].split(".")[0]

def get_predictions(predictions_file):
    predicted_sentences = []
    with open(predictions_file, 'r') as jsonfile:
        data = json.load(jsonfile)
        for key, item in data.items():
            pred_terms = item['pred_terms']
            pred_polarities = item['pred_polarities']
            sentence_for_term = []
            if len(pred_terms) == 0:
                predicted_sentences.append("")
                continue
            for term,polarity in zip(pred_terms, pred_polarities):
                sentence_for_term.append('{} {}'.format(term, polarity))
            sentence_for_term = ' <sep> '.join(sentence_for_term)
            sentence_for_term = sentence_for_term.strip()
            predicted_sentences.append(sentence_for_term)
    return predicted_sentences

def transform_predicted_sentences(transformed_predictions_file, gen_data_file, predicted_sentences):
    with open(gen_data_file, 'r') as gen_file:
        os.makedirs(os.path.dirname(transformed_predictions_file), exist_ok=True)
        with open(transformed_predictions_file, 'w') as transformed_file:
            reader = csv.reader(gen_file)
            writer = csv.writer(transformed_file)
            writer.writerow([" ","Generated Text","Actual Text","Original Sentence"])
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                actual_text = line[3].strip()
                original_sent = line[2].strip()
                writer.writerow([i, predicted_sentences[i-1], actual_text, original_sent])


datasets = ['Rest16_en', 'Rest16_es', 'Rest16_ru', 'Lap14_en', 'Mams_en', 'Mams_short_en']

for dtrain in datasets:
    for dtest in datasets:
        predicted_sentences = get_predictions(PREDICTIONS_FILE.format(dtrain, dtest))
        transform_predicted_sentences(TRANSFORMED_PREDICTIONS_FILE.format(dtrain, dtest),
                                      GEN_DATA_FILE.format(dtest), predicted_sentences)

