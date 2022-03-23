# Importing libraries
import os, time, torch, datetime
from re import I
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import sys
import random

import aux_processor
import evaluate_e2e_tbsa
from utils import EarlyStopping
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rich.table import Column, Table
from rich import box
from rich.console import Console
from shutil import copyfile

# pd.set_option('display.max_colwidth', -1)

ABSA_PROMPT = "aspect analysis: "
# ABSA_PROMPT = ""

# define a rich console logger
console = Console(record=True)

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(0)  # pytorch random seed
np.random.seed(0)  # numpy random seed
torch.backends.cudnn.deterministic = True

# Setting up the device for GPU usage
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

lang_map = {'en': 'english', 'es': 'spanish', 'ru': 'russian'}


class YourDataSetClass(Dataset):
    """
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]
        if "other" in dataframe:
            self.other = self.data["other"].tolist()
        else:
            self.other = None

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.summ_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()

        other_attr = ''
        if self.other is not None:
            other_attr = str(self.other[index])

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'sentences_texts': source_text,
            'other': other_attr
        }


def train(tokenizer, model, device, loader, optimizer):
    """
    Function to be called for training with the parameters passed from main function
    """
    train_losses = []
    model.train()
    for _, data in tqdm(enumerate(loader, 0), total=len(loader), desc='Processing batches..'):
        y = data['target_ids'].to(device, dtype=torch.long)
        lm_labels = y.clone()
        lm_labels[y == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses


def generate(tokenizer, model, device, loader, model_params):
    """
  Function to evaluate model for spanbert-predictions

  """
    model.eval()
    predictions = []
    actuals = []
    data_list = []
    other_list = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):

            # When batch size is 1 and we seed seeds every time, we are ensuring that results are consistent for each
            # sentence irrespective of order in test set.

            torch.manual_seed(model_params['SEED'])  # pytorch random seed
            np.random.seed(model_params['SEED'])  # numpy random seed
            torch.cuda.manual_seed_all(model_params["SEED"])
            random.seed(model_params["SEED"])

            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(input_ids=ids, attention_mask=mask,
                                           max_length=256, do_sample=True, top_p=0.9, top_k=0, num_return_sequences=1)

            # generated_ids = model.generate(input_ids = ids, attention_mask = mask,
            #                                max_length=256, do_sample=True, top_p=0.9, top_k=0, num_return_sequences=1)
            # max_length=(int(sys.argv[1])), num_beams=(int(sys.argv[2])), length_penalty=(float(sys.a[3])), no_repeat_ngram_size=3, early_stopping=True)
            # max_length=256, num_beams=4, length_penalty=1.5, no_repeat_ngram_size=3, early_stopping=True)

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
            other_list.extend(data['other'])
            data_list.extend(data['sentences_texts'])

    return predictions, actuals, data_list, other_list


def T5Trainer(training_loader, validation_loader, tokenizer, model_params, local_model, task=None):

    """
    T5 trainer
    """

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.

    if local_model is not None:
        # logging
        console.log(f"""[Model]: Loading {local_model}...\n""")
        model = T5ForConditionalGeneration.from_pretrained(local_model)
    else:
        try:
            # logging
            console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")
            model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
        except ValueError:
            print("Loading model locally due to Connection Error...")
            model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL_LOCAL"])

    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=model_params["LEARNING_RATE"])
    optimizer = transformers.Adafactor(params=model.parameters(), lr=model_params["LEARNING_RATE"],
                                       scale_parameter=False, relative_step=False)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=model_params["early_stopping_patience"], verbose=True, delta=0.1,
                                   path=f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin')

    training_logger = Table(Column("Epoch", justify="center"), Column("train_loss", justify="center"),
                            Column("Val F1", justify="center"), Column("Epoch Time", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        start_time = time.time()
        train_losses = train(tokenizer, model, device, training_loader, optimizer)
        # valid_losses = validate(tokenizer, model, device, validation_loader)
        epoch_time = round(time.time() - start_time)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        # valid_loss = np.average(valid_losses)

        # preparing the processing time for the epoch and est. the total.
        epoch_time_ = str(datetime.timedelta(seconds=epoch_time))
        total_time_estimated_ = str(
            datetime.timedelta(seconds=(epoch_time * (model_params["TRAIN_EPOCHS"] - epoch - 1))))
        # training_logger.add_row(f'{epoch + 1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}', f'{valid_loss:.5f}',
        #                         f'{epoch_time_} (Total est. {total_time_estimated_})')
        # console.print(training_logger)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        # early_stopping(valid_loss, model)

        print("Early stopping: Calculating VALIDATION SCORE: ")
        prediction_file_name_validation = 'evaluation_predictions_val.csv'
        predictions_filepath_validation = '{}/{}'.format(model_params["OUTPUT_PATH"], prediction_file_name_validation)
        T5Generator(validation_loader, model_params=model_params, output_file=prediction_file_name_validation,
                    model=model, tokenizer=tokenizer)

        if task is None or task == aux_processor.COSMOS:
            validation_accuracy = evaluate_e2e_tbsa.evaluate_exact_match_for_columns(predictions_filepath_validation)
        elif task == aux_processor.SQUAD:
            validation_accuracy = aux_processor.evaluate_squad_predictions(predictions_filepath_validation)
        elif task == aux_processor.WIKITEXT:
            validation_accuracy = aux_processor.evaluate_lm_one_predictions(predictions_filepath_validation)
        else:
            raise AssertionError("Task Evaluation not defined")

        early_stopping(validation_accuracy, model)

        training_logger.add_row(f'{epoch + 1}/{model_params["TRAIN_EPOCHS"]}', f'{train_loss:.5f}',
                                f'{validation_accuracy:.5f}',
                                f'{epoch_time_} (Total est. {total_time_estimated_})')
        console.print(training_logger)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Saving the model after training
    path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    console.log(f"[Saving Model at {path}]...\n")
    console.log(f"[Replace best model with the last model]...\n")
    os.remove(f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin')
    # os.rename(f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin', f'{model_params["OUTPUT_PATH"]}/model_files/last_epoch_pytorch_model.bin')
    copyfile(f'{model_params["OUTPUT_PATH"]}/best_pytorch_model.bin',
             f'{model_params["OUTPUT_PATH"]}/model_files/pytorch_model.bin')
    console.print(f"""[Model] Model saved @ {os.path.join(model_params["OUTPUT_PATH"], "model_files")}\n""")


def T5Generator(data_loader, model_params, output_file, model=None, tokenizer=None):
    ### Setting random seed to 0 so that even if generation is run independently, we get the same results.
    ### Note: Running with CPU and running with GPU give different outcomes.

    torch.manual_seed(model_params['SEED'])  # pytorch random seed
    np.random.seed(model_params['SEED'])  # numpy random seed

    console.log(f"[Loading Model]...\n")
    # Saving the model after training

    if model and tokenizer:
        print("Using passed model and tokenizer")
    else:
        path = os.path.join(model_params["OUTPUT_PATH"], "model_files")
        print("Loading model: {}\n".format(path))
        model = T5ForConditionalGeneration.from_pretrained(path)
        tokenizer = T5Tokenizer.from_pretrained(path)

    model = model.to(device)

    # evaluating test dataset
    console.log(f"[Initiating Generation]...\n")
    for epoch in range(model_params["TEST_EPOCHS"]):
        predictions, actuals, data_list, other_list = generate(tokenizer, model, device, data_loader, model_params)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals,
                                 'Original Sentence': data_list, 'other': other_list})
        final_df.to_csv(os.path.join(model_params["OUTPUT_PATH"], output_file))

    console.save_text(os.path.join(model_params["OUTPUT_PATH"], 'logs.txt'))

    console.log(f"[Generation Completed.]\n")
    console.print(
        f"""[Generation] Generation on Test data saved @ {os.path.join(model_params["OUTPUT_PATH"], output_file)}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(model_params["OUTPUT_PATH"], 'logs.txt')}\n""")
