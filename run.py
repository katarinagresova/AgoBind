import comet_ml
from models import get_dnabert
from eval import get_test_score, compute_pr_curve, get_predictions_and_labels
from transformers import TrainingArguments
from log_utils import log_extra
from training import get_trained_model

config = {
    "train_data" : 'data/train_set_1_1_CLASH2013_paper.tsv',
    "test_data": 'data/evaluation_set_1_1_CLASH2013_paper.tsv',
    "eval_dset_ratio" : 0.2, #Deducted from the train set
    "batch_size" : 32,
    "gradient_accumulation_steps":4,
    "eval_steps" : 100,
    "freeze":False ,
    "layers_to_unfreeze":None,
    "random_weights":False,
    "kmer_len" : 6,
    "stride" : 1,
    "early_stopping_patience" : 2, 
    "learning_rate" : 2e-4,
    "weight_decay":0.01,
    "backbone":get_dnabert, 
}

args = TrainingArguments(output_dir="output_checkpoints",
                        learning_rate=config['learning_rate'],
                        weight_decay=config['weight_decay'], 
                        num_train_epochs=500, 
                        per_device_train_batch_size=config['batch_size'],
                        per_device_eval_batch_size=config['batch_size'],
                        do_train=True,
                        do_eval=True,
                        logging_steps=10000,
                        warmup_steps=5000, 
                        eval_steps=config['eval_steps'],
                        evaluation_strategy="steps",
                        logging_strategy="steps",
                        logging_first_step=True,
                        load_best_model_at_end=True,
                        save_steps=100, 
                        save_total_limit=5,
                        gradient_accumulation_steps=config['gradient_accumulation_steps'],
                        metric_for_best_model="eval_loss"
                        )


comet_ml.init(project_name='dnabert_for_clash', api_key='3NQhHgMmmlfnoqTcvkG03nYo9')

model, tokenizer = config['backbone'](config) 
trained_model, encoded_samples_test = get_trained_model(config, args, model, tokenizer)

predictions, labels = get_predictions_and_labels(config['test_data'], encoded_samples_test, trained_model)
f1_score_test = get_test_score(predictions, labels)
recall, precision = compute_pr_curve(predictions, labels)
log_extra(config, f1_score_test, recall, precision)

print('ALL DONE')