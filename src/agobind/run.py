import comet_ml
from agobind.models import get_dnabert
from agobind.eval import get_f1_score, compute_pr_curve, get_probs_and_labels
from transformers import TrainingArguments
from agobind.log_utils import log_extra
from agobind.training import get_trained_model
from agobind.preprocessing import get_preprocessed_datasets
import sys



{
    "train_data" : sys.argv[1],
    "test_data": sys.argv[2],
    "eval_dset_ratio" : 0.2, #Deducted from the train set
    "batch_size" : 64,
    "gradient_accumulation_steps":4,
    "eval_steps" : 100,
    "freeze":False ,
    "layers_to_unfreeze":None,
    "random_weights":False,
    "kmer_len" : 6,
    "stride" : 1,
    "early_stopping_patience" : 5, 
    "learning_rate" : 2e-4,
    "weight_decay":0.01,
    "backbone":get_dnabert, 
}

def run(config):
    config = sys.argv[1]
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
    trainer, encoded_samples_test = get_trained_model(config, args, model, tokenizer)
    trained_model = trainer.model


    probs, labels = get_probs_and_labels(config['test_data'], encoded_samples_test, trained_model)
    f1_score_test = get_f1_score(probs, labels)
    recall, precision = compute_pr_curve(probs, labels)
    trainer.save_model('./best_model')

    log_extra(config, f1_score_test, recall, precision)

    print('ALL DONE')