from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score
#from genomic_benchmarks.data_check.info import labels_in_order
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def get_test_score(test_data, encoded_samples_test, model):

    predictions, labels = get_predictions_and_labels(test_data, encoded_samples_test, model)

    score = f1_score(labels, predictions)
    print('test f1 score is', score)
    return score

def compute_pr_curve(test_data, encoded_samples_test, model):
    print('Computing precision-recall curve')
    
    predictions, labels = get_predictions_and_labels(test_data, encoded_samples_test, model)

    #calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(labels, predictions)

    #create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='orange')

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    #display plot
    plt.show()
    plt.savefig('pr.png')

    return recall, precision

def get_predictions_and_labels(test_data, encoded_samples_test, model):
    test_loader = DataLoader(
                encoded_samples_test, 
                sampler = SequentialSampler(encoded_samples_test), 
                batch_size = 4 #TODO increase with your CPU
            )

    predictions = []
    # for sample in tqdm(test_loader, total=len(test_dset)/32):

    for sample in tqdm(test_loader, total=len(test_loader)): 

        outputs = model.to("cpu")(**sample)
        # outputs = model(**sample) #TODO make eval on GPU

        preds = outputs.logits.argmax(-1).tolist()
        predictions.extend(preds)

    labels = pd.read_csv(test_data, sep='/t', usecols=['label']).to_numpy()

    return predictions, labels