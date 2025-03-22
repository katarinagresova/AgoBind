#!/usr/bin/env python3
"""
Basic demonstration of AgoBind functionality using real genomic datasets from Hugging Face.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Add parent directory to path to import agobind
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agobind modules
from agobind.data.loader import GenomicDataLoader
from agobind.models.hub import ModelHub
from agobind.embedding.embedder import SequenceEmbedder

# Import datasets library from Hugging Face
from datasets import load_dataset

# Import miRBench for miRNA datasets
try:
    from miRBench.dataset import list_datasets, get_dataset_df
    MIRBENCH_AVAILABLE = True
except ImportError:
    print("miRBench package not found. miRNA dataset functionality will be disabled.")
    MIRBENCH_AVAILABLE = False


def load_genomic_benchmark_data_from_hf(dataset_name="human_nontata_promoters", max_samples=20):
    """
    Load a genomic benchmark dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset to load (without the prefix)
        max_samples: Maximum number of samples to load from each split (for demo purposes)
    
    Returns:
        Dictionary with sequences and labels for both train and test sets
    """
    print(f"Loading dataset: {dataset_name} from Hugging Face")
    
    # Add the HF prefix to the dataset name
    hf_dataset_name = f"katarinagresova/Genomic_Benchmarks_{dataset_name}"
    
    try:
        # Load the dataset from Hugging Face
        print(f"Fetching dataset from: {hf_dataset_name}")
        dataset = load_dataset(hf_dataset_name)
        
        # Convert to sequences and labels
        train_sequences = []
        train_labels = []
        test_sequences = []
        test_labels = []
        
        # Get class names based on dataset
        if dataset_name == "human_nontata_promoters":
            class_names = ["negative", "positive"]
            # For human_nontata_promoters, we need to ensure we have samples from both classes
            samples_per_class = max_samples // 2
            
            # Group by label for train data
            train_by_label = {}
            for sample in dataset["train"]:
                label = sample["label"]
                if label not in train_by_label:
                    train_by_label[label] = []
                train_by_label[label].append(sample)
            
            # Sample from each class for train
            for label, samples in train_by_label.items():
                random.seed(42)  # For reproducibility
                selected = random.sample(samples, min(samples_per_class, len(samples)))
                for sample in selected:
                    train_sequences.append(sample["seq"])
                    train_labels.append(sample["label"])
            
            # Group by label for test data
            test_by_label = {}
            for sample in dataset["test"]:
                label = sample["label"]
                if label not in test_by_label:
                    test_by_label[label] = []
                test_by_label[label].append(sample)
            
            # Sample from each class for test
            for label, samples in test_by_label.items():
                random.seed(42)  # For reproducibility
                selected = random.sample(samples, min(samples_per_class, len(samples)))
                for sample in selected:
                    test_sequences.append(sample["seq"])
                    test_labels.append(sample["label"])
                    
        else:
            # For other datasets, determine classes first
            all_labels = set()
            for sample in dataset["train"]:
                all_labels.add(sample["label"])
            
            # Create generic class names
            unique_labels = sorted(list(all_labels))
            class_names = [f"class_{label}" for label in unique_labels]
            
            # Sample data ensuring equal representation
            samples_per_class = max_samples // len(unique_labels)
            
            # Group by label for train data
            train_by_label = {}
            for sample in dataset["train"]:
                label = sample["label"]
                if label not in train_by_label:
                    train_by_label[label] = []
                train_by_label[label].append(sample)
            
            # Sample from each class for train
            for label, samples in train_by_label.items():
                random.seed(42)  # For reproducibility
                selected = random.sample(samples, min(samples_per_class, len(samples)))
                for sample in selected:
                    train_sequences.append(sample["seq"])
                    train_labels.append(sample["label"])
            
            # Group by label for test data
            test_by_label = {}
            for sample in dataset["test"]:
                label = sample["label"]
                if label not in test_by_label:
                    test_by_label[label] = []
                test_by_label[label].append(sample)
            
            # Sample from each class for test
            for label, samples in test_by_label.items():
                random.seed(42)  # For reproducibility
                selected = random.sample(samples, min(samples_per_class, len(samples)))
                for sample in selected:
                    test_sequences.append(sample["seq"])
                    test_labels.append(sample["label"])
        
        print(f"Sampled data distribution:")
        for i, cls in enumerate(class_names):
            train_count = train_labels.count(i)
            test_count = test_labels.count(i)
            print(f"  {cls}: {train_count} train, {test_count} test samples")
            
        return {
            'train_sequences': train_sequences,
            'train_labels': train_labels,
            'test_sequences': test_sequences,
            'test_labels': test_labels,
            'class_names': class_names
        }
    
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        raise


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix for the classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to 'results/confusion_matrix.png'")
    plt.close()


def load_mirbench_data(dataset_name="AGO2_CLASH_Hejret2023", max_samples=20):
    """
    Load a miRNA-target interaction dataset from miRBench.
    
    Args:
        dataset_name: Name of the dataset to load
        max_samples: Maximum number of samples to load from each class (for demo purposes)
    
    Returns:
        Dictionary with sequences and labels for both train and test sets
    """
    if not MIRBENCH_AVAILABLE:
        raise ImportError("miRBench package is required but not installed. Please run: pip install miRBench")
    
    print(f"Loading dataset: {dataset_name} from miRBench")
    
    # List available datasets
    available_datasets = list_datasets()
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset {dataset_name} not found in miRBench. Available datasets: {available_datasets}")
    
    # Load the dataset
    train_df = get_dataset_df(dataset_name, split="train")
    test_df = get_dataset_df(dataset_name, split="test")
    
    print(f"Loaded miRBench dataset with {len(train_df)} training and {len(test_df)} test samples")
    
    # Process the data - concatenate noncodingRNA and gene sequences
    def process_data(df, max_samples_per_class):
        # Group by label
        grouped = df.groupby('label')
        
        sequences = []
        labels = []
        
        # Sample from each class
        for label, group in grouped:
            # Take a sample of the group
            if len(group) > max_samples_per_class:
                group = group.sample(max_samples_per_class, random_state=42)
            
            # Concatenate noncodingRNA and gene sequences
            for _, row in group.iterrows():
                combined_seq = row['noncodingRNA'] + row['gene']
                sequences.append(combined_seq)
                labels.append(int(row['label']))  # Ensure label is an integer
        
        return sequences, labels
    
    # Process train and test data
    train_sequences, train_labels = process_data(train_df, max_samples // 2)
    test_sequences, test_labels = process_data(test_df, max_samples // 2)
    
    # Get class names (0 and 1 for negative and positive interactions)
    class_names = ["negative", "positive"]
    
    print(f"Sampled data distribution:")
    for i, cls in enumerate(class_names):
        train_count = train_labels.count(i)
        test_count = test_labels.count(i)
        print(f"  {cls}: {train_count} train, {test_count} test samples")
    
    # Sample sequence
    sample_seq = train_sequences[0] if train_sequences else "No sequences available"
    print(f"Sample combined sequence: {sample_seq[:50]}...")
    
    return {
        'train_sequences': train_sequences,
        'train_labels': train_labels,
        'test_sequences': test_sequences,
        'test_labels': test_labels,
        'class_names': class_names
    }


def main():
    """Run a demo of the AgoBind package with genomic datasets."""
    print("Starting AgoBind demo...\n")
    
    # Ask user which data source to use
    print("Choose a data source:")
    print("1. Hugging Face genomic benchmarks")
    print("2. miRBench miRNA-target interaction datasets")
    
    # For the sake of this demo, we'll provide option to choose
    data_source = input("Enter your choice (1 or 2): ").strip()
    
    if data_source == "1":
        # Load data from Hugging Face
        print("\nLoading data from Hugging Face genomic benchmarks...")
        
        # Step 1: List available datasets on Hugging Face
        print("1. Suggested genomic benchmark datasets from Hugging Face:")
        hf_datasets = [
            "human_nontata_promoters",
            "human_enhancers_cohn",
            "human_enhancers_ensembl",
            "human_ensembl_regulatory", 
            "human_ocr_ensembl",
            "drosophila_enhancers_stark",
            "demo_coding_vs_intergenomic_seqs",
            "demo_human_or_worm"
        ]
        
        for i, dataset in enumerate(hf_datasets):
            print(f"  {i+1}. {dataset}")
        
        # Step 2: Load a genomic benchmark dataset from Hugging Face
        dataset_name = "human_nontata_promoters"  # A binary classification dataset
        print(f"\n2. Loading dataset: {dataset_name} from Hugging Face")
        data = load_genomic_benchmark_data_from_hf(dataset_name, max_samples=20000)
        
    elif data_source == "2":
        # Load data from miRBench
        if not MIRBENCH_AVAILABLE:
            print("miRBench package not found. Please install it using: pip install miRBench")
            return
        
        print("\nLoading data from miRBench...")
        
        # Step 1: List available datasets
        print("1. Available miRBench datasets:")
        mirbench_datasets = list_datasets()
        for i, dataset in enumerate(mirbench_datasets):
            print(f"  {i+1}. {dataset}")
        
        # Step 2: Load a miRBench dataset
        dataset_name = "AGO2_CLASH_Hejret2023"  # Default dataset
        print(f"\n2. Loading dataset: {dataset_name} from miRBench")
        data = load_mirbench_data(dataset_name, max_samples=20000)
        
    else:
        print("Invalid choice. Exiting.")
        return
    
    print(f"Loaded {len(data['train_sequences'])} training samples and {len(data['test_sequences'])} test samples")
    print(f"Classes: {data['class_names']}")
    
    # Step 3: Explore available models
    print("\n3. Exploring available genomic models...")
    model_hub = ModelHub()
    models = model_hub.list_genomic_models()
    print("Available genomic models:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model}")
    
    # Step 4: Get model information
    selected_model = "InstaDeepAI/nucleotide-transformer-2.5b-1000g"  # Using a DNA-specific model
    print(f"\n4. Getting information about model: {selected_model}")
    model_info = model_hub.get_model_info(selected_model)
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Step 5: Generate embeddings
    print("\n5. Generating embeddings (this may take a while)...")
    embedding_dir = 'embeddings'
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Check if CUDA is available
    import torch
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"Using device: {device}")
    
    print("\nNote: The embedder will automatically check and configure the tokenizer")
    print("      with necessary special tokens if any are missing.")
    
    # Create the embedder
    embedder = SequenceEmbedder(
        model_name=selected_model,
        output_dir=embedding_dir,
        pooling_strategy="mean",
        device=device
    )
    
    # Generate embeddings for train and test sets
    train_embeddings = embedder.embed_sequences(
        sequences=data['train_sequences'],
        batch_size=8,
        show_progress=True
    )
    
    test_embeddings = embedder.embed_sequences(
        sequences=data['test_sequences'],
        batch_size=8,
        show_progress=True
    )
    
    print(f"Generated train embeddings with shape: {train_embeddings.shape}")
    print(f"Generated test embeddings with shape: {test_embeddings.shape}")
    
    # Step 6: Train a simple classifier
    print("\n6. Training a simple classifier...")
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embeddings, data['train_labels'])
    
    # Evaluate on test set
    test_predictions = clf.predict(test_embeddings)
    accuracy = accuracy_score(data['test_labels'], test_predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    report = classification_report(data['test_labels'], test_predictions, 
                                  target_names=data['class_names'], output_dict=True)
    print("\nClassification Report:")
    for cls in data['class_names']:
        print(f"  {cls}:")
        print(f"    Precision: {report[cls]['precision']:.4f}")
        print(f"    Recall: {report[cls]['recall']:.4f}")
        print(f"    F1-score: {report[cls]['f1-score']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(data['test_labels'], test_predictions, data['class_names'])
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 