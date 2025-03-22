"""
Data loading functionality for genomic sequences.
"""

import os
from typing import Dict, Generator, List, Optional, Tuple, Union

import pandas as pd


class GenomicDataLoader:
    """
    Memory-efficient loader for genomic sequence data from CSV files.
    """

    def __init__(
        self,
        file_path: str,
        sequence_column: Union[str, List[str]],
        label_column: str,
        batch_size: int = 32,
        chunksize: int = 1000,
        **kwargs
    ):
        """
        Initialize the genomic data loader.

        Args:
            file_path: Path to the CSV file containing genomic sequences.
            sequence_column: Name of the column(s) containing genomic sequences.
            label_column: Name of the column containing the labels (0 or 1).
            batch_size: Size of batches to yield when iterating.
            chunksize: Number of rows to read at once from the CSV file.
            **kwargs: Additional arguments to pass to pandas.read_csv.
        """
        self.file_path = file_path
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.chunksize = chunksize
        self.csv_kwargs = kwargs
        
        # Validate that the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Peek at the file to get the column names and validate
        df_peek = pd.read_csv(file_path, nrows=1, **kwargs)
        
        # Validate sequence column(s)
        if isinstance(sequence_column, str):
            if sequence_column not in df_peek.columns:
                raise ValueError(f"Sequence column '{sequence_column}' not found in the CSV file.")
        else:
            for col in sequence_column:
                if col not in df_peek.columns:
                    raise ValueError(f"Sequence column '{col}' not found in the CSV file.")
        
        # Validate label column
        if label_column not in df_peek.columns:
            raise ValueError(f"Label column '{label_column}' not found in the CSV file.")

    def __iter__(self) -> Generator[Tuple[List[str], List[int]], None, None]:
        """
        Iterate over the dataset in batches.

        Yields:
            Tuple of (sequences, labels) where:
                - sequences is a list of genomic sequences
                - labels is a list of corresponding binary labels (0 or 1)
        """
        batch_sequences = []
        batch_labels = []
        
        # Read the CSV file in chunks to avoid loading the entire file into memory
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunksize, **self.csv_kwargs):
            # Process each row in the chunk
            for _, row in chunk.iterrows():
                # Extract sequence(s) from the row
                if isinstance(self.sequence_column, str):
                    sequence = row[self.sequence_column]
                else:
                    sequence = [row[col] for col in self.sequence_column]
                
                # Extract label from the row
                label = row[self.label_column]
                
                # Add to the current batch
                batch_sequences.append(sequence)
                batch_labels.append(label)
                
                # Yield a batch if we've reached batch_size
                if len(batch_sequences) >= self.batch_size:
                    yield batch_sequences, batch_labels
                    batch_sequences = []
                    batch_labels = []
        
        # Yield any remaining data
        if batch_sequences:
            yield batch_sequences, batch_labels

    def get_dataset_info(self) -> Dict:
        """
        Get basic information about the dataset without loading it completely.

        Returns:
            Dict with basic dataset information.
        """
        # Count the total number of rows
        with open(self.file_path, 'r') as f:
            # Subtract 1 for the header
            total_rows = sum(1 for _ in f) - 1
        
        # Get sample of data for statistics
        df_sample = pd.read_csv(self.file_path, nrows=min(1000, total_rows), **self.csv_kwargs)
        
        # Count label distribution in the sample
        label_counts = df_sample[self.label_column].value_counts().to_dict()
        
        # Calculate sequence length statistics if applicable
        seq_length_stats = {}
        if isinstance(self.sequence_column, str):
            lengths = df_sample[self.sequence_column].str.len()
            seq_length_stats = {
                "min_length": lengths.min(),
                "max_length": lengths.max(),
                "mean_length": lengths.mean(),
            }
        
        return {
            "total_rows": total_rows,
            "label_distribution_sample": label_counts,
            "sequence_length_stats": seq_length_stats,
            "columns": list(df_sample.columns),
        } 