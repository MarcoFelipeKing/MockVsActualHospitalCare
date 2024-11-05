# Healthcare Surface Contact Sequence Models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple
import random

class ContactSequenceDataset:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.sequences = self._process_sequences()
        self.surface_to_idx, self.idx_to_surface = self._create_vocabulary()
        
    def _process_sequences(self) -> Dict[int, List[str]]:
        """Convert DataFrame into dictionary of sequences by ActivityID"""
        sequences = {}
        for activity_id, group in self.df.groupby('ActivityID'):
            # Ensure sequence ends with 'Out'
            sequence = group['Surface'].tolist()
            if sequence[-1] != 'Out':
                sequence.append('Out')
            sequences[activity_id] = sequence
        return sequences
    
    def _create_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create surface-to-index mappings"""
        unique_surfaces = sorted(set(self.df['Surface'].unique()) | {'Out'})
        surface_to_idx = {surface: idx for idx, surface in enumerate(unique_surfaces)}
        idx_to_surface = {idx: surface for surface, idx in surface_to_idx.items()}
        return surface_to_idx, idx_to_surface

    def get_numeric_sequences(self) -> List[List[int]]:
        """Convert sequences to numeric form"""
        return [[self.surface_to_idx[surface] for surface in seq] 
                for seq in self.sequences.values()]

class MarkovChainModel:
    def __init__(self, order: int = 1):
        self.order = order
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.start_states = defaultdict(float)
        
    def fit(self, sequences: List[List[int]]):
        """Train the Markov chain on sequences"""
        # Count transitions
        for sequence in sequences:
            # Count start states
            state = tuple(sequence[:self.order])
            self.start_states[state] += 1
            
            # Count transitions
            for i in range(len(sequence) - self.order):
                current = tuple(sequence[i:i + self.order])
                next_state = sequence[i + self.order]
                self.transitions[current][next_state] += 1
        
        # Convert to probabilities
        total_starts = sum(self.start_states.values())
        for state in self.start_states:
            self.start_states[state] /= total_starts
            
        for current in self.transitions:
            total = sum(self.transitions[current].values())
            for next_state in self.transitions[current]:
                self.transitions[current][next_state] /= total
    
    def generate_sequence(self, max_length: int = 50) -> List[int]:
        """Generate a new sequence"""
        # Choose start state
        current = random.choices(list(self.start_states.keys()),
                               weights=list(self.start_states.values()))[0]
        sequence = list(current)
        
        while len(sequence) < max_length:
            current_state = tuple(sequence[-self.order:])
            if current_state not in self.transitions:
                break
                
            next_states = list(self.transitions[current_state].keys())
            probabilities = list(self.transitions[current_state].values())
            next_state = random.choices(next_states, weights=probabilities)[0]
            
            sequence.append(next_state)
            if next_state == sequence[-1]:  # If we reach 'Out'
                break
                
        return sequence

class LSTMSequenceModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden
    
    def generate_sequence(self, 
                         start_sequence: List[int], 
                         max_length: int = 50, 
                         temperature: float = 1.0) -> List[int]:
        """Generate a sequence given a start sequence"""
        self.eval()
        with torch.no_grad():
            sequence = start_sequence.copy()
            hidden = None
            
            while len(sequence) < max_length:
                input_tensor = torch.LongTensor([sequence[-1]]).unsqueeze(0)
                output, hidden = self(input_tensor, hidden)
                
                # Apply temperature
                logits = output[0, -1] / temperature
                probs = torch.softmax(logits, dim=0)
                
                # Sample from distribution
                next_token = torch.multinomial(probs, 1).item()
                sequence.append(next_token)
                
                if next_token == sequence[-1]:  # If we reach 'Out'
                    break
                    
        return sequence

class SequenceEvaluator:
    def __init__(self, dataset: ContactSequenceDataset):
        self.dataset = dataset
        
    def evaluate_model(self, model, num_sequences: int = 100) -> Dict:
        """Evaluate model performance"""
        real_sequences = self.dataset.get_numeric_sequences()
        generated_sequences = []
        
        # Generate sequences
        if isinstance(model, MarkovChainModel):
            generated_sequences = [model.generate_sequence() 
                                 for _ in range(num_sequences)]
        else:  # LSTM model
            for _ in range(num_sequences):
                start_seq = random.choice(real_sequences)[:1]
                generated_sequences.append(model.generate_sequence(start_seq))
        
        # Calculate metrics
        metrics = {
            'length_stats': self._length_statistics(real_sequences, generated_sequences),
            'surface_frequencies': self._surface_frequencies(real_sequences, generated_sequences),
            'transition_similarity': self._transition_similarity(real_sequences, generated_sequences),
            'common_patterns': self._common_patterns(real_sequences, generated_sequences)
        }
        
        return metrics
    
    def _length_statistics(self, real_seqs, gen_seqs):
        real_lengths = [len(seq) for seq in real_seqs]
        gen_lengths = [len(seq) for seq in gen_seqs]
        
        return {
            'real': {
                'mean': np.mean(real_lengths),
                'std': np.std(real_lengths),
                'min': min(real_lengths),
                'max': max(real_lengths)
            },
            'generated': {
                'mean': np.mean(gen_lengths),
                'std': np.std(gen_lengths),
                'min': min(gen_lengths),
                'max': max(gen_lengths)
            }
        }
    
    def _surface_frequencies(self, real_seqs, gen_seqs):
        def get_frequencies(seqs):
            all_surfaces = [s for seq in seqs for s in seq]
            total = len(all_surfaces)
            return {surface: count/total 
                    for surface, count in Counter(all_surfaces).items()}
        
        real_freq = get_frequencies(real_seqs)
        gen_freq = get_frequencies(gen_seqs)
        
        return {'real': real_freq, 'generated': gen_freq}
    
    def _transition_similarity(self, real_seqs, gen_seqs):
        def get_transitions(seqs):
            transitions = defaultdict(lambda: defaultdict(int))
            for seq in seqs:
                for i in range(len(seq)-1):
                    transitions[seq[i]][seq[i+1]] += 1
            return transitions
        
        real_trans = get_transitions(real_seqs)
        gen_trans = get_transitions(gen_seqs)
        
        # Calculate similarity score
        similarity = 0
        total = 0
        for current in real_trans:
            if current in gen_trans:
                for next_state in real_trans[current]:
                    if next_state in gen_trans[current]:
                        similarity += 1
                    total += 1
                    
        return similarity / total if total > 0 else 0
    
    def _common_patterns(self, real_seqs, gen_seqs, pattern_length=3):
        def get_patterns(seqs):
            patterns = defaultdict(int)
            for seq in seqs:
                for i in range(len(seq)-pattern_length+1):
                    pattern = tuple(seq[i:i+pattern_length])
                    patterns[pattern] += 1
            return patterns
        
        real_patterns = get_patterns(real_seqs)
        gen_patterns = get_patterns(gen_seqs)
        
        return {
            'real': dict(sorted(real_patterns.items(), 
                              key=lambda x: x[1], reverse=True)[:10]),
            'generated': dict(sorted(gen_patterns.items(), 
                                   key=lambda x: x[1], reverse=True)[:10])
        }

def plot_comparison(metrics: Dict, dataset: ContactSequenceDataset, save_path: str = None):
    """Plot comparison metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot length distributions
    real_lengths = [len(seq) for seq in dataset.get_numeric_sequences()]
    sns.histplot(real_lengths, ax=ax1, label='Real', alpha=0.5)
    ax1.set_title('Sequence Length Distribution')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Count')
    
    # Plot surface frequencies
    surface_freq = pd.DataFrame({
        'Real': metrics['surface_frequencies']['real'],
        'Generated': metrics['surface_frequencies']['generated']
    }).fillna(0)
    surface_freq.plot(kind='bar', ax=ax2)
    ax2.set_title('Surface Frequencies')
    ax2.set_xlabel('Surface')
    ax2.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # Plot transition similarity matrix
    transition_matrix = np.zeros((len(dataset.surface_to_idx), 
                                len(dataset.surface_to_idx)))
    sns.heatmap(transition_matrix, ax=ax3)
    ax3.set_title('Transition Similarity Matrix')
    
    # Plot common patterns comparison
    real_patterns = metrics['common_patterns']['real']
    gen_patterns = metrics['common_patterns']['generated']
    pattern_comparison = pd.DataFrame({
        'Real': pd.Series(real_patterns),
        'Generated': pd.Series(gen_patterns)
    }).fillna(0)
    pattern_comparison.plot(kind='bar', ax=ax4)
    ax4.set_title('Common Patterns')
    ax4.set_xlabel('Pattern')
    ax4.set_ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Load and prepare data
    dataset = ContactSequenceDataset('movsdf.rbind_orientationcorrected.csv')
    sequences = dataset.get_numeric_sequences()
    train_seqs, test_seqs = train_test_split(sequences, test_size=0.2)

    # Train Markov Chain model
    markov_model = MarkovChainModel(order=2)
    markov_model.fit(train_seqs)
    
    # Train LSTM model
    vocab_size = len(dataset.surface_to_idx)
    lstm_model = LSTMSequenceModel(vocab_size=vocab_size,
                                 embedding_dim=32,
                                 hidden_dim=64)
    
    # Train LSTM (simplified training loop)
    optimizer = torch.optim.Adam(lstm_model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate both models
    evaluator = SequenceEvaluator(dataset)
    markov_metrics = evaluator.evaluate_model(markov_model)
    lstm_metrics = evaluator.evaluate_model(lstm_model)
    
    # Plot comparisons
    plot_comparison(markov_metrics, dataset, 'markov_analysis.png')
    plot_comparison(lstm_metrics, dataset, 'lstm_analysis.png')