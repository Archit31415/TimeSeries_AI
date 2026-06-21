import torch
import torch.nn as nn
from hmmlearn import hmm
import numpy as np
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. OFFLINE MODULE: Gaussian HMM 
# ==========================================
class RegimeHMM:
    """
    Offline Unsupervised Model.
    Fits a Gaussian Hidden Markov Model to discover N latent market states.
    By default, N=3 (0: Downtrend/High Vol, 1: Uptrend/Low Vol, 2: Chop/Mean-Reverting).
    """
    def __init__(self, n_components: int = 3, covariance_type: str = 'full', random_state: int = 42):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type=covariance_type, 
            n_iter=1000, 
            random_state=random_state
        )
        self.is_fitted = False
        
    def fit(self, features: np.ndarray):
        """
        Fits the HMM on historical features (typically Log Returns and Rolling Volatility).
        features shape: (n_samples, n_features)
        """
        logging.info(f"Fitting Gaussian HMM with {self.n_components} components...")
        self.model.fit(features)
        self.is_fitted = True
        logging.info(f"HMM fitted successfully. Converged: {self.model.monitor_.converged}")
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts the hidden states (0, 1, or 2) for the given feature array.
        These predictions serve as the ground-truth labels (Y) for the LSTM.
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before calling predict.")
        return self.model.predict(features)
        
    def save(self, filepath: str):
        """Saves the fitted HMM model for later inference."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"HMM model saved to {filepath}")

    def load(self, filepath: str):
        """Loads a fitted HMM model."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        logging.info(f"HMM model loaded from {filepath}")


# ==========================================
# 2. ONLINE MODULE: Supervised LSTM
# ==========================================
class RegimeLSTM(nn.Module):
    """
    Online Supervised Model.
    Learns the complex mapping between a rolling window of stationary features 
    and the HMM's derived regimes. Designed to be lightweight for fast C++ IPC inference.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.3, num_classes: int = 3):
        super(RegimeLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        logging.info(f"Initializing RegimeLSTM: Input={input_size}, Hidden={hidden_size}, Layers={num_layers}, Classes={num_classes}")

        # Core LSTM Engine
        # batch_first=True -> Input format: (Batch, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output Layer: Maps the bottlenecked hidden state to the N probability classes
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Softmax activation to output a valid probability distribution (e.g., [0.10, 0.75, 0.15])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expected x shape: (batch_size, seq_len=48, num_features)
        Returns: (batch_size, 3) probabilities.
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Extract the output from the final timestep to make the prediction for the next bar
        last_timestep_out = lstm_out[:, -1, :]
        
        # Pass through linear layer to get logits
        logits = self.fc(last_timestep_out)
        
        # Convert to probabilities
        probabilities = self.softmax(logits)
        
        return probabilities

# --- Architecture Initialization & Sanity Check ---
if __name__ == "__main__":
    # Simulate HMM Setup (Offline)
    print("--- Testing HMM Module ---")
    dummy_hmm_features = np.random.randn(1000, 2) # e.g., 1000 samples of [Log Return, Volatility]
    hmm_model = RegimeHMM(n_components=3)
    hmm_model.fit(dummy_hmm_features)
    hmm_labels = hmm_model.predict(dummy_hmm_features)
    print(f"Generated {len(hmm_labels)} HMM labels. Unique classes: {np.unique(hmm_labels)}")
    
    # Simulate LSTM Setup (Online)
    print("\n--- Testing LSTM Module ---")
    BATCH_SIZE = 64
    SEQ_LEN = 48       # 48 bars = 4 hours of 5-minute data
    NUM_FEATURES = 15  # Engineered features
    
    lstm_model = RegimeLSTM(input_size=NUM_FEATURES, hidden_size=32, num_layers=2, dropout=0.3)
    
    # Dummy input matching the expected dimensions
    dummy_lstm_input = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    
    # Forward Pass
    predictions = lstm_model(dummy_lstm_input)
    
    print(f"LSTM Input Shape: {dummy_lstm_input.shape}")
    print(f"LSTM Output Shape: {predictions.shape}")
    print(f"Sample Output Probabilities (Row 0): {predictions[0].detach().numpy()}")
    print(f"Sum of probabilities (should be ~1.0): {torch.sum(predictions[0]).item():.4f}")
    
    # Loss verification: CrossEntropyLoss expects logits usually, but since we output probabilities via Softmax,
    # it is often better to use NLLLoss with LogSoftmax, OR pass raw logits to CrossEntropyLoss.
    # Note: If training with nn.CrossEntropyLoss, ensure you remove the self.softmax layer or use nn.NLLLoss with torch.log().