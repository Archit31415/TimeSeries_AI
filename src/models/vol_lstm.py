import torch
import torch.nn as nn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QLIKELoss(nn.Module):
    """
    Quasi-Likelihood (QLIKE) Loss Function.
    Standard MSE evenly penalizes absolute errors, which is bad for volatility 
    (a 1% error during a quiet market is much worse than a 1% error during a crash).
    QLIKE penalizes relative errors, making it the academic standard for volatility forecasting.
    """
    def __init__(self, eps: float = 1e-8):
        super(QLIKELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Convert Volatility (Standard Deviation) to Variance for the QLIKE formula
        var_true = y_true.pow(2) + self.eps
        var_pred = y_pred.pow(2) + self.eps
        
        # QLIKE Formula: (Variance_True / Variance_Pred) - log(Variance_True / Variance_Pred) - 1
        # The minimum is exactly 0 when var_true == var_pred
        ratio = var_true / var_pred
        loss = ratio - torch.log(ratio) - 1.0
        
        return torch.mean(loss)

class VolatilityLSTM(nn.Module):
    """
    Multivariate LSTM for Realized Volatility Forecasting.
    Takes a 3D tensor of shape (Batch_Size, Sequence_Length, Num_Features) 
    and predicts a single positive scalar representing the next timeframe's volatility.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.3):
        super(VolatilityLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        logging.info(f"Initializing VolatilityLSTM: Input={input_size}, Hidden={hidden_size}, Layers={num_layers}")

        # Core LSTM Engine
        # batch_first=True ensures inputs are (batch, seq, feature) instead of (seq, batch, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Fully Connected Block
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Softplus Activation: f(x) = ln(1 + e^x)
        # Guarantees the network output is strictly > 0. 
        # A model predicting negative volatility will mathematically crash your sizing algorithms.
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        x shape expected: (batch_size, sequence_length, features)
        """
        # h0 and c0 are implicitly initialized to zero by PyTorch if not provided
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # We only care about the LSTM's output at the final timestep of the sequence
        # lstm_out shape is (batch_size, seq_len, hidden_size)
        last_timestep_out = lstm_out[:, -1, :]
        
        # Pass through linear layers
        raw_pred = self.fc_block(last_timestep_out)
        
        # Force positive output
        vol_pred = self.activation(raw_pred)
        
        # Squeeze to turn shape (batch_size, 1) into (batch_size,) to match target tensors
        return vol_pred.squeeze(-1)

# --- Sanity Check / Initialization Test ---
if __name__ == "__main__":
    # Simulate hyperparameters
    BATCH_SIZE = 64
    SEQ_LEN = 78       # Lookback window (approx 1 trading day of 5-min bars)
    NUM_FEATURES = 15  # Total engineered features (Technical + Microstructure)
    
    # Initialize Model and Loss
    model = VolatilityLSTM(input_size=NUM_FEATURES, hidden_size=32, num_layers=2)
    criterion = QLIKELoss()
    mse_baseline = nn.MSELoss()
    
    # Create dummy tensor matching the shape of your preprocessed data
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    dummy_target = torch.abs(torch.randn(BATCH_SIZE)) + 0.01  # Simulated true volatility (strictly positive)
    
    # Forward Pass
    predictions = model(dummy_input)
    
    # Compute Loss
    qlike_val = criterion(predictions, dummy_target)
    mse_val = mse_baseline(predictions, dummy_target)
    
    print("--- Architecture Check Passed ---")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {predictions.shape}")
    print(f"Initial QLIKE Loss: {qlike_val.item():.4f}")
    print(f"Initial MSE Loss:   {mse_val.item():.4f}")