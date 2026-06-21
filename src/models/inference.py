import torch
import numpy as np
import logging
from pathlib import Path

# Import the model architectures
from vol_lstm import VolatilityLSTM
from regime_hmm_lstm import RegimeLSTM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepVolInferenceEngine:
    """
    High-speed inference engine for live trading.
    Loads trained PyTorch models and provides a clean API for the strategy module.
    """
    def __init__(self, num_features: int, vol_weights_path: str, regime_weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = num_features
        
        logging.info(f"Initializing Inference Engine on {self.device}...")
        
        # 1. Initialize Architectures
        # Ensure these hyperparams match exactly what you used during training in the notebooks
        self.vol_model = VolatilityLSTM(input_size=num_features, hidden_size=32, num_layers=2).to(self.device)
        self.regime_model = RegimeLSTM(input_size=num_features, hidden_size=32, num_layers=2, num_classes=3).to(self.device)
        
        # 2. Load Weights safely
        self._load_weights(self.vol_model, vol_weights_path, "Volatility LSTM")
        self._load_weights(self.regime_model, regime_weights_path, "Regime LSTM")
        
        # 3. CRITICAL: Lock models into evaluation mode
        # This disables Dropout and locks BatchNorm layers. Without this, your live 
        # predictions will be randomized by active dropout layers!
        self.vol_model.eval()
        self.regime_model.eval()
        
        logging.info("Inference Engine ready. Models locked in eval mode.")

    def _load_weights(self, model: torch.nn.Module, filepath: str, model_name: str):
        """Helper to load PyTorch state dicts safely across CPU/GPU setups."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CRITICAL: Weights file not found for {model_name} at {filepath}")
            
        # map_location ensures we don't crash if loading a GPU-trained model on a CPU machine
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        logging.info(f"{model_name} weights loaded successfully.")

    def predict_all(self, feature_sequence: np.ndarray) -> dict:
        """
        Executes a forward pass on both models.
        
        Args:
            feature_sequence: A 2D numpy array of shape (Sequence_Length, Num_Features).
                              This is the fully scaled, preprocessed rolling window of data.
                              Must contain at least 78 bars.
                              
        Returns:
            Dictionary containing predicted volatility (float) and regime probabilities (list).
        """
        # Ensure we have enough data for the longest lookback (Volatility model needs 78)
        if len(feature_sequence) < 78:
            raise ValueError(f"Insufficient sequence length. Expected >= 78, got {len(feature_sequence)}")
            
        # The Volatility model expects 78 bars, the Regime model expects 48 bars
        vol_seq = feature_sequence[-78:]
        regime_seq = feature_sequence[-48:]
        
        # Convert to PyTorch tensors and add the Batch Dimension (Batch_Size = 1)
        # Shape becomes: (1, seq_len, num_features)
        vol_tensor = torch.tensor(vol_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        regime_tensor = torch.tensor(regime_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # torch.no_grad() is absolutely mandatory for live inference.
        # It disables the autograd engine, saving massive amounts of RAM and CPU cycles.
        with torch.no_grad():
            # Get Volatility prediction
            predicted_vol = self.vol_model(vol_tensor).item()
            
            # Get Regime probabilities
            regime_probs = self.regime_model(regime_tensor).squeeze(0).cpu().numpy().tolist()
            
        return {
            "predicted_volatility": predicted_vol,
            "regime_probabilities": {
                "State_0_Downtrend": regime_probs[0],
                "State_1_Uptrend": regime_probs[1],
                "State_2_Chop": regime_probs[2]
            }
        }

# --- Module Test & Sanity Check ---
if __name__ == "__main__":
    # Create dummy weight files to test the initialization logic without crashing
    dummy_vol_path = "../../weights/dummy_vol.pth"
    dummy_regime_path = "../../weights/dummy_regime.pth"
    
    Path("../../weights").mkdir(parents=True, exist_ok=True)
    
    # Initialize dummy models just to save their weights
    torch.save(VolatilityLSTM(15).state_dict(), dummy_vol_path)
    torch.save(RegimeLSTM(15).state_dict(), dummy_regime_path)
    
    # Test the Engine
    print("--- Testing DeepVol Inference Engine ---")
    engine = DeepVolInferenceEngine(
        num_features=15, 
        vol_weights_path=dummy_vol_path, 
        regime_weights_path=dummy_regime_path
    )
    
    # Simulate a raw 2D numpy array passed in by the strategy script
    # Shape: 100 recent bars, 15 engineered features
    dummy_live_data = np.random.randn(100, 15)
    
    # Run Inference
    results = engine.predict_all(dummy_live_data)
    
    print("\n[ Inference Output ]")
    print(f"Predicted Next-Bar Volatility (σ): {results['predicted_volatility']:.6f}")
    print("Regime Probabilities:")
    for state, prob in results['regime_probabilities'].items():
        print(f"  - {state}: {prob:.4f}")
        
    # Clean up dummy weights
    Path(dummy_vol_path).unlink()
    Path(dummy_regime_path).unlink()