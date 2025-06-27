import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class DragCoefficientMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting drag coefficient
    """
    def __init__(self, input_size=1, hidden_sizes=[32, 32, 16], output_size=1, dropout_rate=0.1):
        super(DragCoefficientMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DragCoefficientPredictor:
    """
    Main class for training and evaluating the drag coefficient prediction model
    """
    def __init__(self, model_params=None):
        self.model_params = model_params or {}
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data_path):
        """Load and prepare the dataset"""
        data = pd.read_csv(data_path)
        
        # Use log scale for Reynolds number (common in fluid mechanics)
        X = np.log10(data['Reynolds_number'].values).reshape(-1, 1)
        y = data['drag_coefficient'].values.reshape(-1, 1)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_test, y_test
    
    def create_model(self):
        """Create the MLP model"""
        self.model = DragCoefficientMLP(**self.model_params).to(self.device)
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32, learning_rate=0.001):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)
        
        print(f"Test Results:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        
        return mse, rmse, r2, predictions
    
    def plot_results(self, X_test_original, y_test, predictions, train_losses, val_losses):
        """Plot training history and predictions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training history
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Predictions vs actual (log scale)
        Re_test = 10**X_test_original.flatten()
        ax2.loglog(Re_test, y_test.flatten(), 'bo', alpha=0.6, markersize=4, label='Actual')
        ax2.loglog(Re_test, predictions.flatten(), 'ro', alpha=0.6, markersize=4, label='Predicted')
        ax2.set_xlabel('Reynolds Number')
        ax2.set_ylabel('Drag Coefficient')
        ax2.set_title('Predictions vs Actual (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_test.flatten() - predictions.flatten()
        ax3.semilogx(Re_test, residuals, 'go', alpha=0.6, markersize=4)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Reynolds Number')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residual Plot')
        ax3.grid(True, alpha=0.3)
        
        # Parity plot
        ax4.plot(y_test.flatten(), predictions.flatten(), 'bo', alpha=0.6, markersize=4)
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax4.set_xlabel('Actual Drag Coefficient')
        ax4.set_ylabel('Predicted Drag Coefficient')
        ax4.set_title('Parity Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize the predictor
    model_params = {
        'input_size': 1,
        'hidden_sizes': [32, 32, 16],
        'output_size': 1,
        'dropout_rate': 0.1
    }
    
    predictor = DragCoefficientPredictor(model_params)
    
    # Prepare data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = predictor.prepare_data('drag_coefficient_data.csv')
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train the model
    print("\nTraining the model...")
    train_losses, val_losses = predictor.train(
        X_train, y_train, X_test, y_test, 
        epochs=500, batch_size=32, learning_rate=0.001
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    mse, rmse, r2, predictions = predictor.evaluate(X_test, y_test_orig)
    
    # Plot results
    predictor.plot_results(X_test_orig, y_test_orig, predictions, train_losses, val_losses)
    
    # Save the model
    torch.save(predictor.model.state_dict(), 'drag_coefficient_model.pth')
    print("\nModel saved as 'drag_coefficient_model.pth'")

if __name__ == "__main__":
    main()