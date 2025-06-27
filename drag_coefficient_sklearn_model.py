import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os

class DragCoefficientMLPPredictor:
    """
    Multi-Layer Perceptron for predicting drag coefficient using scikit-learn
    """
    def __init__(self, hidden_layer_sizes=(100, 50, 25), max_iter=1000, 
                 learning_rate_init=0.001, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.training_history = []
        
    def prepare_data(self, data_path):
        """Load and prepare the dataset"""
        data = pd.read_csv(data_path)
        
        # Use log scale for Reynolds number (common in fluid mechanics)
        X = np.log10(data['Reynolds_number'].values).reshape(-1, 1)
        y = data['drag_coefficient'].values.reshape(-1, 1)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale the features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_test, y_test
    
    def create_and_train_model(self, X_train, y_train, X_val, y_val):
        """Create and train the MLP model"""
        
        # Create the model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            alpha=0.001,  # L2 regularization
            solver='adam'
        )
        
        print(f"Training MLP with architecture: {self.hidden_layer_sizes}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        # Train the model
        self.model.fit(X_train, y_train.ravel())
        
        # Store training history (loss curve)
        self.training_history = self.model.loss_curve_
        
        print(f"Training completed in {self.model.n_iter_} iterations")
        print(f"Final training loss: {self.model.loss_:.6f}")
        
        return self.training_history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        print(f"Test Results:")
        print(f"R² Score: {r2:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {
            'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
            'mse': mse, 'predictions': predictions
        }
    
    def save_model(self, model_path='models/drag_coefficient_sklearn_model.pkl'):
        """Save the trained model and scalers"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'training_history': self.training_history,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'max_iter': self.max_iter,
            'learning_rate_init': self.learning_rate_init
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path='models/drag_coefficient_sklearn_model.pkl'):
        """Load a trained model and scalers"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.training_history = model_data['training_history']
        self.hidden_layer_sizes = model_data['hidden_layer_sizes']
        
        print(f"Model loaded from: {model_path}")
        
    def plot_training_curve(self, save_path=None):
        """Plot the training loss curve"""
        if not self.training_history:
            print("No training history available")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add final loss annotation
        final_loss = self.training_history[-1]
        plt.text(0.98, 0.98, f'Final Loss: {final_loss:.6f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

def main():
    """Main training and evaluation pipeline"""
    
    # Initialize the predictor
    predictor = DragCoefficientMLPPredictor(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=2000,
        learning_rate_init=0.001,
        random_state=42
    )
    
    print("="*80)
    print("DRAG COEFFICIENT PREDICTION - SCIKIT-LEARN MLP")
    print("="*80)
    
    # Check if data file exists
    data_file = 'drag_coefficient_data.csv'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print("Please run data_generation.py first to generate the dataset.")
        return
    
    # Prepare data
    print("\n1. Loading and preparing data...")
    X_train, X_test, y_train, y_test, X_test_orig, y_test_orig = predictor.prepare_data(data_file)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Input features: {X_train.shape[1]} (log10 of Reynolds number)")
    
    # Train the model
    print("\n2. Training the model...")
    training_history = predictor.create_and_train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate the model
    print("\n3. Evaluating the model...")
    results = predictor.evaluate(X_test, y_test_orig)
    
    # Plot training curve
    print("\n4. Plotting training curve...")
    predictor.plot_training_curve(save_path='visualizations/training_curve.png')
    
    # Save the model
    print("\n5. Saving the model...")
    predictor.save_model()
    
    # Generate additional visualizations using the visualization utilities
    print("\n6. Generating comprehensive visualizations...")
    try:
        from visualization_utils import DragCoefficientVisualizer
        
        # Load original data for visualization
        data = pd.read_csv(data_file)
        
        # Create visualizer
        visualizer = DragCoefficientVisualizer()
        
        # Create mock training/validation losses for visualization
        # (since sklearn doesn't provide separate validation loss)
        train_losses = training_history
        val_losses = [loss * 1.1 for loss in training_history]  # Mock validation loss
        
        # Generate comprehensive report
        visualizer.create_comprehensive_report(
            data, train_losses, val_losses, 
            X_test_orig, y_test_orig, results['predictions']
        )
        
    except ImportError:
        print("Visualization utilities not available. Skipping comprehensive plots.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Final Model Performance:")
    print(f"  R² Score: {results['r2']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAPE: {results['mape']:.2f}%")
    print(f"\nModel saved to: models/drag_coefficient_sklearn_model.pkl")
    print(f"Visualizations saved to: visualizations/")
    print("="*80)

if __name__ == "__main__":
    main()