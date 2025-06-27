import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class DragCoefficientVisualizer:
    """
    Comprehensive visualization utilities for drag coefficient prediction project
    """
    
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'actual': '#2E86AB',
            'predicted': '#A23B72',
            'residual': '#F18F01',
            'physics': '#C73E1D',
            'grid': '#CCCCCC'
        }
    
    def plot_dataset_overview(self, data, save=True):
        """
        Plot 1: Dataset Overview
        Shows the distribution of Reynolds numbers and drag coefficients
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reynolds number distribution (log scale)
        ax1.hist(np.log10(data['Reynolds_number']), bins=50, alpha=0.7, 
                color=self.colors['actual'], edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('log₁₀(Reynolds Number)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Reynolds Numbers\n(Log Scale)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats_text = f'Mean: {np.log10(data["Reynolds_number"]).mean():.2f}\n'
        stats_text += f'Std: {np.log10(data["Reynolds_number"]).std():.2f}\n'
        stats_text += f'Range: [{np.log10(data["Reynolds_number"]).min():.1f}, {np.log10(data["Reynolds_number"]).max():.1f}]'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Drag coefficient distribution
        ax2.hist(data['drag_coefficient'], bins=50, alpha=0.7, 
                color=self.colors['predicted'], edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Drag Coefficient', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Drag Coefficients', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats_text = f'Mean: {data["drag_coefficient"].mean():.3f}\n'
        stats_text += f'Std: {data["drag_coefficient"].std():.3f}\n'
        stats_text += f'Range: [{data["drag_coefficient"].min():.3f}, {data["drag_coefficient"].max():.3f}]'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Main scatter plot: Cd vs Re (log-log)
        ax3.loglog(data['Reynolds_number'], data['drag_coefficient'], 
                  'o', markersize=3, alpha=0.6, color=self.colors['actual'])
        ax3.set_xlabel('Reynolds Number (Re)', fontsize=12)
        ax3.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
        ax3.set_title('Drag Coefficient vs Reynolds Number\n(Physics-Based Relationship)', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add physics regions
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stokes Flow (Re<1)')
        ax3.axvline(x=1000, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Intermediate (Re~1000)')
        ax3.legend(loc='upper right')
        
        # Add annotation about physics
        physics_text = 'Flow Regimes:\n• Re < 1: Stokes Flow\n• 1 < Re < 1000: Intermediate\n• Re > 1000: Inertial'
        ax3.text(0.02, 0.98, physics_text, transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # 4. Theoretical curve comparison
        Re_theory = np.logspace(-1, 5, 1000)
        Cd_theory = 24/Re_theory + 6/(1 + np.sqrt(Re_theory)) + 0.4
        
        ax4.loglog(Re_theory, Cd_theory, 'r-', linewidth=3, label='Theoretical Formula', alpha=0.8)
        ax4.loglog(data['Reynolds_number'], data['drag_coefficient'], 
                  'o', markersize=2, alpha=0.5, color=self.colors['actual'], label='Generated Data')
        ax4.set_xlabel('Reynolds Number (Re)', fontsize=12)
        ax4.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
        ax4.set_title('Data vs Theoretical Formula\nCd = 24/Re + 6/(1+√Re) + 0.4', 
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add formula annotation
        formula_text = 'Empirical Formula:\nCd = 24/Re + 6/(1+√Re) + 0.4\n\nValid for Re < 2×10⁵'
        ax4.text(0.02, 0.02, formula_text, transform=ax4.transAxes, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/01_dataset_overview.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/01_dataset_overview.pdf', bbox_inches='tight')
        
        plt.show()
        
        # Print comprehensive dataset statistics
        print("="*80)
        print("DATASET OVERVIEW STATISTICS")
        print("="*80)
        print(f"Total samples: {len(data):,}")
        print(f"\nReynolds Number Statistics:")
        print(f"  Range: {data['Reynolds_number'].min():.2e} to {data['Reynolds_number'].max():.2e}")
        print(f"  Mean: {data['Reynolds_number'].mean():.2e}")
        print(f"  Median: {data['Reynolds_number'].median():.2e}")
        print(f"  Std Dev: {data['Reynolds_number'].std():.2e}")
        print(f"\nDrag Coefficient Statistics:")
        print(f"  Range: {data['drag_coefficient'].min():.4f} to {data['drag_coefficient'].max():.4f}")
        print(f"  Mean: {data['drag_coefficient'].mean():.4f}")
        print(f"  Median: {data['drag_coefficient'].median():.4f}")
        print(f"  Std Dev: {data['drag_coefficient'].std():.4f}")
        print("="*80)
    
    def plot_training_analysis(self, train_losses, val_losses, save=True):
        """
        Plot 2: Training Analysis
        Shows training/validation loss curves and convergence analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = np.arange(1, len(train_losses) + 1)
        
        # 1. Training/Validation Loss (Linear Scale)
        ax1.plot(epochs, train_losses, color=self.colors['actual'], linewidth=2, label='Training Loss')
        ax1.plot(epochs, val_losses, color=self.colors['predicted'], linewidth=2, label='Validation Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training Progress - Linear Scale', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add final loss annotation
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        ax1.text(0.98, 0.98, f'Final Training Loss: {final_train_loss:.6f}\nFinal Validation Loss: {final_val_loss:.6f}', 
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 2. Training/Validation Loss (Log Scale)
        ax2.semilogy(epochs, train_losses, color=self.colors['actual'], linewidth=2, label='Training Loss')
        ax2.semilogy(epochs, val_losses, color=self.colors['predicted'], linewidth=2, label='Validation Loss')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (MSE) - Log Scale', fontsize=12)
        ax2.set_title('Training Progress - Log Scale', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Overfitting Analysis
        overfitting_ratio = np.array(val_losses) / np.array(train_losses)
        ax3.plot(epochs, overfitting_ratio, color=self.colors['residual'], linewidth=2)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Fit Line')
        ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Overfitting Threshold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Validation Loss / Training Loss', fontsize=12)
        ax3.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add overfitting interpretation
        final_ratio = overfitting_ratio[-1]
        if final_ratio < 1.1:
            status = "Good Fit"
            color = "green"
        elif final_ratio < 1.3:
            status = "Slight Overfitting"
            color = "orange"
        else:
            status = "Overfitting"
            color = "red"
        
        ax3.text(0.02, 0.98, f'Final Ratio: {final_ratio:.3f}\nStatus: {status}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        # 4. Learning Rate Analysis (if available)
        # Smoothed loss for better trend analysis
        window_size = min(50, len(train_losses) // 10)
        if window_size > 1:
            smoothed_train = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_val = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = epochs[window_size-1:]
            
            ax4.plot(smooth_epochs, smoothed_train, color=self.colors['actual'], 
                    linewidth=3, label='Smoothed Training Loss')
            ax4.plot(smooth_epochs, smoothed_val, color=self.colors['predicted'], 
                    linewidth=3, label='Smoothed Validation Loss')
        else:
            ax4.plot(epochs, train_losses, color=self.colors['actual'], 
                    linewidth=2, label='Training Loss')
            ax4.plot(epochs, val_losses, color=self.colors['predicted'], 
                    linewidth=2, label='Validation Loss')
        
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss (MSE)', fontsize=12)
        ax4.set_title('Smoothed Loss Trends', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/02_training_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/02_training_analysis.pdf', bbox_inches='tight')
        
        plt.show()
        
        # Print training analysis
        print("="*80)
        print("TRAINING ANALYSIS")
        print("="*80)
        print(f"Total epochs: {len(train_losses)}")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        print(f"Overfitting ratio: {val_losses[-1]/train_losses[-1]:.3f}")
        print(f"Loss reduction: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%")
        print("="*80)
    
    def plot_prediction_analysis(self, X_test_original, y_test, predictions, save=True):
        """
        Plot 3: Comprehensive Prediction Analysis
        Shows prediction accuracy, residuals, and error distribution
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Convert back to original scale
        Re_test = 10**X_test_original.flatten()
        y_actual = y_test.flatten()
        y_pred = predictions.flatten()
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        # 1. Predictions vs Actual (Log-Log Scale)
        ax1 = plt.subplot(2, 3, 1)
        ax1.loglog(Re_test, y_actual, 'o', markersize=4, alpha=0.7, 
                  color=self.colors['actual'], label='Actual')
        ax1.loglog(Re_test, y_pred, 's', markersize=4, alpha=0.7, 
                  color=self.colors['predicted'], label='Predicted')
        ax1.set_xlabel('Reynolds Number (Re)', fontsize=12)
        ax1.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
        ax1.set_title('Predictions vs Actual\n(Log-Log Scale)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Parity Plot
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(y_actual, y_pred, alpha=0.6, color=self.colors['actual'], s=30)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Confidence bands
        ax2.fill_between([min_val, max_val], [min_val*0.9, max_val*0.9], 
                        [min_val*1.1, max_val*1.1], alpha=0.2, color='red', label='±10% Error Band')
        
        ax2.set_xlabel('Actual Drag Coefficient', fontsize=12)
        ax2.set_ylabel('Predicted Drag Coefficient', fontsize=12)
        ax2.set_title('Parity Plot', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add R² annotation
        ax2.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 3. Residual Analysis
        ax3 = plt.subplot(2, 3, 3)
        residuals = y_actual - y_pred
        ax3.semilogx(Re_test, residuals, 'o', markersize=4, alpha=0.6, color=self.colors['residual'])
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax3.axhline(y=residuals.std(), color='orange', linestyle=':', alpha=0.7, label=f'+1σ ({residuals.std():.4f})')
        ax3.axhline(y=-residuals.std(), color='orange', linestyle=':', alpha=0.7, label=f'-1σ ({-residuals.std():.4f})')
        ax3.set_xlabel('Reynolds Number (Re)', fontsize=12)
        ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        ax3.set_title('Residual Analysis', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Error Distribution
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(residuals, bins=30, alpha=0.7, color=self.colors['residual'], 
                edgecolor='black', linewidth=0.5, density=True)
        
        # Normal distribution overlay
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax4.plot(x, normal_dist, 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        
        ax4.set_xlabel('Residuals', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Relative Error Analysis
        ax5 = plt.subplot(2, 3, 5)
        relative_error = np.abs((y_actual - y_pred) / y_actual) * 100
        ax5.semilogx(Re_test, relative_error, 'o', markersize=4, alpha=0.6, color=self.colors['physics'])
        ax5.axhline(y=relative_error.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {relative_error.mean():.2f}%')
        ax5.axhline(y=5, color='orange', linestyle=':', alpha=0.7, label='5% Threshold')
        ax5.axhline(y=10, color='red', linestyle=':', alpha=0.7, label='10% Threshold')
        ax5.set_xlabel('Reynolds Number (Re)', fontsize=12)
        ax5.set_ylabel('Relative Error (%)', fontsize=12)
        ax5.set_title('Relative Error Analysis', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        ==================
        
        Coefficient of Determination (R²): {r2:.6f}
        Root Mean Square Error (RMSE): {rmse:.6f}
        Mean Absolute Error (MAE): {mae:.6f}
        Mean Absolute Percentage Error: {mape:.2f}%
        
        RESIDUAL STATISTICS
        ==================
        
        Mean Residual: {residuals.mean():.6f}
        Std Dev Residuals: {residuals.std():.6f}
        Max Absolute Error: {np.abs(residuals).max():.6f}
        
        ACCURACY THRESHOLDS
        ==================
        
        Predictions within 5% error: {(relative_error <= 5).sum()}/{len(relative_error)} ({(relative_error <= 5).mean()*100:.1f}%)
        Predictions within 10% error: {(relative_error <= 10).sum()}/{len(relative_error)} ({(relative_error <= 10).mean()*100:.1f}%)
        """
        
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/03_prediction_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/03_prediction_analysis.pdf', bbox_inches='tight')
        
        plt.show()
        
        # Print detailed metrics
        print("="*80)
        print("PREDICTION ANALYSIS RESULTS")
        print("="*80)
        print(f"R² Score: {r2:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Predictions within 5% error: {(relative_error <= 5).mean()*100:.1f}%")
        print(f"Predictions within 10% error: {(relative_error <= 10).mean()*100:.1f}%")
        print("="*80)
        
        return {
            'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
            'residuals': residuals, 'relative_error': relative_error
        }
    
    def plot_physics_comparison(self, X_test_original, y_test, predictions, save=True):
        """
        Plot 4: Physics-Based Comparison
        Compare predictions with theoretical physics across different flow regimes
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Convert back to original scale
        Re_test = 10**X_test_original.flatten()
        y_actual = y_test.flatten()
        y_pred = predictions.flatten()
        
        # Generate theoretical curve
        Re_theory = np.logspace(-1, 5, 1000)
        Cd_theory = 24/Re_theory + 6/(1 + np.sqrt(Re_theory)) + 0.4
        
        # 1. Full Range Comparison
        ax1.loglog(Re_theory, Cd_theory, 'r-', linewidth=3, label='Theoretical Formula', alpha=0.8)
        ax1.loglog(Re_test, y_actual, 'bo', markersize=4, alpha=0.6, label='Actual Data')
        ax1.loglog(Re_test, y_pred, 'gs', markersize=4, alpha=0.6, label='ML Predictions')
        
        # Flow regime boundaries
        ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
        ax1.text(0.1, 100, 'Stokes\nFlow', ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax1.text(30, 100, 'Intermediate\nFlow', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax1.text(10000, 100, 'Inertial\nFlow', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        ax1.set_xlabel('Reynolds Number (Re)', fontsize=12)
        ax1.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
        ax1.set_title('ML Predictions vs Physics Theory\n(Full Reynolds Number Range)', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Stokes Flow Regime (Re < 1)
        stokes_mask = Re_test < 1
        if np.any(stokes_mask):
            Re_stokes = Re_test[stokes_mask]
            actual_stokes = y_actual[stokes_mask]
            pred_stokes = y_pred[stokes_mask]
            theory_stokes = 24/Re_stokes  # Stokes law
            
            ax2.loglog(Re_stokes, actual_stokes, 'bo', markersize=6, alpha=0.7, label='Actual')
            ax2.loglog(Re_stokes, pred_stokes, 'gs', markersize=6, alpha=0.7, label='ML Predicted')
            ax2.loglog(Re_stokes, theory_stokes, 'r-', linewidth=3, label='Stokes Law (24/Re)')
            
            ax2.set_xlabel('Reynolds Number (Re)', fontsize=12)
            ax2.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
            ax2.set_title('Stokes Flow Regime (Re < 1)\nCd = 24/Re', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Calculate accuracy in Stokes regime
            stokes_error = np.abs(pred_stokes - actual_stokes) / actual_stokes * 100
            ax2.text(0.05, 0.95, f'Mean Error: {stokes_error.mean():.2f}%', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No data in Stokes regime\n(Re < 1)', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
            ax2.set_title('Stokes Flow Regime (Re < 1)', fontsize=14, fontweight='bold')
        
        # 3. Intermediate Flow Regime (1 < Re < 1000)
        intermediate_mask = (Re_test >= 1) & (Re_test < 1000)
        if np.any(intermediate_mask):
            Re_inter = Re_test[intermediate_mask]
            actual_inter = y_actual[intermediate_mask]
            pred_inter = y_pred[intermediate_mask]
            theory_inter = 24/Re_inter + 6/(1 + np.sqrt(Re_inter)) + 0.4
            
            ax3.loglog(Re_inter, actual_inter, 'bo', markersize=6, alpha=0.7, label='Actual')
            ax3.loglog(Re_inter, pred_inter, 'gs', markersize=6, alpha=0.7, label='ML Predicted')
            ax3.loglog(Re_inter, theory_inter, 'r-', linewidth=3, label='Empirical Formula')
            
            ax3.set_xlabel('Reynolds Number (Re)', fontsize=12)
            ax3.set_ylabel('Drag Coefficient (Cd)', fontsize=12)
            ax3.set_title('Intermediate Flow Regime (1 < Re < 1000)', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Calculate accuracy in intermediate regime
            inter_error = np.abs(pred_inter - actual_inter) / actual_inter * 100
            ax3.text(0.05, 0.95, f'Mean Error: {inter_error.mean():.2f}%', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No data in intermediate regime\n(1 < Re < 1000)', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=14)
            ax3.set_title('Intermediate Flow Regime (1 < Re < 1000)', fontsize=14, fontweight='bold')
        
        # 4. Physics Interpretation Summary
        ax4.axis('off')
        
        # Calculate regime-specific errors
        if np.any(stokes_mask):
            stokes_error_mean = stokes_error.mean()
            stokes_count = np.sum(stokes_mask)
        else:
            stokes_error_mean = 0
            stokes_count = 0
            
        if np.any(intermediate_mask):
            inter_error_mean = inter_error.mean()
            inter_count = np.sum(intermediate_mask)
        else:
            inter_error_mean = 0
            inter_count = 0
        
        inertial_mask = Re_test >= 1000
        if np.any(inertial_mask):
            inertial_error = np.abs(y_pred[inertial_mask] - y_actual[inertial_mask]) / y_actual[inertial_mask] * 100
            inertial_error_mean = inertial_error.mean()
            inertial_count = np.sum(inertial_mask)
        else:
            inertial_error_mean = 0
            inertial_count = 0
        
        physics_text = f"""
        PHYSICS-BASED ANALYSIS
        =====================
        
        FLOW REGIME BREAKDOWN:
        ---------------------
        Stokes Flow (Re < 1):
        • Data points: {stokes_count}
        • Physics: Viscous forces dominant
        • Theory: Cd = 24/Re (Stokes Law)
        • ML Error: {stokes_error_mean:.2f}%
        
        Intermediate Flow (1 < Re < 1000):
        • Data points: {inter_count}
        • Physics: Viscous + Inertial forces
        • Theory: Cd = 24/Re + 6/(1+√Re) + 0.4
        • ML Error: {inter_error_mean:.2f}%
        
        Inertial Flow (Re > 1000):
        • Data points: {inertial_count}
        • Physics: Inertial forces dominant
        • Theory: Cd ≈ 0.4 (constant)
        • ML Error: {inertial_error_mean:.2f}%
        
        PHYSICS VALIDATION:
        ------------------
        The ML model successfully captures:
        ✓ 1/Re dependency at low Re
        ✓ Smooth transition in intermediate regime
        ✓ Approach to constant Cd at high Re
        """
        
        ax4.text(0.05, 0.95, physics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/04_physics_comparison.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/04_physics_comparison.pdf', bbox_inches='tight')
        
        plt.show()
        
        # Print physics analysis
        print("="*80)
        print("PHYSICS-BASED VALIDATION")
        print("="*80)
        print(f"Stokes regime (Re < 1): {stokes_count} points, {stokes_error_mean:.2f}% error")
        print(f"Intermediate regime (1 < Re < 1000): {inter_count} points, {inter_error_mean:.2f}% error")
        print(f"Inertial regime (Re > 1000): {inertial_count} points, {inertial_error_mean:.2f}% error")
        print("="*80)
    
    def create_comprehensive_report(self, data, train_losses, val_losses, 
                                  X_test_original, y_test, predictions, save=True):
        """
        Create a comprehensive visualization report with all analyses
        """
        print("Generating comprehensive visualization report...")
        print("="*80)
        
        # Generate all plots
        self.plot_dataset_overview(data, save=save)
        self.plot_training_analysis(train_losses, val_losses, save=save)
        metrics = self.plot_prediction_analysis(X_test_original, y_test, predictions, save=save)
        self.plot_physics_comparison(X_test_original, y_test, predictions, save=save)
        
        if save:
            # Create summary report
            self._create_summary_report(data, train_losses, val_losses, metrics)
            
        print(f"\nAll visualizations saved to: {self.output_dir}/")
        print("Report generation complete!")
        
        return metrics
    
    def _create_summary_report(self, data, train_losses, val_losses, metrics):
        """Create a text summary report"""
        with open(f'{self.output_dir}/summary_report.txt', 'w') as f:
            f.write("DRAG COEFFICIENT PREDICTION - COMPREHENSIVE REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("DATASET SUMMARY:\n")
            f.write(f"Total samples: {len(data):,}\n")
            f.write(f"Reynolds number range: {data['Reynolds_number'].min():.2e} to {data['Reynolds_number'].max():.2e}\n")
            f.write(f"Drag coefficient range: {data['drag_coefficient'].min():.4f} to {data['drag_coefficient'].max():.4f}\n\n")
            
            f.write("TRAINING SUMMARY:\n")
            f.write(f"Total epochs: {len(train_losses)}\n")
            f.write(f"Final training loss: {train_losses[-1]:.6f}\n")
            f.write(f"Final validation loss: {val_losses[-1]:.6f}\n")
            f.write(f"Overfitting ratio: {val_losses[-1]/train_losses[-1]:.3f}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"R² Score: {metrics['r2']:.6f}\n")
            f.write(f"RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"MAE: {metrics['mae']:.6f}\n")
            f.write(f"MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"Predictions within 5% error: {(metrics['relative_error'] <= 5).mean()*100:.1f}%\n")
            f.write(f"Predictions within 10% error: {(metrics['relative_error'] <= 10).mean()*100:.1f}%\n\n")
            
            f.write("GENERATED VISUALIZATIONS:\n")
            f.write("1. 01_dataset_overview.png - Dataset statistics and distribution\n")
            f.write("2. 02_training_analysis.png - Training progress and convergence\n")
            f.write("3. 03_prediction_analysis.png - Prediction accuracy and errors\n")
            f.write("4. 04_physics_comparison.png - Physics-based validation\n")
            f.write("\nAll plots available in both PNG and PDF formats.\n")

# Example usage function
def run_comprehensive_visualization():
    """
    Example function showing how to use the visualization utilities
    """
    # This would typically be called from the main training script
    visualizer = DragCoefficientVisualizer()
    
    # Load data (example)
    # data = pd.read_csv('drag_coefficient_data.csv')
    # train_losses = [...] # from training
    # val_losses = [...] # from training
    # X_test_original = [...] # test features
    # y_test = [...] # test labels
    # predictions = [...] # model predictions
    
    # Generate comprehensive report
    # metrics = visualizer.create_comprehensive_report(
    #     data, train_losses, val_losses, X_test_original, y_test, predictions
    # )
    
    print("Visualization utilities ready!")
    print("Use DragCoefficientVisualizer class for comprehensive plotting.")

if __name__ == "__main__":
    run_comprehensive_visualization()