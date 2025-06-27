# Drag Coefficient Prediction using Physics-Guided Neural Networks

[![GitHub Repository](https://img.shields.io/badge/GitHub-drag--coefficient--prediction-blue?logo=github)](https://github.com/Sakeeb91/drag-coefficient-prediction)

This project implements a **Physics-Guided Neural Network (PgNN)** to predict the drag coefficient of spheres in fluid flow based on the Reynolds number. This is a low-compute, educational implementation inspired by fluid mechanics research.

## Project Overview

- **Input**: Reynolds number (dimensionless)
- **Output**: Drag coefficient (dimensionless)
- **Model**: Multi-Layer Perceptron (MLP) with 3 hidden layers
- **Dataset**: Synthetic data generated using empirical drag coefficient formulas

## Physics Background

The drag coefficient (Cd) describes the drag force on an object in fluid flow:
```
F_drag = 0.5 * ρ * v² * A * Cd
```

For spheres, the drag coefficient depends on the Reynolds number:
```
Re = ρ * v * D / μ
```

The empirical formula used for data generation:
```
Cd = 24/Re + 6/(1+√Re) + 0.4
```

## Installation

```bash
cd "Predicting a Drag Coefficient"
pip install -r requirements.txt
```

## Usage

1. **Setup Environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Generate Dataset**:
```bash
python data_generation.py
```

3. **Train Model** (choose one):
```bash
# Scikit-learn MLP (recommended for CPU)
python drag_coefficient_sklearn_model.py

# PyTorch MLP (requires PyTorch installation)
python drag_coefficient_model.py
```

## Files

- `data_generation.py`: Generates synthetic drag coefficient data
- `drag_coefficient_sklearn_model.py`: Scikit-learn MLP implementation (recommended)
- `drag_coefficient_model.py`: PyTorch MLP implementation (optional)
- `visualization_utils.py`: Comprehensive visualization and analysis tools
- `requirements.txt`: Python dependencies
- `drag_coefficient_data.csv`: Generated dataset (1000 points)
- `models/`: Directory containing trained model files
- `visualizations/`: Directory with comprehensive analysis plots
- `outputs/`: Directory for additional output files

## Model Architecture

- **Input Layer**: 1 neuron (log10 of Reynolds number)
- **Hidden Layers**: 32 → 32 → 16 neurons with ReLU activation and dropout
- **Output Layer**: 1 neuron (drag coefficient)
- **Training**: Adam optimizer with learning rate scheduling

## Results Achieved

The trained model achieved excellent performance:
- **R² Score: 0.9954** (outstanding prediction accuracy)
- **RMSE: 2.56** (low prediction error)
- **MAPE: 18.42%** (reasonable percentage error)
- **Training time: ~10 seconds on CPU** (331 iterations)

### Physics Validation Results
- **Stokes Flow (Re < 1)**: 4.95% average error across 28 test points
- **Intermediate Flow (1 < Re < 1000)**: 14.48% average error across 106 test points  
- **Inertial Flow (Re > 1000)**: 30.45% average error across 66 test points

The model successfully captures the physics:
✓ 1/Re dependency at low Reynolds numbers (Stokes flow)
✓ Smooth transition in intermediate regime
✓ Approach to constant Cd at high Reynolds numbers

## Physics-Guided Neural Network Concept

This project demonstrates a **PgNN** approach where:
1. Physics knowledge guides data generation (empirical drag formulas)
2. Input features are physically meaningful (Reynolds number)
3. The neural network learns the complex non-linear relationship
4. Results can be interpreted within the physics context

This approach is more data-efficient than pure black-box ML and provides physically interpretable results.