# Drag Coefficient Prediction using Physics-Guided Neural Networks

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

1. **Generate Dataset**:
```bash
python data_generation.py
```

2. **Train Model**:
```bash
python drag_coefficient_model.py
```

## Files

- `data_generation.py`: Generates synthetic drag coefficient data
- `drag_coefficient_model.py`: MLP model implementation and training
- `requirements.txt`: Python dependencies
- `drag_coefficient_data.csv`: Generated dataset (after running data_generation.py)
- `drag_coefficient_model.pth`: Trained model weights (after training)

## Model Architecture

- **Input Layer**: 1 neuron (log10 of Reynolds number)
- **Hidden Layers**: 32 → 32 → 16 neurons with ReLU activation and dropout
- **Output Layer**: 1 neuron (drag coefficient)
- **Training**: Adam optimizer with learning rate scheduling

## Expected Results

The model should achieve:
- R² > 0.95 on test data
- RMSE < 0.1 for drag coefficient prediction
- Training time: ~30 seconds on CPU

## Physics-Guided Neural Network Concept

This project demonstrates a **PgNN** approach where:
1. Physics knowledge guides data generation (empirical drag formulas)
2. Input features are physically meaningful (Reynolds number)
3. The neural network learns the complex non-linear relationship
4. Results can be interpreted within the physics context

This approach is more data-efficient than pure black-box ML and provides physically interpretable results.