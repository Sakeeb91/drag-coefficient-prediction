import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_drag_coefficient_data(n_samples=1000, noise_level=0.05):
    """
    Generate synthetic drag coefficient data for spheres in fluid flow.
    
    Based on the standard drag coefficient formula for spheres:
    Cd = 24/Re + 6/(1+sqrt(Re)) + 0.4  (for Re < 2e5)
    
    Parameters:
    - n_samples: Number of data points to generate
    - noise_level: Amount of noise to add to the data
    
    Returns:
    - DataFrame with Reynolds number and drag coefficient
    """
    
    # Generate Reynolds numbers in log scale (common in fluid mechanics)
    Re_min, Re_max = 0.1, 1e5
    Re = np.logspace(np.log10(Re_min), np.log10(Re_max), n_samples)
    
    # Calculate drag coefficient using empirical formula
    Cd = 24/Re + 6/(1 + np.sqrt(Re)) + 0.4
    
    # Add some realistic noise
    noise = np.random.normal(0, noise_level * Cd, n_samples)
    Cd_noisy = Cd + noise
    
    # Ensure drag coefficient is positive
    Cd_noisy = np.maximum(Cd_noisy, 0.1)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Reynolds_number': Re,
        'drag_coefficient': Cd_noisy
    })
    
    return data

def plot_drag_data(data):
    """Plot the generated drag coefficient data"""
    plt.figure(figsize=(10, 6))
    plt.loglog(data['Reynolds_number'], data['drag_coefficient'], 'b.', alpha=0.6, markersize=3)
    plt.xlabel('Reynolds Number (Re)')
    plt.ylabel('Drag Coefficient (Cd)')
    plt.title('Drag Coefficient vs Reynolds Number for Spheres')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Generate dataset
    data = generate_drag_coefficient_data(n_samples=1000)
    
    # Save to CSV
    data.to_csv('drag_coefficient_data.csv', index=False)
    print(f"Generated {len(data)} data points")
    print(f"Reynolds number range: {data['Reynolds_number'].min():.2f} to {data['Reynolds_number'].max():.2e}")
    print(f"Drag coefficient range: {data['drag_coefficient'].min():.3f} to {data['drag_coefficient'].max():.3f}")
    
    # Plot the data
    plot_drag_data(data)