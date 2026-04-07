import numpy as np
import matplotlib.pyplot as plt

def gaussian_pulse(x: np.ndarray, pulse_width: float = 2.0e-4, 
                   pulse_shift: float = 1.0e-1) -> np.ndarray:
    """
    Generate a Gaussian pulse.
    
    Parameters:
    -----------
    x : np.ndarray
        Spatial coordinates
    pulse_width : float
        Width parameter of the Gaussian pulse
    pulse_shift : float
        Initial position of the pulse center
        
    Returns:
    --------
    np.ndarray
        Gaussian pulse values at positions x
    """
    return (1 / np.sqrt(pulse_width * np.pi) * 
            np.exp(-((x - pulse_shift)**2) / pulse_width))


def generate_advecting_pulse(
    pulse_width: float = 2.0e-4,
    pulse_shift: float = 1.0e-1,
    speed: float = 10.0,
    final_time: float = 0.1,
    n_time_samples: int = 2000,
    n_space_samples: int = 4096,
    x_min: float = 0.0,
    x_max: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for an advecting Gaussian pulse.
    
    Parameters:
    -----------
    pulse_width : float
        Width parameter of the Gaussian pulse
    pulse_shift : float
        Initial position of the pulse center
    speed : float
        Advection speed of the pulse
    final_time : float
        Final simulation time
    n_time_samples : int
        Number of time samples
    n_space_samples : int
        Number of spatial grid points
    x_min, x_max : float
        Spatial domain bounds
        
    Returns:
    --------
    tuple
        (data_matrix, x_grid, t_grid) where:
        - data_matrix: (n_space_samples, n_time_samples) array
        - x_grid: spatial coordinates
        - t_grid: time coordinates
    """
    # Create spatial and temporal grids
    x = np.linspace(x_min, x_max, n_space_samples, dtype=np.float64)
    t = np.linspace(0, final_time, n_time_samples, dtype=np.float64)
    
    # Initialize data matrix
    data_matrix = np.zeros((n_space_samples, n_time_samples), dtype=np.float64)
    
    # Generate pulse at each time step
    for i, ti in enumerate(t):
        # Pulse position at time ti
        pulse_position = pulse_shift + speed * ti
        
        # Handle periodic boundary conditions (optional)
        # Uncomment if you want the pulse to wrap around
        # pulse_position = pulse_position % (x_max - x_min)
        
        # Generate pulse at current time
        data_matrix[:, i] = gaussian_pulse(x, pulse_width, pulse_position)
    
    return data_matrix, x, t


def generate_advecting_pulse_periodic(
    pulse_width: float = 2.0e-4,
    pulse_shift: float = 1.0e-1,
    speed: float = 10.0,
    final_time: float = 0.1,
    n_time_samples: int = 2000,
    n_space_samples: int = 4096,
    x_min: float = 0.0,
    x_max: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate advecting pulse with periodic boundary conditions.
    """
    x = np.linspace(x_min, x_max, n_space_samples, dtype=np.float64)
    t = np.linspace(0, final_time, n_time_samples, dtype=np.float64)
    
    data_matrix = np.zeros((n_space_samples, n_time_samples), dtype=np.float64)
    domain_length = x_max - x_min
    
    for i, ti in enumerate(t):
        # Pulse position with periodic wrapping
        pulse_position = (pulse_shift + speed * ti) % domain_length
        data_matrix[:, i] = gaussian_pulse(x, pulse_width, pulse_position)
    
    return data_matrix, x, t


if __name__ == "__main__":
    # Example usage
    print("Generating advecting pulse data...")
    
    # Generate data
    data, x_grid, t_grid = generate_advecting_pulse(
        pulse_width=2.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.15,
        n_time_samples=1000,
        n_space_samples=512
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Spatial domain: [{x_grid[0]:.3f}, {x_grid[-1]:.3f}]")
    print(f"Time domain: [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 3D surface
    T, X = np.meshgrid(t_grid, x_grid)
    surf = ax1.contourf(T, X, data, levels=50, cmap='viridis')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Space')
    ax1.set_title('Advecting Gaussian Pulse')
    plt.colorbar(surf, ax=ax1)
    
    # Plot snapshots at different times
    time_indices = [0, len(t_grid)//4, len(t_grid)//2, 3*len(t_grid)//4, -1]
    for idx in time_indices:
        ax2.plot(x_grid, data[:, idx], 
                label=f't = {t_grid[idx]:.3f}', linewidth=2)
    
    ax2.set_xlabel('Space')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Pulse Snapshots')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save data if needed
    # np.savez('advecting_pulse_data.npz', 
    #          data=data, x=x_grid, t=t_grid)