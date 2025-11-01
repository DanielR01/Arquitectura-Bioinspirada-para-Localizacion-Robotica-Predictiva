import torch
import numpy as np

def print_model_size(model):
    """
    Calculates and prints the total size of a model's parameters.
    """
    total_params = 0
    total_size_bytes = 0
    for param in model.parameters():
        total_params += param.numel()
        total_size_bytes += param.numel() * param.element_size()
    
    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
    print(f"\n--- Model Size ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Estimated Size: {total_size_gb:.2f} GB")
    print(f"------------------\n")

def get_win_rate_heatmap(win_rates, grid_shape=(20, 40)):
    """
    Generates a text-based heatmap grid for PRsM win rates, including a
    colored numerical grid showing the exact fire rates.
    
    Args:
        win_rates (torch.Tensor): A 1D tensor of win rates for each PRsM.
        grid_shape (tuple): The (rows, cols) shape for the heatmap grid.
        
    Returns:
        string: A formatted string containing the heatmap and active PRsM count.
    """
    # Ensure win_rates is a numpy array on the CPU for processing
    win_rates_np = win_rates.cpu().numpy()

    # Reshape the win rates into the desired grid
    if len(win_rates_np) != grid_shape[0] * grid_shape[1]:
        print(f"Warning: win_rates length ({len(win_rates_np)}) does not match grid size ({grid_shape[0]*grid_shape[1]}).")
        return "Error: Incompatible grid shape."
    
    heatmap_data = win_rates_np.reshape(grid_shape)

    # Normalize all win rates globally for color mapping
    max_rate = win_rates_np.max() + 1e-9
    
    visual_grid_str = ""
    numerical_grid_str = ""
    # Scale rates to be more readable integers (parts per 10,000)
    scaled_rates = (heatmap_data * 10000).astype(int)

    for row in range(grid_shape[0]):
        visual_row_str = ""
        numerical_row_str = ""
        for col in range(grid_shape[1]):
            rate = heatmap_data[row, col]
            norm_rate = rate / max_rate
            
            # Expanded color mapping for console
            if norm_rate == 0:
                color_char = "\033[38;5;232m" # Dark Grey for zero
            elif norm_rate < 0.1:
                color_char = "\033[38;5;21m"  # Dark Blue
            elif norm_rate < 0.2:
                color_char = "\033[38;5;33m"
            elif norm_rate < 0.3:
                color_char = "\033[38;5;45m"
            elif norm_rate < 0.4:
                color_char = "\033[38;5;50m"
            elif norm_rate < 0.5:
                color_char = "\033[38;5;48m"
            elif norm_rate < 0.6:
                color_char = "\033[38;5;46m"  # Green
            elif norm_rate < 0.7:
                color_char = "\033[38;5;154m"
            elif norm_rate < 0.8:
                color_char = "\033[38;5;190m" # Yellow
            elif norm_rate < 0.9:
                color_char = "\033[38;5;208m" # Orange
            else:
                color_char = "\033[38;5;196m" # Red
            
            # Build the visual grid row
            visual_row_str += f"{color_char}██\033[0m"
            
            # Build the numerical grid row with the same color
            numerical_row_str += f"{color_char}{scaled_rates[row, col]:4d}\033[0m"

        visual_grid_str += visual_row_str + "\n"
        numerical_grid_str += numerical_row_str + "\n"
        
    # Calculate number of active PRsMs
    active_threshold = 1e-6
    num_active = np.sum(win_rates_np > active_threshold)
    
    return (f"Usage Heatmap (Visual):\n{visual_grid_str}"
            f"Usage Heatmap (Numeric, rate * 10000):\n{numerical_grid_str}"
            f"Active PRsMs: {num_active}/{len(win_rates_np)}")
