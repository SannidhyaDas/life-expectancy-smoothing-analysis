import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  --- HELPER FUNCTIONS & CLASSES ---

def clean_column_names(df):
    """Cleans DataFrame column names to snake_case."""
    cols = df.columns
    cols = [c.strip().lower() for c in cols]
    cols = [c.replace(' ', '_').replace('-', '_').replace('/', '_') for c in cols]
    df.columns = cols
    return df

def plot_cv_results(params, errors, param_name, model_name, log_scale=False):
    """Plots cross-validation error curves and saves the figure."""
    plt.figure(figsize=(10, 5))
    plt.plot(params, errors, marker='o', linestyle='--')
    if log_scale:
        plt.xscale('log')
    plt.title(f'5-Fold CV Error for {model_name}')
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    
    # Saves the figure
    filename = f'cv_plot_{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(filename)
    print(f"Saved CV plot: {filename}")
    
    # Displays the plot
    plt.show()

# --- Bin Smoother Class ---
class BinSmoother:
    def __init__(self, n_bins):
        self.n_bins = n_bins
    
    def fit(self, x, y):
        # Create bin edges based on percentiles
        self.bin_edges = np.percentile(x, np.linspace(0, 100, self.n_bins + 1))
        # Ensure the last edge is the max value to include all points
        self.bin_edges[-1] = np.max(x)
        # Digitize x to get bin indices for each point
        bin_indices = np.digitize(x, self.bin_edges[1:-1])
        
        # Calculate the mean of y for each bin
        self.bin_means = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                self.bin_means[i] = np.mean(y[bin_mask])
            else:
                # Handle empty bins by borrowing from the previous bin if possible
                self.bin_means[i] = self.bin_means[i-1] if i > 0 else 0
    
    def predict(self, x):
        # Digitize the new x values to find their bin indices
        bin_indices = np.digitize(x, self.bin_edges[1:-1])
        # Return the pre-calculated mean for that bin
        return self.bin_means[bin_indices]

# --- Kernel Smoother (Nadaraya-Watson) ---
def kernel_smoother(x_train, y_train, x_val, bandwidth):
    """
    Performs Gaussian kernel smoothing (Nadaraya-Watson).
    """
    predictions = []
    for x in x_val.squeeze():
        # Calculate kernel weights for all training points
        u = (x_train.squeeze() - x) / bandwidth
        gaussian_kernel = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        
        # Checks for non-zero weights
        if np.sum(gaussian_kernel) > 0:
            # Return the weighted average
            weighted_avg = np.sum(gaussian_kernel * y_train) / np.sum(gaussian_kernel)
            predictions.append(weighted_avg)
        else:
            # Handle cases where all weights are zero (e.g., bandwidth too small)
            predictions.append(np.mean(y_train)) # Fallback
            
    return np.array(predictions)

# --- Local Regression (Tricube) ---
def locally_weighted_regression(x_val, x_train, y_train, bandwidth):
    """
    Performs local linear regression with a tricube kernel.
    """
    x_val = np.array([1, x_val]) # Adds intercept term
    x_train_with_intercept = np.vstack([np.ones(len(x_train)), x_train.squeeze()]).T
    
    # Calculate distances and scale
    distances = np.abs(x_train.squeeze() - x_val[1])
    u = distances / bandwidth
    
    # Tricube weights
    tricube_weights = np.zeros_like(u)
    mask = u < 1
    tricube_weights[mask] = (1 - u[mask]**3)**3
    
    W = np.diag(tricube_weights)
    
    try:
        # Solve weighted least squares: (X'WX)^-1 * X'Wy
        XTWX = x_train_with_intercept.T @ W @ x_train_with_intercept
        XTWy = x_train_with_intercept.T @ W @ y_train
        beta = np.linalg.solve(XTWX, XTWy)
        
        # Prediction is beta_0 * 1 + beta_1 * x_val
        return x_val @ beta
    except np.linalg.LinAlgError:
        # Handle singular matrix (e.g., no points within bandwidth)
        return np.mean(y_train) # Fallback

print("Helper functions and classes defined.")