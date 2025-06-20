#!/usr/bin/env python
"""
Visualizes results from multiple conjugate gradient runs.
Reads data from results.txt and creates plots similar to those originally in applianceCompileAndRun.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate  # Add this import at the top of the file

def read_results_file(filename="results.txt"):
    """Read the results file and parse into arrays"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file {filename} not found. Run applianceCompileAndRun.py first.")
    
    # Initialize arrays
    size_values = []
    h_values = []
    cpu_times = []
    wse_times = []
    cpu_max_errors = []
    wse_max_errors = []
    cpu_rmse = []      
    wse_rmse = []      
    cpu_iters = []
    wse_iters = []
    
    # Read file and extract data
    with open(filename, "r") as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip header line
                
            # Parse line
            values = line.strip().split()
            if len(values) < 10:  # Expects 10 values per line
                print(f"Warning: Skipping incomplete line: {line.strip()}")
                continue
                
            size_values.append(int(values[0]))
            h_values.append(float(values[1]))
            cpu_times.append(float(values[2]))
            wse_times.append(float(values[3]))
            cpu_max_errors.append(float(values[4]))
            wse_max_errors.append(float(values[5]))
            cpu_rmse.append(float(values[6]))
            wse_rmse.append(float(values[7]))
            cpu_iters.append(int(values[8]))
            wse_iters.append(int(values[9]))
    
    # Sort all data by size
    sorted_indices = np.argsort(size_values)
    size_values = [size_values[i] for i in sorted_indices]
    h_values = [h_values[i] for i in sorted_indices]
    cpu_times = [cpu_times[i] for i in sorted_indices]
    wse_times = [wse_times[i] for i in sorted_indices]
    cpu_max_errors = [cpu_max_errors[i] for i in sorted_indices]
    wse_max_errors = [wse_max_errors[i] for i in sorted_indices]
    cpu_rmse = [cpu_rmse[i] for i in sorted_indices]
    wse_rmse = [wse_rmse[i] for i in sorted_indices]
    cpu_iters = [cpu_iters[i] for i in sorted_indices]
    wse_iters = [wse_iters[i] for i in sorted_indices]
    
    return (size_values, h_values, cpu_times, wse_times,
            cpu_max_errors, wse_max_errors, cpu_rmse, wse_rmse, cpu_iters, wse_iters)

def calculate_convergence_rates(sizes, errors):
    """Calculate convergence rates between consecutive grid sizes"""
    rates = []
    for i in range(1, len(sizes)):
        # Calculate h₁/h₂
        h_ratio = (1.0/(sizes[i-1]-1)) / (1.0/(sizes[i]-1))
        # Calculate log(error₁/error₂)/log(h₁/h₂)
        if errors[i] > 0:  # Avoid division by zero or log of zero
            rate = np.log(errors[i-1]/errors[i]) / np.log(h_ratio)
            rates.append(rate)
        else:
            rates.append(np.nan)
    return rates

def create_convergence_plot(size_values, cpu_max_errors, wse_max_errors, cpu_rmse, wse_rmse):
    """Create the error convergence plots (max error and RMSE)"""
    # Create max error plot
    plt.figure(figsize=(12, 8))
    plt.loglog(size_values, cpu_max_errors, 'bo-', linewidth=2, label='CPU Maximum Error')
    
    # Filter valid WSE results (non-zero)
    valid_wse = [(s, e) for s, e in zip(size_values, wse_max_errors) if e > 0]
    if valid_wse:
        wse_sizes = [v[0] for v in valid_wse]
        wse_errors_filtered = [v[1] for v in valid_wse]
        plt.loglog(wse_sizes, wse_errors_filtered, 'ro-', linewidth=2, label='WSE Maximum Error')
    
    # Calculate reference O(h²) line
    if size_values:
        scaling_factor = cpu_max_errors[0] * (size_values[0]**2)
        reference_values = [scaling_factor / (n**2) for n in size_values]
        plt.loglog(size_values, reference_values, 'k--', linewidth=1, label='O(h²) Reference')
    
    plt.xlabel('Grid Points per Dimension (n)')
    plt.ylabel('Maximum Absolute Error')
    plt.title('Convergence of Maximum Error (Conjugate Gradient Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('cg_max_error_convergence.png', dpi=300)
    
    # Calculate and print convergence rates for max error
    cpu_max_rates = calculate_convergence_rates(size_values, cpu_max_errors)
    print("\nCPU Maximum Error Convergence Rates:")
    for i in range(len(cpu_max_rates)):
        print(f"  From size {size_values[i]} to {size_values[i+1]}: {cpu_max_rates[i]:.4f}")
    
    if valid_wse:
        wse_max_rates = calculate_convergence_rates(wse_sizes, wse_errors_filtered)
        print("\nWSE Maximum Error Convergence Rates:")
        for i in range(len(wse_max_rates)):
            print(f"  From size {wse_sizes[i]} to {wse_sizes[i+1]}: {wse_max_rates[i]:.4f}")
    
    # Create RMSE plot
    plt.figure(figsize=(12, 8))
    plt.loglog(size_values, cpu_rmse, 'bs-', linewidth=2, label='CPU RMSE')
    
    # Filter valid WSE results (non-zero)
    valid_wse_rmse = [(s, e) for s, e in zip(size_values, wse_rmse) if e > 0]
    if valid_wse_rmse:
        wse_rmse_sizes = [v[0] for v in valid_wse_rmse]
        wse_rmse_filtered = [v[1] for v in valid_wse_rmse]
        plt.loglog(wse_rmse_sizes, wse_rmse_filtered, 'rs-', linewidth=2, label='WSE RMSE')
    
    # Calculate reference O(h²) line for RMSE
    if size_values:
        scaling_factor_rmse = cpu_rmse[0] * (size_values[0]**2)
        reference_values_rmse = [scaling_factor_rmse / (n**2) for n in size_values]
        plt.loglog(size_values, reference_values_rmse, 'k--', linewidth=1, label='O(h²) Reference')
    
    plt.xlabel('Grid Points per Dimension (n)')
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.title('Convergence of RMSE (Conjugate Gradient Method)')
    plt.legend()
    plt.grid(True)
    plt.savefig('cg_rmse_convergence.png', dpi=300)
    
    # Calculate and print convergence rates for RMSE
    cpu_rmse_rates = calculate_convergence_rates(size_values, cpu_rmse)
    print("\nCPU RMSE Convergence Rates:")
    for i in range(len(cpu_rmse_rates)):
        print(f"  From size {size_values[i]} to {size_values[i+1]}: {cpu_rmse_rates[i]:.4f}")
    
    if valid_wse_rmse:
        wse_rmse_rates = calculate_convergence_rates(wse_rmse_sizes, wse_rmse_filtered)
        print("\nWSE RMSE Convergence Rates:")
        for i in range(len(wse_rmse_rates)):
            print(f"  From size {wse_rmse_sizes[i]} to {wse_rmse_sizes[i+1]}: {wse_rmse_rates[i]:.4f}")

def create_performance_plot(size_values, cpu_times, wse_times, cpu_iters, wse_iters):
    """Create performance comparison plots"""
    # Execution time plot
    plt.figure(figsize=(12, 8))
    plt.plot(size_values, cpu_times, 'bo-', linewidth=2, label='CPU Time (s)')
    
    # Filter valid WSE results (non-zero)
    valid_wse = [(s, t) for s, t in zip(size_values, wse_times) if t > 0]
    if valid_wse:
        wse_sizes = [v[0] for v in valid_wse]
        wse_times_filtered = [v[1] for v in valid_wse]
        plt.plot(wse_sizes, wse_times_filtered, 'ro-', linewidth=2, label='WSE Time (s)')
    
    plt.xlabel('Grid Points per Dimension (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance of Conjugate Gradient Method')
    plt.legend()
    plt.grid(True)
    plt.savefig('cg_performance.png', dpi=300)
    
    # Create iteration count plot
    plt.figure(figsize=(12, 8))
    plt.plot(size_values, cpu_iters, 'bs-', linewidth=2, label='CPU Iterations')
    
    # Filter valid WSE results (non-zero)
    valid_wse_iters = [(s, i) for s, i in zip(size_values, wse_iters) if i > 0]
    if valid_wse_iters:
        wse_iter_sizes = [v[0] for v in valid_wse_iters]
        wse_iters_filtered = [v[1] for v in valid_wse_iters]
        plt.plot(wse_iter_sizes, wse_iters_filtered, 'rs-', linewidth=2, label='WSE Iterations')
    
    plt.xlabel('Grid Points per Dimension (n)')
    plt.ylabel('Number of Iterations')
    plt.title('Convergence Iterations for Conjugate Gradient Method')
    plt.legend()
    plt.grid(True)
    plt.savefig('cg_iterations.png', dpi=300)

def print_results_table(size_values, h_values, cpu_times, wse_times, 
                      cpu_max_errors, wse_max_errors,
                      cpu_rmse, wse_rmse,
                      cpu_iters, wse_iters):
    """Print a formatted table of the results to stdout"""
    
    # Calculate convergence rates
    cpu_max_error_rates = ['N/A'] + [f"{r:.2f}" for r in calculate_convergence_rates(size_values, cpu_max_errors)]
    cpu_rmse_rates = ['N/A'] + [f"{r:.2f}" for r in calculate_convergence_rates(size_values, cpu_rmse)]
    
    # Prepare WSE error rates (if any WSE data exists)
    if any(e > 0 for e in wse_max_errors):
        valid_wse = [(i, s, e) for i, (s, e) in enumerate(zip(size_values, wse_max_errors)) if e > 0]
        if len(valid_wse) > 1:  # Need at least 2 points for convergence rate
            wse_idx = [v[0] for v in valid_wse]
            wse_sizes = [v[1] for v in valid_wse]
            wse_errors = [v[2] for v in valid_wse]
            wse_max_error_rates = calculate_convergence_rates(wse_sizes, wse_errors)
            
            # Create full list with rates at the right positions
            full_wse_rates = ['N/A'] * len(size_values)
            for i in range(1, len(wse_idx)):
                full_wse_rates[wse_idx[i]] = f"{wse_max_error_rates[i-1]:.2f}"
        else:
            full_wse_rates = ['N/A'] * len(size_values)
    else:
        full_wse_rates = ['N/A'] * len(size_values)
    
    # Similarly for WSE RMSE rates
    if any(e > 0 for e in wse_rmse):
        valid_wse_rmse = [(i, s, e) for i, (s, e) in enumerate(zip(size_values, wse_rmse)) if e > 0]
        if len(valid_wse_rmse) > 1:
            wse_rmse_idx = [v[0] for v in valid_wse_rmse]
            wse_rmse_sizes = [v[1] for v in valid_wse_rmse]
            wse_rmse_errors = [v[2] for v in valid_wse_rmse]
            wse_rmse_rates = calculate_convergence_rates(wse_rmse_sizes, wse_rmse_errors)
            
            full_wse_rmse_rates = ['N/A'] * len(size_values)
            for i in range(1, len(wse_rmse_idx)):
                full_wse_rmse_rates[wse_rmse_idx[i]] = f"{wse_rmse_rates[i-1]:.2f}"
        else:
            full_wse_rmse_rates = ['N/A'] * len(size_values)
    else:
        full_wse_rmse_rates = ['N/A'] * len(size_values)
    
    # Format all data for table
    data = []
    headers = [
        "Grid\nSize",
        "Grid\nSpacing",
        "CPU Time\n(s)",
        "WSE Time\n(s)",
        "CPU\nIters",
        "WSE\nIters",
        "CPU Max\nError",
        "CPU Max\nRate",
        "WSE Max\nError",
        "WSE Max\nRate",
        "CPU\nRMSE",
        "CPU RMSE\nRate",
        "WSE\nRMSE",
        "WSE RMSE\nRate"
    ]
    
    for i in range(len(size_values)):
        row = [
            size_values[i],
            f"{h_values[i]:.4f}",
            f"{cpu_times[i]:.4f}",
            f"{wse_times[i]:.4f}" if wse_times[i] > 0 else "N/A",
            cpu_iters[i],
            wse_iters[i] if wse_iters[i] > 0 else "N/A",
            f"{cpu_max_errors[i]:.2e}",
            cpu_max_error_rates[i],
            f"{wse_max_errors[i]:.2e}" if wse_max_errors[i] > 0 else "N/A",
            full_wse_rates[i],
            f"{cpu_rmse[i]:.2e}",
            cpu_rmse_rates[i],
            f"{wse_rmse[i]:.2e}" if wse_rmse[i] > 0 else "N/A",
            full_wse_rmse_rates[i]
        ]
        data.append(row)
    
    # Print the table
    try:
        print("\nConjugate Gradient Method Results:")
        print(tabulate(data, headers=headers, tablefmt="grid"))
    except ImportError:
        # If tabulate is not available, fall back to a simpler table
        print("\nConjugate Gradient Method Results:")
        print("-" * 120)
        print("  ".join(h.replace("\n", " ") for h in headers))
        print("-" * 120)
        for row in data:
            print("  ".join(str(cell).ljust(10) for cell in row))
        print("-" * 120)

def main():
    """Main function to read results and create plots"""
    print("Reading results data...")
    try:
        results = read_results_file()
        (size_values, h_values, cpu_times, wse_times,
         cpu_max_errors, wse_max_errors, cpu_rmse, wse_rmse, cpu_iters, wse_iters) = results
        
        print(f"Found data for {len(size_values)} grid sizes: {size_values}")
        
        # Print data table to stdout
        print_results_table(
            size_values, h_values, cpu_times, wse_times, 
            cpu_max_errors, wse_max_errors,
            cpu_rmse, wse_rmse,
            cpu_iters, wse_iters
        )
        
        print("\nCreating plots...")
        create_convergence_plot(size_values, cpu_max_errors, wse_max_errors, cpu_rmse, wse_rmse)
        create_performance_plot(size_values, cpu_times, wse_times, cpu_iters, wse_iters)
        
        print("\nPlots saved:")
        print("  - cg_max_error_convergence.png")
        print("  - cg_rmse_convergence.png")
        print("  - cg_performance.png")
        print("  - cg_iterations.png")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        
if __name__ == "__main__":
    main()