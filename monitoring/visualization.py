#FILE: monitoring/visualization.py
# ehsanasgharzde - VISUALIZATION
# hosseinsolymanzadeh - PROPER COMMENTING

import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import logging
from datetime import datetime, timedelta
from dataclasses import asdict

from .logger import MonitoringLogger
from .config import MonitoringConfig
from .system_monitor import SystemMetrics
from .data_monitor import DataMetrics
from .training_monitor import TrainingMonitor

sns.set_style("whitegrid")  # Set seaborn style for plots to "whitegrid" for better readability

class MonitoringVisualizer:
    def __init__(self, config: MonitoringConfig, logger: MonitoringLogger):
        # Initialize with config and logger instances
        self.config = config
        self.logger = logger
        
        # Set output directory for saving plots
        self.output_dir = config.output_directory
        # Flag to enable or disable interactive visualization
        self.interactive = config.enable_visualization
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuration parameters for saving plots
        self.plot_config = {
            "dpi": 150,               # Plot resolution
            "bbox_inches": "tight",   # Bounding box tight to content
            "facecolor": "white",     # Background color of the saved figure
            "edgecolor": "none"       # No edge color on saved figure
        }
        
        # Define a color palette with 8 distinct colors for plotting multiple lines
        self.color_palette = sns.color_palette("husl", 8)
        
        # Update matplotlib global font sizes and style settings for plots
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })

    def _save_plot(self, fig, filename: str, close_after: bool = True):
        # Compose full file path for saving the figure
        save_path = os.path.join(self.output_dir, filename)
        
        # Save the figure with pre-defined plot configuration
        fig.savefig(save_path, **self.plot_config)
        
        # Log the event that a plot was saved, including the file path
        self.logger.log_training_event("plot_saved", {"path": save_path})
        
        # Optionally close the figure to free memory after saving
        if close_after:
            plt.close(fig)
        
        # Return the full path where the figure was saved
        return save_path

    def _create_time_series_plot(self, data: List[Dict[str, Any]], 
                                 y_keys: List[str], 
                                 title: str,
                                 ylabel: str = "Value",
                                 colors: Optional[List[str]] = None) -> plt.Figure: #type: ignore 
        # Create a new figure and axis for time series plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Handle the case where no data is provided
        if not data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract timestamps from data entries or use current UTC time as fallback
        times = [entry.get('timestamp', datetime.utcnow()) for entry in data]
        
        # Plot each y_key as a separate line on the graph
        for i, key in enumerate(y_keys):
            # Extract values for the current key from the data
            values = [entry.get(key, 0) for entry in data]
            # Choose color from provided colors list or fallback to color palette
            color = colors[i] if colors and i < len(colors) else self.color_palette[i % len(self.color_palette)]
            # Plot time series line for current key
            ax.plot(times, values, label=key, color=color, linewidth=2, alpha=0.8)
        
        # Set x-axis label
        ax.set_xlabel('Time')
        # Set y-axis label with given ylabel
        ax.set_ylabel(ylabel)
        # Set plot title
        ax.set_title(title)
        # Add legend to identify lines
        ax.legend()
        # Enable grid with light transparency
        ax.grid(True, alpha=0.3)
        # Rotate x-axis tick labels for better readability
        plt.xticks(rotation=45)
        # Adjust layout to fit elements nicely
        plt.tight_layout()
        
        # Return the created figure object
        return fig

    def plot_system_metrics(self, metrics_history: List[SystemMetrics]) -> str:
        # Check if metrics data is available; warn and return empty string if not
        if not metrics_history:
            logging.warning("No system metrics data available for visualization")
            return ""
        
        # Convert list of SystemMetrics objects to list of dictionaries
        data = [asdict(m) for m in metrics_history]
        
        # Create time series plot for CPU, memory, and disk usage percentages
        fig = self._create_time_series_plot(
            data,
            ['cpu_percent', 'memory_percent', 'disk_usage'],
            'System Resource Utilization',
            'Percentage (%)',
            ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Custom colors for each metric line
        )
        
        # Save the plot and return the saved file path
        return self._save_plot(fig, "system_metrics.png")

    def plot_gpu_metrics(self, metrics_history: List[SystemMetrics]) -> str:
        # Check if GPU metrics exist and are non-empty; warn and return empty string if not
        if not metrics_history or not metrics_history[0].gpu_utilization:
            logging.warning("No GPU metrics data available for visualization")
            return ""
        
        # Create a 2x2 grid of subplots for different GPU metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract timestamps from metrics history for x-axis
        times = [m.timestamp for m in metrics_history]
        
        # Calculate mean GPU utilization per timestamp or zero if data missing
        gpu_utils = [np.mean(m.gpu_utilization) if m.gpu_utilization else 0 for m in metrics_history]
        # Calculate mean GPU temperature per timestamp or zero if missing
        gpu_temps = [np.mean(m.gpu_temperature) if m.gpu_temperature else 0 for m in metrics_history]
        # Calculate mean GPU memory used per timestamp or zero if missing
        gpu_mem_used = [np.mean(m.gpu_memory_used) if m.gpu_memory_used else 0 for m in metrics_history]
        # Calculate mean GPU total memory per timestamp or zero if missing
        gpu_mem_total = [np.mean(m.gpu_memory_total) if m.gpu_memory_total else 0 for m in metrics_history]
        
        # Plot GPU utilization over time on first subplot
        axes[0, 0].plot(times, gpu_utils, color='#FF6B6B', linewidth=2)
        axes[0, 0].set_title('GPU Utilization (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot GPU temperature over time on second subplot
        axes[0, 1].plot(times, gpu_temps, color='#FFA07A', linewidth=2)
        axes[0, 1].set_title('GPU Temperature (°C)')
        axes[0, 1].set_ylabel('Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot GPU memory used and total memory on third subplot
        axes[1, 0].plot(times, gpu_mem_used, color='#4ECDC4', linewidth=2, label='Used')
        axes[1, 0].plot(times, gpu_mem_total, color='#45B7D1', linewidth=2, label='Total')
        axes[1, 0].set_title('GPU Memory (MB)')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calculate GPU memory usage percentage and plot on fourth subplot if total memory exists
        if gpu_mem_total:
            mem_usage_pct = [(used/total)*100 if total > 0 else 0 for used, total in zip(gpu_mem_used, gpu_mem_total)]
            axes[1, 1].plot(times, mem_usage_pct, color='#96CEB4', linewidth=2)
            axes[1, 1].set_title('GPU Memory Usage (%)')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Adjust layout for all subplots to prevent overlap
        plt.tight_layout()
        
        # Save the GPU metrics plot and return the saved file path
        return self._save_plot(fig, "gpu_metrics.png")

    def plot_training_progress(self, training_monitor: TrainingMonitor) -> str:
        # Check if there are any loss values recorded; log warning and return empty if none
        if not training_monitor.losses: #type: ignore 
            logging.warning("No training data available for visualization")
            return ""

        # Create a 2x2 grid of subplots for various training progress metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract step numbers, loss values, and learning rates from the training monitor
        steps = training_monitor.steps #type: ignore 
        losses = training_monitor.losses #type: ignore 
        learning_rates = training_monitor.learning_rates #type: ignore 

        # Plot training loss over steps on the first subplot
        axes[0, 0].plot(steps, losses, color='#FF6B6B', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')  # Use logarithmic scale for better visualization of loss

        # Plot learning rate schedule over steps on the second subplot
        axes[0, 1].plot(steps, learning_rates, color='#4ECDC4', linewidth=2)
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')  # Log scale to better see rate changes

        # If convergence info is available, plot the loss convergence rate on third subplot
        if training_monitor.convergence_info: #type: ignore 
            conv_data = list(training_monitor.convergence_info) #type: ignore 
            axes[1, 0].plot(conv_data, color='#45B7D1', linewidth=2)
            axes[1, 0].set_title('Loss Convergence Rate')
            axes[1, 0].set_xlabel('Recent Steps')
            axes[1, 0].set_ylabel('Loss Delta')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')  # Log scale to highlight convergence dynamics

        # If batch processing times are recorded, plot them on the fourth subplot
        if training_monitor.batch_times: #type: ignore 
            batch_times = training_monitor.batch_times #type: ignore 
            axes[1, 1].plot(batch_times, color='#96CEB4', linewidth=2)
            axes[1, 1].set_title('Batch Processing Time')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].grid(True, alpha=0.3)

        # Adjust subplot layout to avoid overlap
        plt.tight_layout()
        # Save the training progress plot and return the saved file path
        return self._save_plot(fig, "training_progress.png")

    def plot_data_distribution(self, data_metrics: List[DataMetrics], title: str = "Data Distribution") -> str:
        # Check if data metrics list is empty; log warning and return empty string if so
        if not data_metrics:
            logging.warning("No data metrics available for visualization")
            return ""

        # Create a 2x2 grid of subplots for different statistical distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract mean, std deviation, NaN counts, and Inf counts from data metrics
        means = [m.mean_value for m in data_metrics]
        stds = [m.std_value for m in data_metrics]
        nan_counts = [m.nan_count for m in data_metrics]
        inf_counts = [m.inf_count for m in data_metrics]

        # Plot histogram of mean values on the first subplot
        axes[0, 0].hist(means, bins=30, alpha=0.7, color='#FF6B6B', edgecolor='black')
        axes[0, 0].set_title('Mean Values Distribution')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot histogram of standard deviations on the second subplot
        axes[0, 1].hist(stds, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
        axes[0, 1].set_title('Standard Deviation Distribution')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot bar chart of NaN counts per tensor on the third subplot
        axes[1, 0].bar(range(len(nan_counts)), nan_counts, color='#FFA07A', alpha=0.7)
        axes[1, 0].set_title('NaN Count per Tensor')
        axes[1, 0].set_xlabel('Tensor Index')
        axes[1, 0].set_ylabel('NaN Count')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot bar chart of Inf counts per tensor on the fourth subplot
        axes[1, 1].bar(range(len(inf_counts)), inf_counts, color='#45B7D1', alpha=0.7)
        axes[1, 1].set_title('Inf Count per Tensor')
        axes[1, 1].set_xlabel('Tensor Index')
        axes[1, 1].set_ylabel('Inf Count')
        axes[1, 1].grid(True, alpha=0.3)

        # Set a suptitle for the entire figure
        plt.suptitle(title, fontsize=16)
        # Adjust layout to prevent overlap of subplots and title
        plt.tight_layout()
        # Save the data distribution plot and return the saved file path
        return self._save_plot(fig, "data_distribution.png")

    def plot_gradient_flow(self, gradient_stats: Dict[str, List[float]]) -> str:
        # Check if gradient stats dictionary is empty; warn and return empty string if so
        if not gradient_stats:
            logging.warning("No gradient statistics available for visualization")
            return ""
        
        # Create two vertical subplots for average and maximum gradients per layer
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Extract layer names from the keys of the gradient stats dictionary
        layers = list(gradient_stats.keys())
        # Compute average gradient norm for each layer or 0 if empty
        avg_grads = [np.mean(values) if values else 0 for values in gradient_stats.values()]
        # Compute maximum gradient norm for each layer or 0 if empty
        max_grads = [np.max(values) if values else 0 for values in gradient_stats.values()]
        
        # Generate positions for the x-axis ticks based on number of layers
        x_pos = np.arange(len(layers))
        
        # Plot average gradients as a bar chart on the first subplot
        axes[0].bar(x_pos, avg_grads, alpha=0.7, color='#FF6B6B', edgecolor='black')
        axes[0].set_title('Average Gradient Norms by Layer')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Average Gradient Norm')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(layers, rotation=45, ha='right')  # Rotate layer labels for readability
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Log scale to better visualize gradient magnitudes
        
        # Plot maximum gradients as a bar chart on the second subplot
        axes[1].bar(x_pos, max_grads, alpha=0.7, color='#4ECDC4', edgecolor='black')
        axes[1].set_title('Maximum Gradient Norms by Layer')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Maximum Gradient Norm')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(layers, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Adjust layout to prevent overlap of elements
        plt.tight_layout()
        # Save the figure and return saved path
        return self._save_plot(fig, "gradient_flow.png")
    
    def plot_memory_usage(self, metrics_history: List[SystemMetrics]) -> str:
        # Check if memory metrics history is empty; warn and return empty string if so
        if not metrics_history:
            logging.warning("No memory usage data available for visualization")
            return ""
        
        # Create two vertical subplots for memory usage and available memory over time
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Extract timestamps, memory usage percentage, and available memory from metrics
        times = [m.timestamp for m in metrics_history]
        memory_pct = [m.memory_percent for m in metrics_history]
        memory_available = [m.memory_available for m in metrics_history]
        
        # Plot system memory usage percentage over time
        axes[0].plot(times, memory_pct, color='#FF6B6B', linewidth=2)
        axes[0].set_title('System Memory Usage (%)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Memory Usage (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)  # Limit y-axis from 0 to 100%
        
        # Plot available memory in megabytes over time
        axes[1].plot(times, memory_available, color='#4ECDC4', linewidth=2)
        axes[1].set_title('Available Memory (MB)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Available Memory (MB)')
        axes[1].grid(True, alpha=0.3)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the plot and return the saved file path
        return self._save_plot(fig, "memory_usage.png")
    
    def create_interactive_dashboard(self, system_metrics: List[SystemMetrics], 
                                   training_monitor: TrainingMonitor) -> str:
        # If interactive visualization is disabled, log info and return empty string
        if not self.interactive:
            logging.info("Interactive visualization disabled")
            return ""
        
        # Create a 3-row, 2-column subplot dashboard using plotly with specified titles
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Training Loss', 'Learning Rate', 'GPU Usage', 'Batch Times'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # If system metrics are available, add traces for CPU, memory, and GPU usage
        if system_metrics:
            times = [m.timestamp for m in system_metrics]
            cpu_data = [m.cpu_percent for m in system_metrics]
            memory_data = [m.memory_percent for m in system_metrics]
            gpu_data = [np.mean(m.gpu_utilization) if m.gpu_utilization else 0 for m in system_metrics]
            
            fig.add_trace(go.Scatter(x=times, y=cpu_data, name="CPU %", line=dict(color='#FF6B6B')), row=1, col=1)
            fig.add_trace(go.Scatter(x=times, y=memory_data, name="Memory %", line=dict(color='#4ECDC4')), row=1, col=2)
            fig.add_trace(go.Scatter(x=times, y=gpu_data, name="GPU %", line=dict(color='#45B7D1')), row=3, col=1)
        
        # If training monitor has loss data, add training loss and learning rate traces
        if training_monitor.losses: #type: ignore 
            steps = training_monitor.steps #type: ignore 
            losses = training_monitor.losses #type: ignore 
            learning_rates = training_monitor.learning_rates #type: ignore 
            
            fig.add_trace(go.Scatter(x=steps, y=losses, name="Training Loss", line=dict(color='#FFA07A')), row=2, col=1)
            fig.add_trace(go.Scatter(x=steps, y=learning_rates, name="Learning Rate", line=dict(color='#96CEB4')), row=2, col=2)
        
        # If batch times are available, add batch processing time trace
        if training_monitor.batch_times: #type: ignore 
            batch_indices = list(range(len(training_monitor.batch_times))) #type: ignore 
            fig.add_trace(go.Scatter(x=batch_indices, y=training_monitor.batch_times, name="Batch Time", line=dict(color='#DDA0DD')), row=3, col=2) #type: ignore 
        
        # Update overall layout properties of the dashboard figure
        fig.update_layout(
            height=900,
            title_text="Real-time Monitoring Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        # Define the output HTML file path for the dashboard
        dashboard_path = os.path.join(self.output_dir, "monitoring_dashboard.html")
        # Write the dashboard figure as an interactive HTML file
        fig.write_html(dashboard_path)
        
        # Log event of dashboard creation with file path
        self.logger.log_training_event("dashboard_created", {"path": dashboard_path})
        # Return the path to the saved interactive dashboard
        return dashboard_path
    
    def generate_comprehensive_report(self, system_metrics: List[SystemMetrics],
                                    training_monitor: TrainingMonitor,
                                    data_metrics: List[DataMetrics]) -> str:
        
        # Initialize list to collect paths of generated plot files
        plots_generated = []
        
        try:
            # Generate and collect individual metric plots and dashboard
            plots_generated.append(self.plot_system_metrics(system_metrics))
            plots_generated.append(self.plot_gpu_metrics(system_metrics))
            plots_generated.append(self.plot_training_progress(training_monitor))
            plots_generated.append(self.plot_data_distribution(data_metrics))
            plots_generated.append(self.plot_memory_usage(system_metrics))
            plots_generated.append(self.create_interactive_dashboard(system_metrics, training_monitor))
        except Exception as e:
            # Log any exception raised during report generation along with context
            self.logger.log_error_event(e, {"context": "comprehensive_report_generation"})
            logging.error(f"Error generating comprehensive report: {e}")
        
        # Filter out any empty results from failed plot generations
        plots_generated = [p for p in plots_generated if p]
        
        # Prepare a summary dictionary of the report generation details
        report_summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "plots_generated": len(plots_generated),
            "output_directory": self.output_dir,
            "plot_files": plots_generated
        }
        
        # Log event indicating comprehensive report generation completion
        self.logger.log_training_event("comprehensive_report_generated", report_summary)
        # Return path to the interactive dashboard as main report entry point
        return os.path.join(self.output_dir, "monitoring_dashboard.html")
    
    def cleanup_old_plots(self, days_to_keep: int = 7):
        # Calculate cutoff date; files older than this will be deleted
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Iterate over all files in the output directory
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            # Only consider files, not directories
            if os.path.isfile(file_path):
                # Get last modification datetime of the file
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                # If the file is older than cutoff, delete it
                if file_modified < cutoff_date:
                    os.remove(file_path)
                    # Log event of old plot removal with filename
                    self.logger.log_training_event("old_plot_removed", {"file": filename})