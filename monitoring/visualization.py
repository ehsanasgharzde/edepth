#FILE: monitoring/visualization.py
# ehsanasgharzde - VISUALIZATION

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

sns.set_style("whitegrid")

class MonitoringVisualizer:
    def __init__(self, config: MonitoringConfig, logger: MonitoringLogger):
        self.config = config
        self.logger = logger
        self.output_dir = config.output_directory
        self.interactive = config.enable_visualization
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.plot_config = {
            "dpi": 150,
            "bbox_inches": "tight",
            "facecolor": "white",
            "edgecolor": "none"
        }
        
        self.color_palette = sns.color_palette("husl", 8)
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
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, **self.plot_config)
        self.logger.log_training_event("plot_saved", {"path": save_path})
        if close_after:
            plt.close(fig)
        return save_path

    def _create_time_series_plot(self, data: List[Dict[str, Any]], 
                                 y_keys: List[str], 
                                 title: str,
                                 ylabel: str = "Value",
                                 colors: Optional[List[str]] = None) -> plt.Figure: #type: ignore 
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if not data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        times = [entry.get('timestamp', datetime.utcnow()) for entry in data]
        
        for i, key in enumerate(y_keys):
            values = [entry.get(key, 0) for entry in data]
            color = colors[i] if colors and i < len(colors) else self.color_palette[i % len(self.color_palette)]
            ax.plot(times, values, label=key, color=color, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig

    def plot_system_metrics(self, metrics_history: List[SystemMetrics]) -> str:
        if not metrics_history:
            logging.warning("No system metrics data available for visualization")
            return ""
        
        data = [asdict(m) for m in metrics_history]
        fig = self._create_time_series_plot(
            data,
            ['cpu_percent', 'memory_percent', 'disk_usage'],
            'System Resource Utilization',
            'Percentage (%)',
            ['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        
        return self._save_plot(fig, "system_metrics.png")

    def plot_gpu_metrics(self, metrics_history: List[SystemMetrics]) -> str:
        if not metrics_history or not metrics_history[0].gpu_utilization:
            logging.warning("No GPU metrics data available for visualization")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        times = [m.timestamp for m in metrics_history]
        
        gpu_utils = [np.mean(m.gpu_utilization) if m.gpu_utilization else 0 for m in metrics_history]
        gpu_temps = [np.mean(m.gpu_temperature) if m.gpu_temperature else 0 for m in metrics_history]
        gpu_mem_used = [np.mean(m.gpu_memory_used) if m.gpu_memory_used else 0 for m in metrics_history]
        gpu_mem_total = [np.mean(m.gpu_memory_total) if m.gpu_memory_total else 0 for m in metrics_history]
        
        axes[0, 0].plot(times, gpu_utils, color='#FF6B6B', linewidth=2)
        axes[0, 0].set_title('GPU Utilization (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(times, gpu_temps, color='#FFA07A', linewidth=2)
        axes[0, 1].set_title('GPU Temperature (°C)')
        axes[0, 1].set_ylabel('Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(times, gpu_mem_used, color='#4ECDC4', linewidth=2, label='Used')
        axes[1, 0].plot(times, gpu_mem_total, color='#45B7D1', linewidth=2, label='Total')
        axes[1, 0].set_title('GPU Memory (MB)')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        if gpu_mem_total:
            mem_usage_pct = [(used/total)*100 if total > 0 else 0 for used, total in zip(gpu_mem_used, gpu_mem_total)]
            axes[1, 1].plot(times, mem_usage_pct, color='#96CEB4', linewidth=2)
            axes[1, 1].set_title('GPU Memory Usage (%)')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_plot(fig, "gpu_metrics.png")

    def plot_training_progress(self, training_monitor: TrainingMonitor) -> str:
        if not training_monitor.losses: #type: ignore 
            logging.warning("No training data available for visualization")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = training_monitor.steps #type: ignore 
        losses = training_monitor.losses #type: ignore 
        learning_rates = training_monitor.learning_rates #type: ignore 
        
        axes[0, 0].plot(steps, losses, color='#FF6B6B', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].plot(steps, learning_rates, color='#4ECDC4', linewidth=2)
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        if training_monitor.convergence_info: #type: ignore 
            conv_data = list(training_monitor.convergence_info) #type: ignore 
            axes[1, 0].plot(conv_data, color='#45B7D1', linewidth=2)
            axes[1, 0].set_title('Loss Convergence Rate')
            axes[1, 0].set_xlabel('Recent Steps')
            axes[1, 0].set_ylabel('Loss Delta')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        if training_monitor.batch_times: #type: ignore 
            batch_times = training_monitor.batch_times #type: ignore 
            axes[1, 1].plot(batch_times, color='#96CEB4', linewidth=2)
            axes[1, 1].set_title('Batch Processing Time')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_plot(fig, "training_progress.png")

    def plot_data_distribution(self, data_metrics: List[DataMetrics], title: str = "Data Distribution") -> str:
        if not data_metrics:
            logging.warning("No data metrics available for visualization")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        means = [m.mean_value for m in data_metrics]
        stds = [m.std_value for m in data_metrics]
        nan_counts = [m.nan_count for m in data_metrics]
        inf_counts = [m.inf_count for m in data_metrics]
        
        axes[0, 0].hist(means, bins=30, alpha=0.7, color='#FF6B6B', edgecolor='black')
        axes[0, 0].set_title('Mean Values Distribution')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(stds, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
        axes[0, 1].set_title('Standard Deviation Distribution')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(range(len(nan_counts)), nan_counts, color='#FFA07A', alpha=0.7)
        axes[1, 0].set_title('NaN Count per Tensor')
        axes[1, 0].set_xlabel('Tensor Index')
        axes[1, 0].set_ylabel('NaN Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(range(len(inf_counts)), inf_counts, color='#45B7D1', alpha=0.7)
        axes[1, 1].set_title('Inf Count per Tensor')
        axes[1, 1].set_xlabel('Tensor Index')
        axes[1, 1].set_ylabel('Inf Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return self._save_plot(fig, "data_distribution.png")

    def plot_gradient_flow(self, gradient_stats: Dict[str, List[float]]) -> str:
        if not gradient_stats:
            logging.warning("No gradient statistics available for visualization")
            return ""
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        layers = list(gradient_stats.keys())
        avg_grads = [np.mean(values) if values else 0 for values in gradient_stats.values()]
        max_grads = [np.max(values) if values else 0 for values in gradient_stats.values()]
        
        x_pos = np.arange(len(layers))
        
        axes[0].bar(x_pos, avg_grads, alpha=0.7, color='#FF6B6B', edgecolor='black')
        axes[0].set_title('Average Gradient Norms by Layer')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Average Gradient Norm')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(layers, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        axes[1].bar(x_pos, max_grads, alpha=0.7, color='#4ECDC4', edgecolor='black')
        axes[1].set_title('Maximum Gradient Norms by Layer')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Maximum Gradient Norm')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(layers, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        return self._save_plot(fig, "gradient_flow.png")

    def plot_memory_usage(self, metrics_history: List[SystemMetrics]) -> str:
        if not metrics_history:
            logging.warning("No memory usage data available for visualization")
            return ""
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        times = [m.timestamp for m in metrics_history]
        memory_pct = [m.memory_percent for m in metrics_history]
        memory_available = [m.memory_available for m in metrics_history]
        
        axes[0].plot(times, memory_pct, color='#FF6B6B', linewidth=2)
        axes[0].set_title('System Memory Usage (%)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Memory Usage (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        axes[1].plot(times, memory_available, color='#4ECDC4', linewidth=2)
        axes[1].set_title('Available Memory (MB)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Available Memory (MB)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_plot(fig, "memory_usage.png")

    def create_interactive_dashboard(self, system_metrics: List[SystemMetrics], 
                                   training_monitor: TrainingMonitor) -> str:
        if not self.interactive:
            logging.info("Interactive visualization disabled")
            return ""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Training Loss', 'Learning Rate', 'GPU Usage', 'Batch Times'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if system_metrics:
            times = [m.timestamp for m in system_metrics]
            cpu_data = [m.cpu_percent for m in system_metrics]
            memory_data = [m.memory_percent for m in system_metrics]
            gpu_data = [np.mean(m.gpu_utilization) if m.gpu_utilization else 0 for m in system_metrics]
            
            fig.add_trace(go.Scatter(x=times, y=cpu_data, name="CPU %", line=dict(color='#FF6B6B')), row=1, col=1)
            fig.add_trace(go.Scatter(x=times, y=memory_data, name="Memory %", line=dict(color='#4ECDC4')), row=1, col=2)
            fig.add_trace(go.Scatter(x=times, y=gpu_data, name="GPU %", line=dict(color='#45B7D1')), row=3, col=1)
        
        if training_monitor.losses: #type: ignore 
            steps = training_monitor.steps #type: ignore 
            losses = training_monitor.losses #type: ignore 
            learning_rates = training_monitor.learning_rates #type: ignore 
            
            fig.add_trace(go.Scatter(x=steps, y=losses, name="Training Loss", line=dict(color='#FFA07A')), row=2, col=1)
            fig.add_trace(go.Scatter(x=steps, y=learning_rates, name="Learning Rate", line=dict(color='#96CEB4')), row=2, col=2)
        
        if training_monitor.batch_times: #type: ignore 
            batch_indices = list(range(len(training_monitor.batch_times))) #type: ignore 
            fig.add_trace(go.Scatter(x=batch_indices, y=training_monitor.batch_times, name="Batch Time", line=dict(color='#DDA0DD')), row=3, col=2) #type: ignore 
        
        fig.update_layout(
            height=900,
            title_text="Real-time Monitoring Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        dashboard_path = os.path.join(self.output_dir, "monitoring_dashboard.html")
        fig.write_html(dashboard_path)
        
        self.logger.log_training_event("dashboard_created", {"path": dashboard_path})
        return dashboard_path

    def generate_comprehensive_report(self, system_metrics: List[SystemMetrics],
                                    training_monitor: TrainingMonitor,
                                    data_metrics: List[DataMetrics]) -> str:
        
        plots_generated = []
        
        try:
            plots_generated.append(self.plot_system_metrics(system_metrics))
            plots_generated.append(self.plot_gpu_metrics(system_metrics))
            plots_generated.append(self.plot_training_progress(training_monitor))
            plots_generated.append(self.plot_data_distribution(data_metrics))
            plots_generated.append(self.plot_memory_usage(system_metrics))
            plots_generated.append(self.create_interactive_dashboard(system_metrics, training_monitor))
        except Exception as e:
            self.logger.log_error_event(e, {"context": "comprehensive_report_generation"})
            logging.error(f"Error generating comprehensive report: {e}")
        
        plots_generated = [p for p in plots_generated if p]
        
        report_summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "plots_generated": len(plots_generated),
            "output_directory": self.output_dir,
            "plot_files": plots_generated
        }
        
        self.logger.log_training_event("comprehensive_report_generated", report_summary)
        return os.path.join(self.output_dir, "monitoring_dashboard.html")

    def cleanup_old_plots(self, days_to_keep: int = 7):
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            if os.path.isfile(file_path):
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_modified < cutoff_date:
                    os.remove(file_path)
                    self.logger.log_training_event("old_plot_removed", {"file": filename})