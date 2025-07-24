# FILE: monitoring/streamlit_dashboard.py
# ehsanasgharzde - SYSTEM MONITOR

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import logging

# Import our monitoring components
from configs.config import MonitoringConfig
from model_integration import ModelMonitoringIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Models Monitoring Dashboard",
    page_icon="e",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'monitoring_integration' not in st.session_state:
    st.session_state.monitoring_integration = None
if 'config' not in st.session_state:
    st.session_state.config = MonitoringConfig()
if 'monitoring_started' not in st.session_state:
    st.session_state.monitoring_started = False

def initialize_monitoring():
    if st.session_state.monitoring_integration is None:
        st.session_state.monitoring_integration = ModelMonitoringIntegration(
            config=st.session_state.config
        )

def create_gauge_chart(value: float, title: str, max_value: float = 100, 
                      threshold: float = 80, unit: str = "%") -> go.Figure:
    color = "green" if value < threshold * 0.7 else "orange" if value < threshold else "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title} ({unit})"},
        delta={'reference': threshold},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 0.7], 'color': "lightgray"},
                {'range': [threshold * 0.7, threshold], 'color': "yellow"},
                {'range': [threshold, max_value], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_time_series_chart(data: List[Dict], x_field: str, y_field: str, 
                           title: str, color: str = "blue") -> go.Figure:
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=300)
        return fig

    df = pd.DataFrame(data)
    fig = px.line(df, x=x_field, y=y_field, title=title, color_discrete_sequence=[color])
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_overview_page():
    st.title("edepth Monitoring Dashboard")

    if not st.session_state.monitoring_started:
        st.warning("Monitoring is not started. Please start monitoring from the sidebar.")
        return

    integration = st.session_state.monitoring_integration
    if integration is None:
        st.error("Monitoring integration not initialized")
        return

    # Get monitoring summary
    summary = integration.get_monitoring_summary()

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Monitoring Status",
            value="Active" if summary.get('monitoring_active', False) else "Inactive",
            delta="Running" if summary.get('monitoring_active', False) else "Stopped"
        )

    with col2:
        model_count = len(summary.get('model_summaries', {}))
        st.metric(
            label="Registered Models",
            value=model_count,
            delta=f"{model_count} models"
        )

    with col3:
        training_summary = summary.get('training_summary', {})
        current_epoch = training_summary.get('current_epoch', 0)
        st.metric(
            label="Current Epoch",
            value=current_epoch,
            delta=f"Step {training_summary.get('current_step', 0)}"
        )

    with col4:
        current_loss = training_summary.get('current_loss', 0.0)
        st.metric(
            label="Current Loss",
            value=f"{current_loss:.4f}" if current_loss else "N/A",
            delta=f"Min: {training_summary.get('min_loss', 0.0):.4f}" if training_summary.get('min_loss') else None
        )

    st.divider()

    # System metrics gauges
    st.subheader("System Resources")
    system_metrics = summary.get('current_system_metrics')

    if system_metrics:
        col1, col2, col3 = st.columns(3)

        with col1:
            cpu_chart = create_gauge_chart(
                system_metrics.cpu_percent, 
                "CPU Usage", 
                threshold=st.session_state.config.cpu_alert_threshold
            )
            st.plotly_chart(cpu_chart, use_container_width=True)

        with col2:
            memory_chart = create_gauge_chart(
                system_metrics.memory_percent,
                "Memory Usage",
                threshold=st.session_state.config.memory_alert_threshold
            )
            st.plotly_chart(memory_chart, use_container_width=True)

        with col3:
            if system_metrics.gpu_utilization:
                avg_gpu = sum(system_metrics.gpu_utilization) / len(system_metrics.gpu_utilization)
                gpu_chart = create_gauge_chart(
                    avg_gpu,
                    "Average GPU Usage",
                    threshold=st.session_state.config.gpu_utilization_threshold
                )
                st.plotly_chart(gpu_chart, use_container_width=True)
            else:
                st.info("No GPU metrics available")
    else:
        st.info("No system metrics available")

    # Model performance summary
    st.subheader("Model Performance")
    model_summaries = summary.get('model_summaries', {})

    if model_summaries:
        for model_name, model_summary in model_summaries.items():
            if model_summary:  # Check if summary is not empty
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label=f"{model_name} - Avg Inference Time",
                        value=f"{model_summary.get('avg_inference_time', 0):.3f}s",
                        delta=f"Min: {model_summary.get('min_inference_time', 0):.3f}s"
                    )

                with col2:
                    st.metric(
                        label=f"{model_name} - Throughput",
                        value=f"{model_summary.get('avg_throughput', 0):.1f} samples/s",
                        delta=f"{model_summary.get('total_measurements', 0)} measurements"
                    )

                with col3:
                    st.metric(
                        label=f"{model_name} - Memory Usage",
                        value=f"{model_summary.get('avg_memory_usage', 0):.1f} MB",
                        delta="Average"
                    )

                with col4:
                    st.metric(
                        label=f"{model_name} - Parameters",
                        value=f"{model_summary.get('parameters_count', 0):,}",
                        delta="Total params"
                    )
    else:
        st.info("No model performance data available")

def render_model_metrics_page():
    st.title("Model Performance Metrics")

    if not st.session_state.monitoring_started:
        st.warning("Monitoring is not started.")
        return

    integration = st.session_state.monitoring_integration
    if integration is None:
        return

    # Get all metrics
    all_metrics = integration.model_monitor.get_all_metrics()
    model_metrics = all_metrics.get('model_metrics', {})

    if not model_metrics:
        st.info("No model metrics available yet. Run some inference to collect data.")
        return

    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=list(model_metrics.keys()),
        index=0
    )

    if selected_model and model_metrics[selected_model]:
        metrics_data = model_metrics[selected_model]

        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            options=["Last 10 minutes", "Last 30 minutes", "Last hour", "All time"],
            index=0
        )

        # Filter data based on time range
        now = datetime.now(timezone.utc)
        if time_range == "Last 10 minutes":
            cutoff = now - timedelta(minutes=10)
        elif time_range == "Last 30 minutes":
            cutoff = now - timedelta(minutes=30)
        elif time_range == "Last hour":
            cutoff = now - timedelta(hours=1)
        else:
            cutoff = datetime.min

        filtered_data = [
            m for m in metrics_data 
            if datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00')) >= cutoff
        ]

        if not filtered_data:
            st.warning("No data available for the selected time range.")
            return

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Inference time chart
            inference_chart = create_time_series_chart(
                filtered_data, 'timestamp', 'inference_time',
                f"{selected_model} - Inference Time", "blue"
            )
            st.plotly_chart(inference_chart, use_container_width=True)

            # Memory usage chart
            memory_chart = create_time_series_chart(
                filtered_data, 'timestamp', 'memory_usage',
                f"{selected_model} - Memory Usage (MB)", "red"
            )
            st.plotly_chart(memory_chart, use_container_width=True)

        with col2:
            # Throughput chart
            throughput_chart = create_time_series_chart(
                filtered_data, 'timestamp', 'throughput',
                f"{selected_model} - Throughput (samples/s)", "green"
            )
            st.plotly_chart(throughput_chart, use_container_width=True)

            # Batch size chart
            batch_chart = create_time_series_chart(
                filtered_data, 'timestamp', 'batch_size',
                f"{selected_model} - Batch Size", "orange"
            )
            st.plotly_chart(batch_chart, use_container_width=True)

        # Statistics table
        st.subheader("Statistics")
        if filtered_data:
            df = pd.DataFrame(filtered_data)

            stats = {
                'Metric': ['Inference Time (s)', 'Memory Usage (MB)', 'Throughput (samples/s)', 'Batch Size'],
                'Mean': [
                    df['inference_time'].mean(),
                    df['memory_usage'].mean(),
                    df['throughput'].mean(),
                    df['batch_size'].mean()
                ],
                'Std': [
                    df['inference_time'].std(),
                    df['memory_usage'].std(),
                    df['throughput'].std(),
                    df['batch_size'].std()
                ],
                'Min': [
                    df['inference_time'].min(),
                    df['memory_usage'].min(),
                    df['throughput'].min(),
                    df['batch_size'].min()
                ],
                'Max': [
                    df['inference_time'].max(),
                    df['memory_usage'].max(),    
                    df['throughput'].max(),
                    df['batch_size'].max()
                ]
            }

            st.dataframe(pd.DataFrame(stats), use_container_width=True)

def render_training_metrics_page():
    st.title("Training Metrics")

    if not st.session_state.monitoring_started:
        st.warning("Monitoring is not started.")
        return

    integration = st.session_state.monitoring_integration
    if integration is None:
        return

    # Get training metrics
    all_metrics = integration.model_monitor.get_all_metrics()
    training_metrics = all_metrics.get('training_metrics', [])

    if not training_metrics:
        st.info("No training metrics available yet. Start training to collect data.")
        return

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(training_metrics)

    # Time range selection
    time_range = st.selectbox(
        "Time Range",
        options=["Last 100 steps", "Last 500 steps", "Last 1000 steps", "All steps"],
        index=0
    )

    # Filter data
    if time_range == "Last 100 steps":
        df_filtered = df.tail(100)
    elif time_range == "Last 500 steps":
        df_filtered = df.tail(500)
    elif time_range == "Last 1000 steps":
        df_filtered = df.tail(1000)
    else:
        df_filtered = df

    # Training progress charts
    col1, col2 = st.columns(2)

    with col1:
        # Loss chart
        loss_chart = create_time_series_chart(
            df_filtered.to_dict('records'), 'step', 'loss',
            "Training Loss", "red"
        )
        st.plotly_chart(loss_chart, use_container_width=True)

        # Gradient norm chart (if available)
        if 'gradient_norm' in df_filtered.columns and df_filtered['gradient_norm'].notna().any(): # type:ignore
            grad_chart = create_time_series_chart(
                df_filtered.dropna(subset=['gradient_norm']).to_dict('records'),
                'step', 'gradient_norm',
                "Gradient Norm", "purple"
            )
            st.plotly_chart(grad_chart, use_container_width=True)

    with col2:
        # Learning rate chart
        lr_chart = create_time_series_chart(
            df_filtered.to_dict('records'), 'step', 'learning_rate',
            "Learning Rate", "green"
        )
        st.plotly_chart(lr_chart, use_container_width=True)

        # Validation loss chart (if available)
        if 'validation_loss' in df_filtered.columns and df_filtered['validation_loss'].notna().any(): # type:ignore
            val_loss_chart = create_time_series_chart(
                df_filtered.dropna(subset=['validation_loss']).to_dict('records'),
                'step', 'validation_loss',
                "Validation Loss", "orange"
            )
            st.plotly_chart(val_loss_chart, use_container_width=True)

    # Training summary
    st.subheader("Training Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Steps",
            value=len(df),
            delta=f"Current: {df.iloc[-1]['step']}" if len(df) > 0 else None
        )

    with col2:
        current_loss = df.iloc[-1]['loss'] if len(df) > 0 else 0
        min_loss = df['loss'].min() if len(df) > 0 else 0
        st.metric(
            label="Current Loss",
            value=f"{current_loss:.6f}",
            delta=f"Min: {min_loss:.6f}"
        )

    with col3:
        current_lr = df.iloc[-1]['learning_rate'] if len(df) > 0 else 0
        st.metric(
            label="Learning Rate", 
            value=f"{current_lr:.2e}",
            delta="Current"
        )

    with col4:
        if 'gradient_norm' in df.columns and df['gradient_norm'].notna().any(): # type:ignore
            current_grad = df.dropna(subset=['gradient_norm']).iloc[-1]['gradient_norm']
            avg_grad = df['gradient_norm'].mean()
            st.metric(
                label="Gradient Norm",
                value=f"{current_grad:.4f}",
                delta=f"Avg: {avg_grad:.4f}"
            )

def render_system_metrics_page():
    st.title("System Resources")

    if not st.session_state.monitoring_started:
        st.warning("Monitoring is not started.")
        return

    integration = st.session_state.monitoring_integration
    if integration is None:
        return

    # Get system metrics
    system_summary = integration.system_monitor.get_summary_stats(minutes=10)
    current_metrics = integration.system_monitor.get_current_metrics()

    if not current_metrics:
        st.info("No system metrics available yet.")
        return

    # Current system status
    st.subheader("Current Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        cpu_chart = create_gauge_chart(
            current_metrics.cpu_percent,
            "CPU Usage",
            threshold=st.session_state.config.cpu_alert_threshold
        )
        st.plotly_chart(cpu_chart, use_container_width=True)

    with col2:
        memory_chart = create_gauge_chart(
            current_metrics.memory_percent,
            "Memory Usage", 
            threshold=st.session_state.config.memory_alert_threshold
        )
        st.plotly_chart(memory_chart, use_container_width=True)

    with col3:
        disk_chart = create_gauge_chart(
            current_metrics.disk_usage,
            "Disk Usage",
            threshold=st.session_state.config.disk_usage_threshold
        )
        st.plotly_chart(disk_chart, use_container_width=True)

    # GPU metrics (if available)
    if current_metrics.gpu_utilization:
        st.subheader("GPU Metrics")

        gpu_cols = st.columns(len(current_metrics.gpu_utilization))

        for i, (gpu_util, gpu_mem_used, gpu_mem_total) in enumerate(
            zip(current_metrics.gpu_utilization, 
                current_metrics.gpu_memory_used,
                current_metrics.gpu_memory_total)
        ):
            with gpu_cols[i]:
                # GPU utilization
                gpu_chart = create_gauge_chart(
                    gpu_util,
                    f"GPU {i} Usage",
                    threshold=st.session_state.config.gpu_utilization_threshold
                )
                st.plotly_chart(gpu_chart, use_container_width=True)

                # GPU memory
                if gpu_mem_total > 0:
                    gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
                    st.metric(
                        label=f"GPU {i} Memory",
                        value=f"{gpu_mem_percent:.1f}%",
                        delta=f"{gpu_mem_used:.0f}MB / {gpu_mem_total:.0f}MB"
                    )

    # Historical data
    st.subheader("Historical Data (Last 10 minutes)")

    recent_metrics = integration.system_monitor.get_metrics_history(minutes=10)

    if recent_metrics:
        # Convert to format suitable for plotting
        timestamps = [m.timestamp.isoformat() for m in recent_metrics]
        cpu_data = [{'timestamp': ts, 'value': m.cpu_percent} for ts, m in zip(timestamps, recent_metrics)]
        memory_data = [{'timestamp': ts, 'value': m.memory_percent} for ts, m in zip(timestamps, recent_metrics)]

        col1, col2 = st.columns(2)

        with col1:
            cpu_history_chart = create_time_series_chart(
                cpu_data, 'timestamp', 'value',
                "CPU Usage History (%)", "blue"
            )
            st.plotly_chart(cpu_history_chart, use_container_width=True)

        with col2:
            memory_history_chart = create_time_series_chart(
                memory_data, 'timestamp', 'value', 
                "Memory Usage History (%)", "red"
            )
            st.plotly_chart(memory_history_chart, use_container_width=True)

        # GPU history (if available)
        if recent_metrics[0].gpu_utilization:
            st.subheader("GPU Usage History")

            for gpu_id in range(len(recent_metrics[0].gpu_utilization)):
                gpu_data = [
                    {'timestamp': ts, 'value': m.gpu_utilization[gpu_id] if len(m.gpu_utilization) > gpu_id else 0}
                    for ts, m in zip(timestamps, recent_metrics)
                ]

                gpu_history_chart = create_time_series_chart(
                    gpu_data, 'timestamp', 'value',
                    f"GPU {gpu_id} Usage History (%)", "green"
                )
                st.plotly_chart(gpu_history_chart, use_container_width=True)

def render_configuration_page():
    st.title("Configuration")

    # Load current configuration
    config = st.session_state.config

    st.subheader("Monitoring Settings")

    col1, col2 = st.columns(2)

    with col1:
        enable_model_monitoring = st.checkbox(
            "Enable Model Monitoring",
            value=config.enable_model_monitoring
        )

        enable_system_monitoring = st.checkbox(
            "Enable System Monitoring", 
            value=config.enable_system_monitoring
        )

        gradient_monitoring = st.checkbox(
            "Enable Gradient Monitoring",
            value=config.gradient_monitoring
        )

        model_history_size = st.number_input(
            "Model History Size",
            min_value=100,
            max_value=10000,
            value=config.model_history_size,
            step=100
        )

    with col2:
        system_update_interval = st.number_input(
            "System Update Interval (seconds)",
            min_value=0.1, 
            max_value=10.0,
            value=config.system_update_interval,
            step=0.1
        )

        dashboard_update_interval = st.number_input(
            "Dashboard Update Interval (seconds)",
            min_value=1,
            max_value=60,
            value=config.dashboard_update_interval,
            step=1
        )

        export_format = st.selectbox(
            "Export Format",
            options=["json", "csv"],
            index=0 if config.export_format == "json" else 1
        )

    st.subheader("Alert Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        cpu_threshold = st.slider(
            "CPU Alert Threshold (%)",
            min_value=50.0,
            max_value=95.0,
            value=config.cpu_alert_threshold,
            step=5.0
        )

        memory_threshold = st.slider(
            "Memory Alert Threshold (%)",
            min_value=50.0,
            max_value=95.0, 
            value=config.memory_alert_threshold,
            step=5.0
        )

    with col2:
        gpu_util_threshold = st.slider(
            "GPU Utilization Threshold (%)",
            min_value=50.0,
            max_value=95.0,
            value=config.gpu_utilization_threshold,
            step=5.0
        )

        gpu_temp_threshold = st.slider(
            "GPU Temperature Threshold (°C)",
            min_value=60.0,
            max_value=90.0,
            value=config.gpu_temperature_threshold,
            step=5.0
        )

    # Save configuration
    if st.button("Save Configuration"):
        # Update configuration
        st.session_state.config.enable_model_monitoring = enable_model_monitoring
        st.session_state.config.enable_system_monitoring = enable_system_monitoring
        st.session_state.config.gradient_monitoring = gradient_monitoring
        st.session_state.config.model_history_size = model_history_size
        st.session_state.config.system_update_interval = system_update_interval
        st.session_state.config.dashboard_update_interval = dashboard_update_interval
        st.session_state.config.export_format = export_format
        st.session_state.config.cpu_alert_threshold = cpu_threshold
        st.session_state.config.memory_alert_threshold = memory_threshold
        st.session_state.config.gpu_utilization_threshold = gpu_util_threshold
        st.session_state.config.gpu_temperature_threshold = gpu_temp_threshold

        # Reinitialize monitoring with new config
        if st.session_state.monitoring_integration:
            st.session_state.monitoring_integration.stop_monitoring()
            st.session_state.monitoring_integration = ModelMonitoringIntegration(
                config=st.session_state.config
            )
            if st.session_state.monitoring_started:
                st.session_state.monitoring_integration.start_monitoring()

        st.success("Configuration saved successfully!")

    # Export/Import configuration
    st.subheader("Configuration Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Configuration"):
            config_dict = st.session_state.config.to_dict()
            st.download_button(
                label="Download config.json",
                data=json.dumps(config_dict, indent=2),
                file_name="monitoring_config.json",
                mime="application/json"
            )

    with col2:
        uploaded_config = st.file_uploader(
            "Import Configuration",
            type=['json', 'yaml', 'yml']
        )

        if uploaded_config is not None:
            try:
                if uploaded_config.name.endswith('.json'):
                    config_data = json.load(uploaded_config)
                else:
                    import yaml
                    config_data = yaml.safe_load(uploaded_config)

                st.session_state.config = MonitoringConfig.from_dict(config_data)
                st.success("Configuration imported successfully!")
                st.experimental_rerun() # type:ignore
            except Exception as e:
                st.error(f"Failed to import configuration: {e}")

def render_export_page():
    st.title("Export Data")

    if not st.session_state.monitoring_started:
        st.warning("Monitoring is not started.")
        return

    integration = st.session_state.monitoring_integration
    if integration is None:
        return

    st.subheader("Available Data")

    # Show data summary
    summary = integration.get_monitoring_summary()

    col1, col2, col3 = st.columns(3)

    with col1:
        model_summaries = summary.get('model_summaries', {})
        total_model_measurements = sum(
            s.get('total_measurements', 0) for s in model_summaries.values() if s
        )
        st.metric("Model Measurements", total_model_measurements)

    with col2:
        training_summary = summary.get('training_summary', {})
        total_training_steps = training_summary.get('total_steps', 0)
        st.metric("Training Steps", total_training_steps)

    with col3:
        system_summary = summary.get('system_summary', {})
        system_duration = system_summary.get('timestamp_range', {}).get('duration_minutes', 0)
        st.metric("System Data (minutes)", system_duration)

    st.subheader("Export Options")

    col1, col2 = st.columns(2)

    with col1:
        export_format = st.selectbox(
            "Export Format",
            options=["json", "csv"],
            index=0
        )

        export_what = st.multiselect(
            "What to Export",
            options=["Model Metrics", "Training Metrics", "System Metrics"],
            default=["Model Metrics", "Training Metrics", "System Metrics"]
        )

    with col2:
        filename_base = st.text_input(
            "Filename (without extension)",
            value=f"monitoring_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    if st.button("Export Data"):
        try:
            if "Model Metrics" in export_what or "Training Metrics" in export_what:
                model_filepath = f"{filename_base}_model.{export_format}"
                integration.model_monitor.export_metrics(model_filepath, export_format)

                with open(model_filepath, 'rb') as f:
                    st.download_button(
                        label=f"Download {model_filepath}",
                        data=f.read(),
                        file_name=model_filepath,
                        mime="application/json" if export_format == "json" else "text/csv"
                    )

            if "System Metrics" in export_what:
                system_filepath = f"{filename_base}_system.{export_format}"
                integration.system_monitor.export_metrics(system_filepath, export_format)

                with open(system_filepath, 'rb') as f:
                    st.download_button(
                        label=f"Download {system_filepath}",
                        data=f.read(),
                        file_name=system_filepath,
                        mime="application/json" if export_format == "json" else "text/csv"
                    )

            st.success("Data exported successfully!")

        except Exception as e:
            st.error(f"Export failed: {e}")

def main():
    # Initialize monitoring
    initialize_monitoring()

    # Sidebar
    with st.sidebar:
        st.title("Control Panel")

        # Monitoring control
        st.subheader("Monitoring Control")

        if not st.session_state.monitoring_started:
            if st.button("Start Monitoring", type="primary"):
                if st.session_state.monitoring_integration:
                    st.session_state.monitoring_integration.start_monitoring()
                    st.session_state.monitoring_started = True
                    st.success("Monitoring started!")
                    st.experimental_rerun() # type:ignore
        else:
            if st.button("Stop Monitoring", type="secondary"):
                if st.session_state.monitoring_integration:
                    st.session_state.monitoring_integration.stop_monitoring()
                    st.session_state.monitoring_started = False
                    st.success("Monitoring stopped!")
                    st.experimental_rerun() # type:ignore

        st.divider()

        # Navigation
        st.subheader("Navigation")
        page = st.selectbox(
            "Select Page",
            options=[
                "Overview",
                "Model Metrics", 
                "Training Metrics",
                "System Resources",
                "Configuration",
                "Export Data"
            ],
            index=0
        )

        st.divider()

        # Status info
        st.subheader("Status")
        st.write(f"**Monitoring:** {'Active' if st.session_state.monitoring_started else 'Inactive'}")

        if st.session_state.monitoring_integration:
            summary = st.session_state.monitoring_integration.get_monitoring_summary()
            model_count = len(summary.get('model_summaries', {}))
            st.write(f"**Models:** {model_count}")

            training_summary = summary.get('training_summary', {})
            if training_summary:
                st.write(f"**Current Epoch:** {training_summary.get('current_epoch', 0)}")
                st.write(f"**Training Steps:** {training_summary.get('total_steps', 0)}")

        # Auto-refresh
        if st.session_state.monitoring_started:
            st.divider()
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            if auto_refresh:
                time.sleep(st.session_state.config.dashboard_update_interval)
                st.experimental_rerun() # type:ignore

    # Main content area
    if page == "Overview":
        render_overview_page()
    elif page == "Model Metrics":
        render_model_metrics_page()
    elif page == "Training Metrics":
        render_training_metrics_page()
    elif page == "System Resources":
        render_system_metrics_page()
    elif page == "Configuration":
        render_configuration_page()
    elif page == "Export Data":
        render_export_page()
