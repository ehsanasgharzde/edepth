#FILE: monitoring/profiler.py
# ehsanasgharzde - PROFILER
# hosseinsolymanzadeh - PROPER COMMENTING

import cProfile
import pstats
import io
import time
from contextlib import contextmanager
from typing import Callable, Dict, Any, List, Optional
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
import numpy as np
from pathlib import Path

class PerformanceProfiler:
    def __init__(self, profile_memory: bool = True, profile_cuda: bool = True, 
                 log_interval: int = 10, output_dir: str = "profiler_output"):
        # Initialize profiler settings and output directory
        self.profile_memory = profile_memory
        self.profile_cuda = profile_cuda
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
        
        # Initialize storage for profiling results and state tracking variables
        self.profiler_results = {}
        self.current_section = None
        self.step_counter = 0
        
        # Import and initialize a logger for monitoring training events
        from .logger import MonitoringLogger
        self.logger = MonitoringLogger(log_dir=str(self.output_dir / "logs"))

    def _get_torch_activities(self) -> List[ProfilerActivity]:
        # Determine which PyTorch profiler activities to include (CPU always, CUDA if enabled and available)
        activities = [ProfilerActivity.CPU]
        if self.profile_cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        return activities

    def _log_profiler_event(self, event_name: str, data: Dict[str, Any]) -> None:
        # Prepare and send a profiling event log entry including step and timestamp
        event_data = {
            "event": event_name,
            "step": self.step_counter,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        self.logger.log_training_event(event_name, event_data)

    def _save_profile_results(self, name: str, results: Dict[str, Any]) -> None:
        # Save profiling results to a JSON file named with section and step number
        save_path = self.output_dir / f"{name}_step_{self.step_counter}.json"
        with open(save_path, 'w') as f:
            torch.save(results, f) #type: ignore 

    @contextmanager
    def profile_context(self, name: str):
        # Context manager for profiling a code section named `name`
        self.current_section = name
        
        # Set up the standard Python profiler
        pr = cProfile.Profile()
        pr.enable()
        
        # Set up PyTorch profiler with selected activities and memory tracking
        torch_profiler = profile(
            activities=self._get_torch_activities(),
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True
        )
        
        # Start timing the profiling context
        start_time = time.time()
        torch_profiler.__enter__()
        
        try:
            # Run the user code block with a named CUDA/NCCL record function for profiling
            with record_function(name):
                yield
        except Exception as e:
            # Log any exceptions raised during profiling with section and step info, then re-raise
            self.logger.log_error_event(e, {"section": name, "step": self.step_counter})
            raise
        finally:
            # On exiting the context, stop timers and profilers
            end_time = time.time()
            torch_profiler.__exit__(None, None, None)
            pr.disable()
            
            # Collect Python cProfile stats into a string buffer
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            # Aggregate profiling results including timing, CPU profile, and PyTorch profile summary
            results = {
                "section": name,
                "step": self.step_counter,
                "duration": end_time - start_time,
                "cpu_profile": s.getvalue(),
                "torch_profile": torch_profiler.key_averages().table(
                    sort_by="cpu_time_total" if not self.profile_cuda else "cuda_time_total", 
                    row_limit=10
                )
            }
            
            # Store results in internal dictionary and log the profiling event
            self.profiler_results[name] = results
            self._log_profiler_event("profile_context", results)
            
            # Save profiling results periodically based on configured log interval
            if self.step_counter % self.log_interval == 0:
                self._save_profile_results(name, results)

    def profile_function(self, func: Callable, name: Optional[str] = None, *args, **kwargs) -> Dict[str, Any]:
        # Determine the function name to use in logs (either provided or func.__name__)
        func_name = name or func.__name__
        # Create a cProfile profiler instance
        pr = cProfile.Profile()
        # Start profiling
        pr.enable()
        
        # Record start time for runtime measurement
        start_time = time.time()
        try:
            # Call the target function with given arguments
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            # In case of exception, capture error and log it
            result = None
            success = False
            error = str(e)
            self.logger.log_error_event(e, {"function": func_name, "step": self.step_counter})
        finally:
            # Record end time and stop profiling regardless of success or failure
            end_time = time.time()
            pr.disable()
        
        # Capture profiling stats output into a string buffer
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Prepare a dictionary of profiling results and metadata
        profile_result = {
            "function": func_name,
            "step": self.step_counter,
            "result": result,
            "success": success,
            "error": error,
            "runtime": end_time - start_time,
            "profile_output": s.getvalue()
        }
        
        # Log the profiling event with collected data
        self._log_profiler_event("function_profile", profile_result)
        # Return the profiling information
        return profile_result
    
    def profile_model_inference(self, model: torch.nn.Module, input_data: torch.Tensor, 
                               warmup_steps: int = 5, benchmark_steps: int = 10) -> Dict[str, Any]:
        # Set model to evaluation mode to disable training-specific layers like dropout
        model.eval()
        # Get the device of model parameters (CPU or GPU)
        device = next(model.parameters()).device
        # Move input data to the same device as the model
        input_data = input_data.to(device)
        
        # Perform warm-up iterations to stabilize performance (e.g., GPU caching)
        for _ in range(warmup_steps):
            with torch.no_grad():
                _ = model(input_data)
        
        # Synchronize CUDA to ensure warm-up is complete if using GPU
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Profile the model inference execution with PyTorch profiler
        with profile(
            activities=self._get_torch_activities(),
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                times = []
                # Run benchmark iterations, timing each inference call
                for _ in range(benchmark_steps):
                    start = time.time()
                    output = model(input_data)
                    # Synchronize CUDA after each inference if GPU is available
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    times.append(time.time() - start)
        
        # Aggregate statistics about the inference performance
        inference_stats = {
            "model_name": model.__class__.__name__,
            "input_shape": tuple(input_data.shape),
            "device": str(device),
            "warmup_steps": warmup_steps,
            "benchmark_steps": benchmark_steps,
            "avg_inference_time": np.mean(times),
            "std_inference_time": np.std(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "throughput_samples_per_sec": input_data.size(0) / np.mean(times),
            "torch_profile": prof.key_averages().table(
                sort_by="cuda_time_total" if self.profile_cuda else "cpu_time_total", 
                row_limit=10
            )
        }
        
        # Log the profiling data for model inference
        self._log_profiler_event("model_inference", inference_stats)
        # Return the collected inference statistics
        return inference_stats
    
    def profile_data_loading(self, dataloader, max_batches: int = 10) -> Dict[str, Any]:
        # List to keep track of time taken to load each batch
        batch_times = []
        # Counter for total number of samples loaded
        total_samples = 0
        
        # Record start time of data loading process
        start_time = time.time()
        
        # Iterate over batches in the dataloader up to max_batches
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            # Start timing this batch
            batch_start = time.time()
            
            # Determine batch size: handle cases where batch is tuple/list or single tensor
            if isinstance(batch, (tuple, list)):
                batch_size = len(batch[0]) if hasattr(batch[0], '__len__') else 1
            else:
                batch_size = len(batch) if hasattr(batch, '__len__') else 1
            
            # Accumulate total samples processed
            total_samples += batch_size
            # Record time taken to load this batch
            batch_times.append(time.time() - batch_start)
        
        # Compute total elapsed time for loading all batches
        total_time = time.time() - start_time
        
        # Compile statistics on data loading performance
        loading_stats = {
            "total_batches": len(batch_times),
            "total_samples": total_samples,
            "total_time": total_time,
            "avg_batch_time": np.mean(batch_times),
            "std_batch_time": np.std(batch_times),
            "min_batch_time": np.min(batch_times),
            "max_batch_time": np.max(batch_times),
            "throughput_samples_per_sec": total_samples / total_time,
            "throughput_batches_per_sec": len(batch_times) / total_time
        }
        
        # Log the data loading profiling event
        self._log_profiler_event("data_loading", loading_stats)
        # Return the data loading performance statistics
        return loading_stats

    def profile_training_step(self, model: torch.nn.Module, loss_fn: Callable, 
                             optimizer: torch.optim.Optimizer, input_data: torch.Tensor, 
                             targets: torch.Tensor) -> Dict[str, Any]:
        model.train()
        
        timings = {}
        
        start_time = time.time()
        
        with profile(
            activities=self._get_torch_activities(),
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True
        ) as prof:
            forward_start = time.time()
            output = model(input_data)
            timings["forward_pass"] = time.time() - forward_start
            
            loss_start = time.time()
            loss = loss_fn(output, targets)
            timings["loss_computation"] = time.time() - loss_start
            
            backward_start = time.time()
            loss.backward()
            timings["backward_pass"] = time.time() - backward_start
            
            optimizer_start = time.time()
            optimizer.step()
            optimizer.zero_grad()
            timings["optimizer_step"] = time.time() - optimizer_start
        
        total_time = time.time() - start_time
        
        training_stats = {
            "step": self.step_counter,
            "total_time": total_time,
            "timings": timings,
            "loss_value": loss.item(),
            "torch_profile": prof.key_averages().table(
                sort_by="cuda_time_total" if self.profile_cuda else "cpu_time_total", 
                row_limit=10
            )
        }
        
        self.step_counter += 1
        self._log_profiler_event("training_step", training_stats)
        
        return training_stats

    def profile_memory_usage(self) -> Dict[str, Any]:
        memory_stats = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                memory_stats[f"cuda_{i}"] = {
                    "allocated_mb": torch.cuda.memory_allocated(i) / (1024 ** 2),
                    "reserved_mb": torch.cuda.memory_reserved(i) / (1024 ** 2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated(i) / (1024 ** 2),
                    "max_reserved_mb": torch.cuda.max_memory_reserved(i) / (1024 ** 2)
                }
        
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_stats["system"] = {
            "rss_mb": memory_info.rss / (1024 ** 2),
            "vms_mb": memory_info.vms / (1024 ** 2),
            "percent": process.memory_percent()
        }
        
        self._log_profiler_event("memory_usage", memory_stats)
        return memory_stats

    def generate_performance_report(self, save_path: Optional[str] = None) -> str:
        report_lines = [
            "=" * 50,
            "PERFORMANCE PROFILING REPORT",
            "=" * 50,
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Total Steps: {self.step_counter}",
            ""
        ]
        
        for section_name, results in self.profiler_results.items():
            report_lines.extend([
                f"--- {section_name.upper()} ---",
                f"Duration: {results.get('duration', 'N/A')}s",
                ""
            ])
            
            if 'torch_profile' in results:
                report_lines.extend([
                    "PyTorch Profile:",
                    results['torch_profile'],
                    ""
                ])
        
        memory_summary = self.profile_memory_usage()
        report_lines.extend([
            "--- MEMORY SUMMARY ---",
            str(memory_summary),
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
        
        return report_content

    def reset_profiler(self) -> None:
        self.profiler_results.clear()
        self.step_counter = 0
        self.current_section = None
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)