#FILE: monitoring/profiler.py
# ehsanasgharzde - PROFILER

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
        self.profile_memory = profile_memory
        self.profile_cuda = profile_cuda
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.profiler_results = {}
        self.current_section = None
        self.step_counter = 0
        
        from .logger import MonitoringLogger
        self.logger = MonitoringLogger(log_dir=str(self.output_dir / "logs"))

    def _get_torch_activities(self) -> List[ProfilerActivity]:
        activities = [ProfilerActivity.CPU]
        if self.profile_cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        return activities

    def _log_profiler_event(self, event_name: str, data: Dict[str, Any]) -> None:
        event_data = {
            "event": event_name,
            "step": self.step_counter,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        self.logger.log_training_event(event_name, event_data)

    def _save_profile_results(self, name: str, results: Dict[str, Any]) -> None:
        save_path = self.output_dir / f"{name}_step_{self.step_counter}.json"
        with open(save_path, 'w') as f:
            torch.save(results, f) #type: ignore 

    @contextmanager
    def profile_context(self, name: str):
        self.current_section = name
        pr = cProfile.Profile()
        pr.enable()
        
        torch_profiler = profile(
            activities=self._get_torch_activities(),
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True
        )
        
        start_time = time.time()
        torch_profiler.__enter__()
        
        try:
            with record_function(name):
                yield
        except Exception as e:
            self.logger.log_error_event(e, {"section": name, "step": self.step_counter})
            raise
        finally:
            end_time = time.time()
            torch_profiler.__exit__(None, None, None)
            pr.disable()
            
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
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
            
            self.profiler_results[name] = results
            self._log_profiler_event("profile_context", results)
            
            if self.step_counter % self.log_interval == 0:
                self._save_profile_results(name, results)

    def profile_function(self, func: Callable, name: Optional[str] = None, *args, **kwargs) -> Dict[str, Any]:
        func_name = name or func.__name__
        pr = cProfile.Profile()
        pr.enable()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            self.logger.log_error_event(e, {"function": func_name, "step": self.step_counter})
        finally:
            end_time = time.time()
            pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        profile_result = {
            "function": func_name,
            "step": self.step_counter,
            "result": result,
            "success": success,
            "error": error,
            "runtime": end_time - start_time,
            "profile_output": s.getvalue()
        }
        
        self._log_profiler_event("function_profile", profile_result)
        return profile_result

    def profile_model_inference(self, model: torch.nn.Module, input_data: torch.Tensor, 
                               warmup_steps: int = 5, benchmark_steps: int = 10) -> Dict[str, Any]:
        model.eval()
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        for _ in range(warmup_steps):
            with torch.no_grad():
                _ = model(input_data)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        with profile(
            activities=self._get_torch_activities(),
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                times = []
                for _ in range(benchmark_steps):
                    start = time.time()
                    output = model(input_data)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    times.append(time.time() - start)
        
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
        
        self._log_profiler_event("model_inference", inference_stats)
        return inference_stats

    def profile_data_loading(self, dataloader, max_batches: int = 10) -> Dict[str, Any]:
        batch_times = []
        total_samples = 0
        
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            batch_start = time.time()
            if isinstance(batch, (tuple, list)):
                batch_size = len(batch[0]) if hasattr(batch[0], '__len__') else 1
            else:
                batch_size = len(batch) if hasattr(batch, '__len__') else 1
            
            total_samples += batch_size
            batch_times.append(time.time() - batch_start)
        
        total_time = time.time() - start_time
        
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
        
        self._log_profiler_event("data_loading", loading_stats)
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