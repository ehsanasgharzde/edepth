# FILE: benchmarks/benchmark.py

from dataclasses import dataclass
from typing import List, Dict
from tabulate import tabulate
import json


@dataclass
class ModelArchitectureBenchmark:
    model: str
    fps: int
    latency_ms: float
    mae: float
    rmse: float
    params: str
    size_mb: int

    def __post_init__(self):
        if isinstance(self.params, str):
            self.params_numeric = float(self.params.replace('M', '')) * 1e6
        else:
            self.params_numeric = self.params


@dataclass
class ExportFormatBenchmark:
    format: str
    latency_ms: float
    size_mb: int
    accuracy_drop: str
    load_time: str

    def __post_init__(self):
        if self.accuracy_drop == "0":
            self.accuracy_drop_numeric = 0.0
        else:
            # Convert "-0.3%" to -0.3
            self.accuracy_drop_numeric = float(self.accuracy_drop.replace('%', ''))


@dataclass
class PrecisionBenchmark:
    precision: str
    inference_time_ms: int
    accuracy_rmse: float
    size_mb: int


@dataclass
class PlatformBenchmark:
    platform: str
    model: str
    latency_ms: int
    fps: float


class ModelArchitectureBenchmarks:
    
    def __init__(self):
        self.benchmarks = [
        ]
    
    def get_all_benchmarks(self) -> List[ModelArchitectureBenchmark]:
        return self.benchmarks
    
    def get_fastest_model(self) -> ModelArchitectureBenchmark:
        return max(self.benchmarks, key=lambda x: x.fps)
    
    def get_most_accurate_model(self) -> ModelArchitectureBenchmark:
        return min(self.benchmarks, key=lambda x: x.rmse)
    
    def get_smallest_model(self) -> ModelArchitectureBenchmark:
        return min(self.benchmarks, key=lambda x: x.size_mb)
    
    def get_best_for_edge_deployment(self) -> ModelArchitectureBenchmark:
        # Score based on normalized metrics (lower is better)
        def edge_score(model):
            max_latency = max(b.latency_ms for b in self.benchmarks)
            max_rmse = max(b.rmse for b in self.benchmarks)
            max_size = max(b.size_mb for b in self.benchmarks)
            
            return (model.latency_ms / max_latency + 
                   model.rmse / max_rmse + 
                   model.size_mb / max_size) / 3
        
        return min(self.benchmarks, key=edge_score)
    
    def display_table(self) -> str:
        headers = ["Model", "FPS", "Latency (ms)", "MAE", "RMSE", "Params", "Size (MB)"]
        rows = [[b.model, b.fps, b.latency_ms, b.mae, b.rmse, b.params, b.size_mb] 
                for b in self.benchmarks]
        return tabulate(rows, headers=headers, tablefmt="grid")


class ExportFormatBenchmarks:
    
    def __init__(self):
        self.benchmarks = [
        ]
    
    def get_all_benchmarks(self) -> List[ExportFormatBenchmark]:
        return self.benchmarks
    
    def get_fastest_format(self) -> ExportFormatBenchmark:
        return min(self.benchmarks, key=lambda x: x.latency_ms)
    
    def get_most_accurate_format(self) -> ExportFormatBenchmark:
        return min(self.benchmarks, key=lambda x: abs(x.accuracy_drop_numeric))
    
    def get_smallest_format(self) -> ExportFormatBenchmark:
        return min(self.benchmarks, key=lambda x: x.size_mb)
    
    def display_table(self) -> str:
        headers = ["Format", "Latency (ms)", "Size (MB)", "Accuracy Drop", "Load Time"]
        rows = [[b.format, b.latency_ms, b.size_mb, b.accuracy_drop, b.load_time] 
                for b in self.benchmarks]
        return tabulate(rows, headers=headers, tablefmt="grid")


class PrecisionBenchmarks:
    
    def __init__(self):
        self.benchmarks = [
        ]
    
    def get_all_benchmarks(self) -> List[PrecisionBenchmark]:
        return self.benchmarks
    
    def get_fastest_precision(self) -> PrecisionBenchmark:
        return min(self.benchmarks, key=lambda x: x.inference_time_ms)
    
    def get_most_accurate_precision(self) -> PrecisionBenchmark:
        return min(self.benchmarks, key=lambda x: x.accuracy_rmse)
    
    def get_smallest_precision(self) -> PrecisionBenchmark:
        return min(self.benchmarks, key=lambda x: x.size_mb)
    
    def get_best_speed_accuracy_tradeoff(self) -> PrecisionBenchmark:
        # Score based on normalized speed and accuracy (lower is better)
        def tradeoff_score(precision):
            max_time = max(b.inference_time_ms for b in self.benchmarks)
            max_rmse = max(b.accuracy_rmse for b in self.benchmarks)
            
            return (precision.inference_time_ms / max_time + 
                   precision.accuracy_rmse / max_rmse) / 2
        
        return min(self.benchmarks, key=tradeoff_score)
    
    def display_table(self) -> str:
        headers = ["Precision", "Inference Time (ms)", "Accuracy (RMSE)", "Size (MB)"]
        rows = [[b.precision, b.inference_time_ms, b.accuracy_rmse, b.size_mb] 
                for b in self.benchmarks]
        return tabulate(rows, headers=headers, tablefmt="grid")


class PlatformBenchmarks:
    
    def __init__(self):
        self.benchmarks = [
        ]
    
    def get_all_benchmarks(self) -> List[PlatformBenchmark]:
        return self.benchmarks
    
    def get_fastest_platform(self) -> PlatformBenchmark:
        return max(self.benchmarks, key=lambda x: x.fps)
    
    def get_lowest_latency_platform(self) -> PlatformBenchmark:
        return min(self.benchmarks, key=lambda x: x.latency_ms)
    
    def get_edge_platforms(self) -> List[PlatformBenchmark]:
        edge_keywords = ["jetson", "nano", "edge", "mobile"]
        return [b for b in self.benchmarks 
                if any(keyword in b.platform.lower() for keyword in edge_keywords)]
    
    def display_table(self) -> str:
        headers = ["Platform", "Model", "Latency (ms)", "FPS"]
        rows = [[b.platform, b.model, f"{b.latency_ms}ms", b.fps] 
                for b in self.benchmarks]
        return tabulate(rows, headers=headers, tablefmt="grid")


class BenchmarkManager:
    
    def __init__(self):
        self.model_arch = ModelArchitectureBenchmarks()
        self.export_format = ExportFormatBenchmarks()
        self.precision = PrecisionBenchmarks()
        self.platform = PlatformBenchmarks()
    
    def display_all_benchmarks(self):
        print("=" * 80)
        print("DEPTH ESTIMATION MODEL BENCHMARKS")
        print("=" * 80)
        
        print("\n1. MODEL ARCHITECTURE BENCHMARKS")
        print("-" * 50)
        print(self.model_arch.display_table())
        
        print("\n2. EXPORT FORMAT BENCHMARKS")
        print("-" * 50)
        print(self.export_format.display_table())
        
        print("\n3. PRECISION BENCHMARKS")
        print("-" * 50)
        print(self.precision.display_table())
        
        print("\n4. PLATFORM BENCHMARKS")
        print("-" * 50)
        print(self.platform.display_table())
    
    def get_recommendations(self) -> Dict[str, str]:
        recommendations = {}
        
        # Best overall model
        best_model = self.model_arch.get_best_for_edge_deployment()
        recommendations["best_edge_model"] = f"{best_model.model} (balanced speed/accuracy/size)"
        
        # Best export format
        best_format = self.export_format.get_fastest_format()
        recommendations["best_export_format"] = f"{best_format.format} (lowest latency: {best_format.latency_ms}ms)"
        
        # Best precision
        best_precision = self.precision.get_best_speed_accuracy_tradeoff()
        recommendations["best_precision"] = f"{best_precision.precision} (best speed-accuracy balance)"
        
        # Best platform
        best_platform = self.platform.get_fastest_platform()
        recommendations["best_platform"] = f"{best_platform.platform} (highest FPS: {best_platform.fps})"
        
        return recommendations
    
    def display_recommendations(self):
        print("\nSMART RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = self.get_recommendations()
        for category, recommendation in recommendations.items():
            print(f"• {category.replace('_', ' ').title()}: {recommendation}")
    
    def export_to_json(self, filename: str = "benchmarks.json"):
        data = {
            "model_architecture": [
                {
                    "model": b.model,
                    "fps": b.fps,
                    "latency_ms": b.latency_ms,
                    "mae": b.mae,
                    "rmse": b.rmse,
                    "params": b.params,
                    "size_mb": b.size_mb
                } for b in self.model_arch.benchmarks
            ],
            "export_format": [
                {
                    "format": b.format,
                    "latency_ms": b.latency_ms,
                    "size_mb": b.size_mb,
                    "accuracy_drop": b.accuracy_drop,
                    "load_time": b.load_time
                } for b in self.export_format.benchmarks
            ],
            "precision": [
                {
                    "precision": b.precision,
                    "inference_time_ms": b.inference_time_ms,
                    "accuracy_rmse": b.accuracy_rmse,
                    "size_mb": b.size_mb
                } for b in self.precision.benchmarks
            ],
            "platform": [
                {
                    "platform": b.platform,
                    "model": b.model,
                    "latency_ms": b.latency_ms,
                    "fps": b.fps
                } for b in self.platform.benchmarks
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nBenchmarks exported to {filename}")

def query_best_model_for_use_case(use_case: str) -> str:
    manager = BenchmarkManager()
    
    use_case = use_case.lower()
    
    if "edge" in use_case or "mobile" in use_case or "embedded" in use_case:
        model = manager.model_arch.get_best_for_edge_deployment()
        precision = manager.precision.get_smallest_precision()
        format_rec = manager.export_format.get_fastest_format()
        
        return f"""
EDGE DEPLOYMENT RECOMMENDATION:
• Model: {model.model} (FPS: {model.fps}, Size: {model.size_mb}MB)
• Precision: {precision.precision} (Size: {precision.size_mb}MB, Speed: {precision.inference_time_ms}ms)
• Format: {format_rec.format} (Latency: {format_rec.latency_ms}ms)
        """.strip()
    
    elif "accuracy" in use_case or "precise" in use_case:
        model = manager.model_arch.get_most_accurate_model()
        precision = manager.precision.get_most_accurate_precision()
        format_rec = manager.export_format.get_most_accurate_format()
        
        return f"""
HIGH ACCURACY RECOMMENDATION:
• Model: {model.model} (RMSE: {model.rmse}, MAE: {model.mae})
• Precision: {precision.precision} (RMSE: {precision.accuracy_rmse})
• Format: {format_rec.format} (Accuracy Drop: {format_rec.accuracy_drop})
        """.strip()
    
    elif "speed" in use_case or "fast" in use_case or "realtime" in use_case:
        model = manager.model_arch.get_fastest_model()
        precision = manager.precision.get_fastest_precision()
        format_rec = manager.export_format.get_fastest_format()
        
        return f"""
HIGH SPEED RECOMMENDATION:
• Model: {model.model} (FPS: {model.fps}, Latency: {model.latency_ms}ms)
• Precision: {precision.precision} (Inference: {precision.inference_time_ms}ms)
• Format: {format_rec.format} (Latency: {format_rec.latency_ms}ms)
        """.strip()
    
    else:
        return "Please specify use case: 'edge', 'accuracy', or 'speed'"

def main():
    # Initialize benchmark manager
    manager = BenchmarkManager()
    
    # Display all benchmarks
    manager.display_all_benchmarks()
    
    # Show recommendations
    manager.display_recommendations()
    
    # Export to JSON
    manager.export_to_json()
    
    # Demo use case queries
    print("\n" + "=" * 80)
    print("USE CASE QUERY EXAMPLES")
    print("=" * 80)
    
    use_cases = ["edge deployment", "high accuracy", "real-time speed"]
    for use_case in use_cases:
        print(f"\nQuery: '{use_case}'")
        print(query_best_model_for_use_case(use_case))