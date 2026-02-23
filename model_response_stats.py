"""
Response Metadata and Statistics Collection Utilities

This module contains utilities for collecting and populating response metadata
and usage statistics for the VibeVoice AudioClone server.
"""

import uuid
import psutil
import torch
import os
import time
import threading
import resource
from audiocloneserver import clone_interface_pb2
from audiomessages import audio_message_pb2


def get_thread_cpu_time():
    """Get CPU time for current thread."""
    try:
        # Get resource usage for current thread (Linux/Unix)
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        return usage.ru_utime + usage.ru_stime  # User + system time in seconds
    except (AttributeError, OSError):
        # Fallback to process-level if thread-level not available
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_utime + usage.ru_stime
        except (AttributeError, OSError):
            return 0.0


def get_thread_wall_time():
    """Get wall-clock time for current thread."""
    return time.perf_counter()


class ThreadCpuTracker:
    """Track CPU time for a specific thread."""
    
    def __init__(self):
        self.start_cpu_time = get_thread_cpu_time()
        self.start_wall_time = get_thread_wall_time()
        self.thread_id = threading.get_ident()
    
    def get_elapsed_cpu_time(self):
        """Get elapsed CPU time since tracking started."""
        current_cpu_time = get_thread_cpu_time()
        return current_cpu_time - self.start_cpu_time
    
    def get_elapsed_wall_time(self):
        """Get elapsed wall time since tracking started."""
        current_wall_time = get_thread_wall_time()
        return current_wall_time - self.start_wall_time
    
    def get_cpu_efficiency(self):
        """Get CPU efficiency ratio (CPU time / wall time)."""
        cpu_time = self.get_elapsed_cpu_time()
        wall_time = self.get_elapsed_wall_time()
        return cpu_time / wall_time if wall_time > 0 else 0.0


def track_thread_resources(operation_name="unknown"):
    """Start tracking resources for a specific operation."""
    tracker = ThreadCpuTracker()
    return {
        "operation": operation_name,
        "tracker": tracker,
        "start_time": time.time(),
        "thread_id": tracker.thread_id
    }


def get_thread_resource_stats(tracking_info):
    """Get resource statistics for a tracked operation."""
    if not tracking_info:
        return {}
    
    tracker = tracking_info["tracker"]
    elapsed_time = (time.time() - tracking_info["start_time"]) * 1000  # ms
    
    return {
        "operation": tracking_info["operation"],
        "thread_id": tracking_info["thread_id"],
        "elapsed_ms": elapsed_time,
        "cpu_time_seconds": tracker.get_elapsed_cpu_time(),
        "wall_time_seconds": tracker.get_elapsed_wall_time(),
        "cpu_efficiency": tracker.get_cpu_efficiency(),
        "cpu_time_ms": tracker.get_elapsed_cpu_time() * 1000
    }


def get_system_resources():
    """Get current system resource utilization using non-blocking calls."""
    
    resources = {}
    
    # CPU usage (non-blocking, uses last measurement)
    cpu_percent = psutil.cpu_percent(interval=None)
    resources["cpu_percent"] = cpu_percent if cpu_percent > 0 else 0.0
    resources["cpu_count"] = psutil.cpu_count(logical=True)
    
    # Get detailed CPU times for more accurate averaging
    cpu_times = psutil.cpu_times(percpu=False)
    resources["cpu_times"] = {
        "user": cpu_times.user,
        "system": cpu_times.system,
        "idle": cpu_times.idle
    }
    
    # Memory usage (detailed)
    memory = psutil.virtual_memory()
    resources["memory_used_gb"] = memory.used / (1024**3)
    resources["memory_total_gb"] = memory.total / (1024**3)
    resources["memory_percent"] = memory.percent
    resources["memory_available_gb"] = memory.available / (1024**3)
    resources["memory_cached_gb"] = memory.cached / (1024**3) if hasattr(memory, 'cached') else 0
    
    # Process-specific metrics (current process)
    try:
        current_process = psutil.Process()
        proc_memory = current_process.memory_info()
        resources["process_memory_rss_gb"] = proc_memory.rss / (1024**3)  # Physical memory
        resources["process_memory_vms_gb"] = proc_memory.vms / (1024**3)  # Virtual memory
        resources["process_cpu_percent"] = current_process.cpu_percent(interval=None)
        resources["process_num_threads"] = current_process.num_threads()
        resources["process_create_time"] = current_process.create_time()
        
        # Thread-specific info
        threads = current_process.threads()
        resources["process_threads"] = [
            {
                "id": thread.id,
                "user_time": thread.user_time,
                "system_time": thread.system_time
            } for thread in threads
        ] if threads else []
        
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Fallback if process info not available
        resources["process_memory_rss_gb"] = 0
        resources["process_memory_vms_gb"] = 0
        resources["process_cpu_percent"] = 0
        resources["process_num_threads"] = 0
        resources["process_create_time"] = 0
        resources["process_threads"] = []
    
    # GPU usage (if available)
    if torch.cuda.is_available():
        resources["gpu_available"] = True
        resources["gpu_count"] = torch.cuda.device_count()
        
        # Get GPU memory for each device
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
            memory_total = props.total_memory / (1024**3)              # GB
            
            gpu_memory.append({
                "device": i,
                "name": torch.cuda.get_device_name(i),
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "total_gb": memory_total,
                "utilization_percent": (memory_allocated / memory_total) * 100,
                "free_gb": memory_total - memory_allocated
            })
        
        resources["gpu_memory"] = gpu_memory
    else:
        resources["gpu_available"] = False
        # For Apple Silicon (MPS) or other devices
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            resources["device_type"] = "mps"
            resources["device_name"] = "Apple Silicon GPU"
        else:
            resources["device_type"] = "cpu"
            resources["device_name"] = "CPU"
    
    return resources


def get_resource_delta(start_resources, end_resources):
    """Calculate resource consumption between two measurements."""    
    delta = {}
    
    # CPU delta (use both system and process CPU for better accuracy)
    system_cpu_avg = (start_resources["cpu_percent"] + end_resources["cpu_percent"]) / 2
    process_cpu_avg = (start_resources.get("process_cpu_percent", 0) + end_resources.get("process_cpu_percent", 0)) / 2
    
    delta["cpu_avg_percent"] = system_cpu_avg
    delta["process_cpu_avg_percent"] = process_cpu_avg
    delta["cpu_peak_percent"] = max(start_resources["cpu_percent"], end_resources["cpu_percent"])
    delta["process_cpu_peak_percent"] = max(start_resources.get("process_cpu_percent", 0), end_resources.get("process_cpu_percent", 0))
    
    # Memory delta (system and process)
    delta["memory_peak_gb"] = max(start_resources["memory_used_gb"], end_resources["memory_used_gb"])
    delta["memory_delta_gb"] = end_resources["memory_used_gb"] - start_resources["memory_used_gb"]
    delta["memory_peak_percent"] = max(start_resources["memory_percent"], end_resources["memory_percent"])
    
    # Process-specific memory deltas
    delta["process_memory_peak_rss_gb"] = max(
        start_resources.get("process_memory_rss_gb", 0), 
        end_resources.get("process_memory_rss_gb", 0)
    )
    delta["process_memory_delta_rss_gb"] = end_resources.get("process_memory_rss_gb", 0) - start_resources.get("process_memory_rss_gb", 0)
    delta["process_memory_peak_vms_gb"] = max(
        start_resources.get("process_memory_vms_gb", 0), 
        end_resources.get("process_memory_vms_gb", 0)
    )
    
    # Thread count delta
    delta["process_threads_start"] = start_resources.get("process_num_threads", 0)
    delta["process_threads_end"] = end_resources.get("process_num_threads", 0)
    delta["process_threads_peak"] = max(
        start_resources.get("process_num_threads", 0), 
        end_resources.get("process_num_threads", 0)
    )
    
    # GPU delta
    if start_resources.get("gpu_available", False):
        delta["gpu_peak_memory_gb"] = []
        for start_gpu, end_gpu in zip(start_resources["gpu_memory"], end_resources["gpu_memory"]):
            peak_memory = max(start_gpu["allocated_gb"], end_gpu["allocated_gb"])
            delta["gpu_peak_memory_gb"].append({
                "device": start_gpu["device"],
                "name": start_gpu.get("name", f"GPU_{start_gpu['device']}"),
                "peak_gb": peak_memory,
                "peak_percent": (peak_memory / start_gpu["total_gb"]) * 100,
                "start_allocated_gb": start_gpu["allocated_gb"],
                "end_allocated_gb": end_gpu["allocated_gb"],
                "delta_gb": end_gpu["allocated_gb"] - start_gpu["allocated_gb"]
            })
    
    return delta


def collect_response_metadata(response, text, sample_audio_bytes, cloned_audio_bytes, generation_stats, elapsed_ms, model_id):
    """
    Collect and populate response metadata and usage statistics.
    
    Args:
        response: CloneResponse object to populate
        text: Input text
        sample_audio_bytes: Original audio sample bytes
        cloned_audio_bytes: Generated audio bytes
        generation_stats: Statistics from audio generation
        elapsed_ms: Total processing time in milliseconds
        model_id: Model identifier string
    """
    # Initialize metadata with UUIDv4 for better randomization and time-based ordering
    request_id = str(uuid.uuid4())
    response.processing_meta.request_id = request_id
    response.processing_meta.status_code = 200
    response.processing_meta.time_taken_ms = elapsed_ms
    
    # Add usage statistics
    response.processing_meta.usage_stats["model"] = model_id
    response.processing_meta.usage_stats["input_text_length"] = str(len(text))
    response.processing_meta.usage_stats["audio_sample_size"] = str(len(sample_audio_bytes))
    response.processing_meta.usage_stats["generated_audio_size"] = str(len(cloned_audio_bytes))
    response.processing_meta.usage_stats["generation_time_ms"] = str(generation_stats.get("generation_time_ms", 0))
    response.processing_meta.usage_stats["device"] = generation_stats.get("device", "unknown")
    response.processing_meta.usage_stats["input_tokens"] = str(generation_stats.get("input_tokens", 0))
    response.processing_meta.usage_stats["generated_tokens"] = str(generation_stats.get("generated_tokens", 0))
    
    # Calculate token efficiency
    if generation_stats.get("generated_tokens", 0) > 0:
        token_efficiency = f"{generation_stats.get('generated_tokens', 0)}/{generation_stats.get('input_tokens', 0)}"
    else:
        token_efficiency = f"0/{generation_stats.get('input_tokens', 0)}"
    
    response.processing_meta.usage_stats["token_efficiency"] = token_efficiency
    response.processing_meta.usage_stats["max_new_tokens"] = str(generation_stats.get("max_new_tokens", 0))
    response.processing_meta.usage_stats["cfg_scale"] = str(generation_stats.get("cfg_scale", 0))
    response.processing_meta.usage_stats["output_audio_samples"] = str(generation_stats.get("output_audio_samples", 0))
    
    # Add resource utilization statistics
    response.processing_meta.usage_stats["cpu_avg_percent"] = f"{generation_stats.get('cpu_avg_percent', 0):.1f}"
    response.processing_meta.usage_stats["memory_peak_gb"] = f"{generation_stats.get('memory_peak_gb', 0):.2f}"
    response.processing_meta.usage_stats["memory_delta_gb"] = f"{generation_stats.get('memory_delta_gb', 0):.2f}"
    
    if generation_stats.get("gpu_available", False):
        response.processing_meta.usage_stats["gpu_available"] = "true"
        gpu_peak_memory = generation_stats.get("gpu_peak_memory_gb", [])
        if gpu_peak_memory:
            for i, gpu_mem in enumerate(gpu_peak_memory):
                response.processing_meta.usage_stats[f"gpu_{gpu_mem['device']}_peak_gb"] = f"{gpu_mem['peak_gb']:.2f}"
                response.processing_meta.usage_stats[f"gpu_{gpu_mem['device']}_peak_percent"] = f"{gpu_mem['peak_percent']:.1f}"
    else:
        response.processing_meta.usage_stats["gpu_available"] = "false"
        response.processing_meta.usage_stats["device_type"] = generation_stats.get("device_type", "unknown")
        response.processing_meta.usage_stats["device_name"] = generation_stats.get("device_name", "unknown")
    
    # Add model info to the cloned audio message
    model_info = audio_message_pb2.AudioMessageInfo()
    model_info.name = "model_name"
    model_info.value = model_id
    response.cloned_audio_message.audio_generator_model_name_version.CopyFrom(model_info)


def add_custom_model_info(response, model_name):
    """
    Add custom model info to the cloned audio message.
    
    Args:
        response: CloneResponse object to populate
        model_name: Custom model name to use
    """
    model_info = audio_message_pb2.AudioMessageInfo()
    model_info.name = "model_name"
    model_info.value = model_name
    response.cloned_audio_message.audio_generator_model_name_version.CopyFrom(model_info)
