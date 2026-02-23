"""
VibeVoice AudioCloneServer Implementation

gRPC server that implements the AudioCloneModelWorker interface using VibeVoice model.
"""

import os
import io
import logging
import time
import torch
import soundfile as sf
import librosa
import numpy as np
import uuid
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from audiocloneserver.grpc_server_launcher import start_grpc_server
from audiocloneserver import clone_interface_pb2_grpc, clone_interface_pb2
from audiocloneserver.server import AudioCloneModelWorkerServicer
from audiomessages import audio_message_pb2
from model_response_stats import collect_response_metadata, add_custom_model_info, get_system_resources, get_resource_delta

# Load environment variables from .env file
def load_env_file(env_file_path=".env"):
    """Load environment variables from a .env file."""
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        logging.info(f"Loaded environment variables from {env_file_path}")
    else:
        logging.warning(f"Environment file {env_file_path} not found, using defaults")

load_env_file()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def calculate_generated_tokens(output, inputs):
    """
    Calculate the number of generated tokens from model output.
    
    Args:
        output: Model output containing generated sequences
        inputs: Input dictionary containing input_ids
        
    Returns:
        int: Number of generated tokens
    """
    generated_tokens = 0
    
    # Try different common output field names
    token_fields = [
        'generated_token_ids', 'output_sequences', 'sequences', 
        'generated_ids', 'output_ids', 'token_ids'
    ]
    
    for field in token_fields:
        if hasattr(output, field):
            tokens = getattr(output, field)
            if tokens is not None and len(tokens) > 0:
                # Calculate difference between output and input length
                input_length = len(inputs['input_ids'][0]) if 'input_ids' in inputs else 0
                output_length = len(tokens[0]) if hasattr(tokens[0], '__len__') else len(tokens)
                generated_tokens = max(0, output_length - input_length)
                break
        elif field in output:
            tokens = output[field]
            if tokens is not None and len(tokens) > 0:
                input_length = len(inputs['input_ids'][0]) if 'input_ids' in inputs else 0
                output_length = len(tokens[0]) if hasattr(tokens[0], '__len__') else len(tokens)
                generated_tokens = max(0, output_length - input_length)
                break
    
    return generated_tokens

# ============================================================================
# Model Initialization (Singleton)
# ============================================================================

MODEL_ID = os.getenv("VIBE_MODEL_ID", "microsoft/VibeVoice-1.5B")
MODEL_PATH = os.getenv("VIBE_MODEL_PATH", "./models/VibeVoice-1.5B")

# Device detection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

logger.info(f"Initializing VibeVoice model: {MODEL_ID}")
logger.info(f"Device: {DEVICE}, dtype: {TORCH_DTYPE}")

# Load processor and model once at module level
processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE
)
logger.info("VibeVoice model loaded successfully")


# ============================================================================
# Audio Processing Utilities
# ============================================================================

def resample_to_24khz(audio_tensor: torch.Tensor, original_sr: int) -> tuple[torch.Tensor, int]:
    """Resample audio tensor to 24kHz using librosa."""
    audio_tensor = audio_tensor.squeeze(0)
    audio_resampled = librosa.resample(
        audio_tensor.numpy(),
        orig_sr=original_sr,
        target_sr=24000
    )
    audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0)
    return audio_tensor, 24000


def process_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Process raw audio bytes into numpy array suitable for the model.
    Returns (audio_numpy, sample_rate).
    """
    audio_buffer = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_buffer)
    
    # Convert to float32 tensor
    audio_tensor = torch.from_numpy(data).float()
    
    # Handle stereo to mono
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.mean(dim=1)
    
    # Ensure shape is [1, samples]
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Resample to 24kHz if needed
    if samplerate != 24000:
        audio_tensor, samplerate = resample_to_24khz(audio_tensor, samplerate)
    
    # Convert to numpy for processor
    audio_numpy = audio_tensor.squeeze(0).numpy()
    return audio_numpy, samplerate


def load_audio_from_file(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio from file path into numpy array suitable for the model.
    Returns (audio_numpy, sample_rate).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    data, samplerate = sf.read(file_path)
    
    # Convert to float32 tensor
    audio_tensor = torch.from_numpy(data).float()
    
    # Handle stereo to mono
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.mean(dim=1)
    
    # Ensure shape is [1, samples]
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Resample to 24kHz if needed
    if samplerate != 24000:
        audio_tensor, samplerate = resample_to_24khz(audio_tensor, samplerate)
    
    # Convert to numpy for processor
    audio_numpy = audio_tensor.squeeze(0).numpy()
    return audio_numpy, samplerate


def generate_cloned_audio(text: str, voice_sample: np.ndarray, max_new_tokens: int = None, cfg_scale: float = None) -> tuple[bytes, dict]:
    """
    Generate cloned audio using VibeVoice model.
    
    Args:
        text: Text to synthesize
        voice_sample: Voice sample as numpy array (24kHz mono)
        max_new_tokens: Maximum number of new tokens to generate (default from env: 512)
        cfg_scale: Classifier-free guidance scale (default from env: 1.5)
    
    Returns:
        Tuple of (generated_audio_bytes, statistics_dict)
    """
    # Use environment variables if not provided
    if max_new_tokens is None:
        max_new_tokens = int(os.environ.get('MAX_NEW_TOKENS', '512'))
    if cfg_scale is None:
        cfg_scale = float(os.environ.get('CFG_SCALE', '1.5'))
    
    logger.debug(f"Starting audio generation for text: '{text[:50]}...'")
    logger.debug(f"Voice sample shape: {voice_sample.shape}, dtype: {voice_sample.dtype}")
    logger.debug(f"Using max_new_tokens: {max_new_tokens}, cfg_scale: {cfg_scale}")
    
    # Prepare inputs for the model
    logger.debug("Preparing inputs for processor...")
    inputs = processor(
        text=[text],
        voice_samples=[voice_sample],
        return_tensors="pt"
    ).to(DEVICE)
    
    logger.debug(f"Input keys: {list(inputs.keys())}")
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            logger.debug(f"  {key}: {type(value)}")
    
    # Generate audio
    logger.debug("Starting model generation...")
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale
        )
    
    generation_time = (time.time() - start_time) * 1000
    
    logger.debug(f"Model output type: {type(output)}")
    
    # Iterate over model output as dictionary and show key-value pairs
    logger.debug("Model output contents:")
    for key, value in output.items():
        if hasattr(value, 'shape'):
            logger.debug(f"  {key}: {type(value)}, shape: {value.shape}")
        elif hasattr(value, '__len__') and not callable(value):
            logger.debug(f"  {key}: {type(value)} (length: {len(value)})")
        elif isinstance(value, (int, float, str, bool)):
            logger.debug(f"  {key}: {value}")
        else:
            logger.debug(f"  {key}: {type(value)}")
    
    # Also show all attributes for completeness
    logger.debug(f"Model output attributes: {dir(output)}")
    
    # Extract audio from output
    audio_data = None
    if hasattr(output, 'speech_outputs') and len(output.speech_outputs) > 0:
        logger.debug(f"Found {len(output.speech_outputs)} speech outputs")
        for i, speech in enumerate(output.speech_outputs):
            if speech is not None:
                logger.debug(f"Using speech output {i}, shape: {speech.shape if hasattr(speech, 'shape') else 'N/A'}")
                audio_data = speech
                break
    else:
        logger.debug("No speech_outputs attribute found or empty")
    
    if audio_data is None:
        logger.error("No valid audio generated from model")
        raise RuntimeError("No valid audio generated from model")
    
    # Collect detailed statistics
    statistics = {
        "generation_time_ms": generation_time,
        "device": str(DEVICE),
        "model_name": MODEL_ID,
        "input_tokens": len(inputs['input_ids'][0]) if 'input_ids' in inputs else 0,
        "max_new_tokens": max_new_tokens,
        "cfg_scale": cfg_scale,
        "output_audio_samples": len(audio_data.flatten()) if hasattr(audio_data, 'flatten') else 0
    }
    
    # Extract generated tokens using the function
    statistics["generated_tokens"] = calculate_generated_tokens(output, inputs)
    
    logger.debug(f"Generation statistics: {statistics}")
    
    # Convert to numpy float32
    logger.debug("Converting audio to numpy...")
    audio_numpy = audio_data.to(torch.float32).cpu().numpy()
    logger.debug(f"Audio numpy shape: {audio_numpy.shape}, dtype: {audio_numpy.dtype}")
    
    # Reshape for soundfile
    original_shape = audio_numpy.shape
    if audio_numpy.ndim == 3:
        audio_numpy = audio_numpy[0, 0, :]
    elif audio_numpy.ndim == 2:
        audio_numpy = audio_numpy[0, :]
    
    logger.debug(f"Reshaped audio from {original_shape} to {audio_numpy.shape}")
    
    # Write to bytes buffer
    logger.debug("Writing audio to WAV buffer...")
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_numpy, 24000, format='WAV', subtype='PCM_16')
    audio_buffer.seek(0)
    
    wav_bytes = audio_buffer.read()
    logger.debug(f"Generated WAV audio: {len(wav_bytes)} bytes")
    
    return wav_bytes, statistics


# ============================================================================
# Clone Handlers
# ============================================================================


def clone_handler(request: clone_interface_pb2.CloneRequest, context) -> clone_interface_pb2.CloneResponse:
    """
    Handle a single clone request using VibeVoice model.
    
    Args:
        request: CloneRequest with request_audio_message (text) and sample_audio_message (voice sample)
        context: gRPC context
    
    Returns:
        CloneResponse with cloned audio
    """
    import time
    start_time = time.time()
    
    # Measure starting resources
    start_resources = get_system_resources()
    logger.debug(f"Starting resources: CPU {start_resources.get('cpu_percent', 0):.1f}%, Memory {start_resources.get('memory_used_gb', 0):.2f}GB")
    
    response = clone_interface_pb2.CloneResponse()
    
    try:
        # Extract text from request
        text = "Speaker 1: Cloned with Suvani Vibe voice " + request.request_audio_message.text
        if not text:
            raise ValueError("No text provided in request_audio_message")
        
        # Extract voice sample
        sample_audio_bytes = request.sample_audio_message.audio_binary
        sample_audio_path = request.sample_audio_message.audio_file_path
        
        if sample_audio_path:
            # Load audio from file path
            logger.info(f"Loading audio from file: {sample_audio_path}")
            voice_sample, _ = load_audio_from_file(sample_audio_path)
        elif sample_audio_bytes:
            # Process audio bytes
            logger.info(f"Processing audio bytes: {len(sample_audio_bytes)} bytes")
            voice_sample, _ = process_audio_bytes(sample_audio_bytes)
        else:
            raise ValueError("No audio_binary or audio_file_path provided in sample_audio_message")
        
        logger.info(f"Processing clone request: text='{text[:50]}...', audio_source={'file' if sample_audio_path else 'bytes'}")
        
        # Generate cloned audio
        cloned_audio_bytes, generation_stats = generate_cloned_audio(text, voice_sample)
        
        # Measure ending resources
        end_resources = get_system_resources()
        resource_delta = get_resource_delta(start_resources, end_resources)
        logger.debug(f"Ending resources: CPU {end_resources.get('cpu_percent', 0):.1f}%, Memory {end_resources.get('memory_used_gb', 0):.2f}GB")
        
        # Extract token generation statistics from generation_stats
        if generation_stats.get("generated_tokens", 0) > 0:
            token_efficiency = f"{generation_stats.get('generated_tokens', 0)}/{generation_stats.get('input_tokens', 0)}"
        else:
            token_efficiency = f"0/{generation_stats.get('input_tokens', 0)}"
        
        # Add resource utilization statistics to generation_stats
        generation_stats["cpu_avg_percent"] = resource_delta.get("cpu_avg_percent", 0)
        generation_stats["memory_peak_gb"] = resource_delta.get("memory_peak_gb", 0)
        generation_stats["memory_delta_gb"] = resource_delta.get("memory_delta_gb", 0)
        
        if start_resources.get("gpu_available", False):
            generation_stats["gpu_peak_memory_gb"] = resource_delta.get("gpu_peak_memory_gb", [])
            generation_stats["gpu_available"] = True
        else:
            generation_stats["gpu_available"] = False
            generation_stats["device_type"] = start_resources.get("device_type", "unknown")
            generation_stats["device_name"] = start_resources.get("device_name", "unknown")
        
        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Build response
        response.cloned_audio_message.audio_binary = cloned_audio_bytes
        
        # Collect metadata and usage statistics
        collect_response_metadata(response, text, sample_audio_bytes, cloned_audio_bytes, generation_stats, elapsed_ms, MODEL_ID)
        
        # Add model info to the cloned audio message
        add_custom_model_info(response, MODEL_ID)
        
        logger.info(f"Clone request completed in {elapsed_ms:.2f}ms")
        
    except Exception as e:
        logger.error(f"Clone request failed: {e}", exc_info=True)
        if not response.processing_meta:
            response.processing_meta = clone_interface_pb2.ProcessingMetadata()
        response.processing_meta.status_code = 500
        response.processing_meta.error_message = str(e)
    
    return response


def stream_clone_handler(request_iterator, context):
    """
    Handle streaming clone requests using VibeVoice model.
    
    Args:
        request_iterator: Iterator of CloneRequest messages
        context: gRPC context
    
    Yields:
        CloneResponse for each request
    """
    import time
    
    for request in request_iterator:
        start_time = time.time()
        
        # Measure starting resources
        start_resources = get_system_resources()
        logger.debug(f"Starting resources: CPU {start_resources.get('cpu_percent', 0):.1f}%, Memory {start_resources.get('memory_used_gb', 0):.2f}GB")
        
        response = clone_interface_pb2.CloneResponse()
        
        try:
            # Extract text from request
            text = request.request_audio_message.text
            if not text:
                raise ValueError("No text provided in request_audio_message")
            
            # Extract voice sample
            sample_audio_bytes = request.sample_audio_message.audio_binary
            sample_audio_path = request.sample_audio_message.audio_file_path
            
            if sample_audio_path:
                # Load audio from file path
                logger.info(f"Loading audio from file: {sample_audio_path}")
                voice_sample, _ = load_audio_from_file(sample_audio_path)
            elif sample_audio_bytes:
                # Process audio bytes
                logger.info(f"Processing audio bytes: {len(sample_audio_bytes)} bytes")
                voice_sample, _ = process_audio_bytes(sample_audio_bytes)
            else:
                raise ValueError("No audio_binary or audio_file_path provided in sample_audio_message")
            
            logger.info(f"Processing stream clone request: text='{text[:50]}...', audio_source={'file' if sample_audio_path else 'bytes'}")
            
            # Generate cloned audio
            cloned_audio_bytes, generation_stats = generate_cloned_audio(text, voice_sample)
            
            # Measure ending resources
            end_resources = get_system_resources()
            resource_delta = get_resource_delta(start_resources, end_resources)
            logger.debug(f"Ending resources: CPU {end_resources.get('cpu_percent', 0):.1f}%, Memory {end_resources.get('memory_used_gb', 0):.2f}GB")
            
            # Extract token generation statistics from generation_stats
            if generation_stats.get("generated_tokens", 0) > 0:
                token_efficiency = f"{generation_stats.get('generated_tokens', 0)}/{generation_stats.get('input_tokens', 0)}"
            else:
                token_efficiency = f"0/{generation_stats.get('input_tokens', 0)}"
            
            # Add resource utilization statistics to generation_stats
            generation_stats["cpu_avg_percent"] = resource_delta.get("cpu_avg_percent", 0)
            generation_stats["memory_peak_gb"] = resource_delta.get("memory_peak_gb", 0)
            generation_stats["memory_delta_gb"] = resource_delta.get("memory_delta_gb", 0)
            
            if start_resources.get("gpu_available", False):
                generation_stats["gpu_peak_memory_gb"] = resource_delta.get("gpu_peak_memory_gb", [])
                generation_stats["gpu_available"] = True
            else:
                generation_stats["gpu_available"] = False
                generation_stats["device_type"] = start_resources.get("device_type", "unknown")
                generation_stats["device_name"] = start_resources.get("device_name", "unknown")
            
            # Build response
            response.cloned_audio_message.audio_binary = cloned_audio_bytes
            
            # Calculate elapsed time
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Collect metadata and usage statistics
            collect_response_metadata(response, text, sample_audio_bytes, cloned_audio_bytes, generation_stats, elapsed_ms, MODEL_ID)
            
            # Add model info to the cloned audio message
            add_custom_model_info(response, MODEL_ID)
            
            logger.info(f"Stream clone request completed in {elapsed_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Stream clone request failed: {e}", exc_info=True)
            if not response.processing_meta:
                response.processing_meta = clone_interface_pb2.ProcessingMetadata()
            response.processing_meta.status_code = 500
            response.processing_meta.error_message = str(e)
        
        yield response


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the VibeVoice AudioClone gRPC server."""
    
    # Load configuration from environment
    port = int(os.getenv("GRPC_PORT", "50051"))
    workers = int(os.getenv("GRPC_WORKERS", "10"))
    processes = int(os.getenv("GRPC_PROCESSES", "1"))
    log_level = os.getenv("GRPC_LOG_LEVEL", "DEBUG")
    #log_file = os.getenv("GRPC_LOG_FILE", "server.log")
    daemon = os.getenv("GRPC_DAEMON", "false").lower() == "true"
    pid_file = os.getenv("GRPC_PID_FILE", "audiocloneserver.pid")
    max_message_size = int(os.getenv("GRPC_MAX_MESSAGE_SIZE_MB", "4"))
    
    logger.info(f"Starting VibeVoice AudioClone gRPC server on port {port}")
    logger.info(f"Configuration: workers={workers}, processes={processes}, log_level={log_level}")
    
    start_grpc_server(
        service_add_func=clone_interface_pb2_grpc.add_AudioCloneModelWorkerServicer_to_server,
        service_class=AudioCloneModelWorkerServicer,
        port=port,
        max_workers=workers,
        num_processes=processes,
        log_level=log_level,
        #log_file=log_file,
        daemon=daemon,
        pid_file=pid_file,
        max_message_length_mb=max_message_size,
        service_init_kwargs={
            'clone_handler': clone_handler,
            'stream_clone_handler': stream_clone_handler
        }
    )


if __name__ == "__main__":
    main()
