"""
VibeVoice AudioCloneServer Implementation

gRPC server that implements the AudioCloneModelWorker interface using VibeVoice model.
"""

import os
import io
import logging
import torch
import soundfile as sf
import librosa
import numpy as np
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from audiocloneserver.grpc_server_launcher import start_grpc_server
from audiocloneserver import clone_interface_pb2_grpc, clone_interface_pb2
from audiocloneserver.server import AudioCloneModelWorkerServicer

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

# ============================================================================
# Model Initialization (Singleton)
# ============================================================================

MODEL_ID = os.getenv("VIBE_MODEL_ID", "microsoft/VibeVoice-1.5B")
MODEL_PATH = os.getenv("VIBE_MODEL_PATH", "./models/VibeVoice-1.5B")

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def generate_cloned_audio(text: str, voice_sample: np.ndarray) -> bytes:
    """
    Generate cloned audio using VibeVoice model.
    
    Args:
        text: Text to synthesize
        voice_sample: Voice sample as numpy array (24kHz mono)
    
    Returns:
        Generated audio as WAV bytes
    """
    # Prepare inputs for the model
    inputs = processor(
        text=[text],
        voice_samples=[voice_sample],
        return_tensors="pt"
    ).to(DEVICE)
    
    # Generate audio
    with torch.no_grad():
        output = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            max_new_tokens=512,
            cfg_scale=1.5
        )
    
    # Extract audio from output
    audio_data = None
    if hasattr(output, 'speech_outputs') and len(output.speech_outputs) > 0:
        for speech in output.speech_outputs:
            if speech is not None:
                audio_data = speech
                break
    
    if audio_data is None:
        raise RuntimeError("No valid audio generated from model")
    
    # Convert to numpy float32
    audio_numpy = audio_data.to(torch.float32).cpu().numpy()
    
    # Reshape for soundfile
    if audio_numpy.ndim == 3:
        audio_numpy = audio_numpy[0, 0, :]
    elif audio_numpy.ndim == 2:
        audio_numpy = audio_numpy[0, :]
    
    # Write to bytes buffer
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_numpy, 24000, format='WAV', subtype='PCM_16')
    audio_buffer.seek(0)
    
    return audio_buffer.read()


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
    
    response = clone_interface_pb2.CloneResponse()
    
    try:
        # Extract text from request
        text = request.request_audio_message.text
        if not text:
            raise ValueError("No text provided in request_audio_message")
        
        # Extract voice sample
        sample_audio_bytes = request.sample_audio_message.audio_binary
        if not sample_audio_bytes:
            raise ValueError("No audio_binary provided in sample_audio_message")
        
        logger.info(f"Processing clone request: text='{text[:50]}...', sample_size={len(sample_audio_bytes)}")
        
        # Process voice sample
        voice_sample, _ = process_audio_bytes(sample_audio_bytes)
        
        # Generate cloned audio
        cloned_audio_bytes = generate_cloned_audio(text, voice_sample)
        
        # Build response
        response.cloned_audio_message.audio_binary = cloned_audio_bytes
        if request.model_name:
            response.model_name = request.model_name
        
        elapsed_ms = (time.time() - start_time) * 1000
        response.meta.status_code = 200
        response.meta.time_taken_ms = elapsed_ms
        
        logger.info(f"Clone request completed in {elapsed_ms:.2f}ms")
        
    except Exception as e:
        logger.error(f"Clone request failed: {e}")
        response.meta.status_code = 500
        response.meta.error_message = str(e)
    
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
        response = clone_interface_pb2.CloneResponse()
        
        try:
            # Extract text from request
            text = request.request_audio_message.text
            if not text:
                raise ValueError("No text provided in request_audio_message")
            
            # Extract voice sample
            sample_audio_bytes = request.sample_audio_message.audio_binary
            if not sample_audio_bytes:
                raise ValueError("No audio_binary provided in sample_audio_message")
            
            logger.info(f"Processing stream clone request: text='{text[:50]}...'")
            
            # Process voice sample
            voice_sample, _ = process_audio_bytes(sample_audio_bytes)
            
            # Generate cloned audio
            cloned_audio_bytes = generate_cloned_audio(text, voice_sample)
            
            # Build response
            response.cloned_audio_message.audio_binary = cloned_audio_bytes
            if request.model_name:
                response.model_name = request.model_name
            
            elapsed_ms = (time.time() - start_time) * 1000
            response.meta.status_code = 200
            response.meta.time_taken_ms = elapsed_ms
            
            logger.info(f"Stream clone request completed in {elapsed_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Stream clone request failed: {e}")
            response.meta.status_code = 500
            response.meta.error_message = str(e)
        
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
    log_level = os.getenv("GRPC_LOG_LEVEL", "INFO")
    log_file = os.getenv("GRPC_LOG_FILE", "server.log")
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
        log_file=log_file,
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
