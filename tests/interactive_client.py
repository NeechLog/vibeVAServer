#!/usr/bin/env python3
"""
Interactive client for VibeVA AudioClone service.

This client allows users to:
1. Select an audio file for voice cloning
2. Enter text to be spoken in the cloned voice
3. Send the request to the VibeVA server via gRPC
4. Save the generated audio to a file
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add the parent directory to Python path to import the audiocloneclient package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "audiocloneclient"))

from audiocloneclient.client import AudioCloneClient
from audiocloneclient import clone_interface_pb2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_audio_file_path() -> Optional[str]:
    """Prompt user for audio file path and validate it exists."""
    while True:
        file_path = input("\nEnter the path to your voice sample audio file (or 'quit' to exit): ").strip()
        
        if file_path.lower() in ['quit', 'exit', 'q']:
            return None
        
        if not file_path:
            print("Please enter a valid file path.")
            continue
        
        # Convert to absolute path
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        if not os.path.isfile(file_path):
            print(f"Path is not a file: {file_path}")
            continue
        
        # Check if it's an audio file (basic check)
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in audio_extensions:
            print(f"Warning: {file_ext} may not be a supported audio format.")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        return file_path

def get_text_to_clone() -> Optional[str]:
    """Prompt user for text to be cloned."""
    while True:
        text = input("\nEnter the text you want to speak in the cloned voice (or 'quit' to exit): ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            return None
        
        if not text:
            print("Please enter some text.")
            continue
        
        if len(text) > 500:
            print("Text is quite long. Consider shorter text for better results.")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        return text

def read_audio_file(file_path: str) -> bytes:
    """Read audio file as bytes."""
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read audio file {file_path}: {e}")
        raise

def save_audio_file(audio_bytes: bytes, output_path: str):
    """Save audio bytes to file."""
    try:
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        logger.info(f"Audio saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save audio file {output_path}: {e}")
        raise

def main():
    """Main interactive client loop."""
    print("=" * 60)
    print("VibeVA Interactive Audio Clone Client")
    print("=" * 60)
    print("\nThis client connects to the VibeVA server to clone voices.")
    print("Make sure the VibeVA server is running on localhost:50051")
    
    server_address = "localhost:50051"
    
    # Allow custom server address
    custom_address = input(f"\nServer address [{server_address}]: ").strip()
    if custom_address:
        server_address = custom_address
    
    print(f"\nConnecting to server at: {server_address}")
    
    try:
        with AudioCloneClient(server_address) as client:
            print("✓ Connected to server successfully!")
            
            while True:
                print("\n" + "-" * 40)
                print("New Clone Request")
                print("-" * 40)
                
                # Get audio file
                audio_file = get_audio_file_path()
                if audio_file is None:
                    print("Goodbye!")
                    break
                
                # Get text to clone
                text = get_text_to_clone()
                if text is None:
                    print("Goodbye!")
                    break
                
                print(f"\nProcessing request...")
                print(f"Audio file: {audio_file}")
                print(f"Text: '{text}'")
                
                try:
                    # Read audio file
                    audio_bytes = read_audio_file(audio_file)
                    print(f"✓ Audio file loaded ({len(audio_bytes)} bytes)")
                    
                    # Create request
                    request = clone_interface_pb2.CloneRequest()
                    request.request_audio_message.text = text
                    request.sample_audio_message.audio_binary = audio_bytes
                    request.model_name = "vibevoice-1.5b"
                    
                    print("Sending request to server...")
                    
                    # Send request
                    response = client.clone(request)
                    
                    # Check response
                    if response.processing_meta.status_code == 200:
                        print("✓ Request successful!")
                        print(f"Processing time: {response.processing_meta.time_taken_ms:.2f}ms")
                        
                        # Print all processing metadata
                        print("\n=== Processing Metadata ===")
                        print(f"Request ID: {response.processing_meta.request_id}")
                        print(f"Status Code: {response.processing_meta.status_code}")
                        print(f"Time Taken: {response.processing_meta.time_taken_ms:.2f}ms")
                        if hasattr(response.processing_meta, 'error_message') and response.processing_meta.error_message:
                            print(f"Error Message: {response.processing_meta.error_message}")
                        
                        # Print all usage statistics
                        if response.processing_meta.usage_stats:
                            print("\n=== Usage Statistics ===")
                            for key, value in sorted(response.processing_meta.usage_stats.items()):
                                print(f"  {key}: {value}")
                        
                        # Print audio message metadata
                        print("\n=== Audio Message Metadata ===")
                        print(f"Generated audio size: {len(response.cloned_audio_message.audio_binary)} bytes")
                        
                        if response.cloned_audio_message.audio_generator_model_name_version:
                            print(f"Model used: {response.cloned_audio_message.audio_generator_model_name_version.value}")
                        
                        # Print all audio message fields if available
                        if hasattr(response.cloned_audio_message, 'audio_file_path') and response.cloned_audio_message.audio_file_path:
                            print(f"Audio file path: {response.cloned_audio_message.audio_file_path}")
                        
                        if hasattr(response.cloned_audio_message, 'audio_format') and response.cloned_audio_message.audio_format:
                            print(f"Audio format: {response.cloned_audio_message.audio_format}")
                        
                        if hasattr(response.cloned_audio_message, 'sample_rate') and response.cloned_audio_message.sample_rate:
                            print(f"Sample rate: {response.cloned_audio_message.sample_rate}")
                        
                        # Print all additional model info if available
                        if hasattr(response.cloned_audio_message, 'additional_model_info'):
                            for info in response.cloned_audio_message.additional_model_info:
                                print(f"Additional model info - {info.name}: {info.value}")
                        
                        print("\n" + "="*50)
                        
                        # Save output
                        output_filename = f"cloned_audio_{len(text)}chars.wav"
                        output_path = os.path.join(os.getcwd(), output_filename)
                        
                        save_audio_file(response.cloned_audio_message.audio_binary, output_path)
                        print(f"✓ Audio saved as: {output_filename}")
                        
                    else:
                        print(f"✗ Request failed with status {response.processing_meta.status_code}")
                        if response.processing_meta.error_message:
                            print(f"Error: {response.processing_meta.error_message}")
                
                except Exception as e:
                    print(f"✗ Error during request: {e}")
                    logger.exception("Request failed")
                
                # Ask if user wants to continue
                continue_choice = input("\nMake another request? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("Goodbye!")
                    break
    
    except Exception as e:
        print(f"✗ Failed to connect to server: {e}")
        print("\nPlease ensure:")
        print("1. The VibeVA server is running")
        print(f"2. The server is accessible at {server_address}")
        print("3. All dependencies are installed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
