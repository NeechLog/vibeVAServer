#!/usr/bin/env python3
"""
Simple test script for the VibeVA interactive client.

This script provides a quick way to test the client functionality
without requiring interactive input.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to Python path to import the audiocloneclient package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "audiocloneclient"))

from audiocloneclient.client import AudioCloneClient
from audiocloneclient import clone_interface_pb2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    """Test basic connection to the server."""
    print("Testing connection to VibeVA server...")
    
    server_address = "localhost:50051"
    
    try:
        with AudioCloneClient(server_address) as client:
            print("✓ Connected to server successfully!")
            return True
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

def create_test_request():
    """Create a test request with dummy data."""
    print("Creating test request...")
    
    # Create a simple test audio (silence)
    import wave
    import struct
    
    # Create a simple WAV file with silence
    sample_rate = 24000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note
    
    samples = int(sample_rate * duration)
    audio_data = []
    
    for i in range(samples):
        # Generate a simple sine wave
        value = int(32767 * 0.1 * (i / samples))  # Quiet fade-in
        audio_data.append(value)
    
    # Convert to bytes
    audio_bytes = struct.pack('<' + 'h' * len(audio_data), *audio_data)
    
    request = clone_interface_pb2.CloneRequest()
    request.request_audio_message.text = "Hello, this is a test of the voice cloning system."
    request.sample_audio_message.audio_binary = audio_bytes
    request.model_name = "vibevoice-test"
    
    print(f"✓ Test request created (audio size: {len(audio_bytes)} bytes)")
    return request

def test_clone_request():
    """Test a clone request with the server."""
    print("\nTesting clone request...")
    
    server_address = "localhost:50051"
    
    try:
        with AudioCloneClient(server_address) as client:
            request = create_test_request()
            
            print("Sending request to server...")
            response = client.clone(request)
            
            if response.processing_meta.status_code == 200:
                print("✓ Request successful!")
                print(f"Processing time: {response.processing_meta.time_taken_ms:.2f}ms")
                if response.processing_meta.request_id:
                    print(f"Request ID: {response.processing_meta.request_id}")
                if response.processing_meta.usage_stats:
                    print("Usage Statistics:")
                    for key, value in response.processing_meta.usage_stats.items():
                        print(f"  {key}: {value}")
                if response.cloned_audio_message.audio_generator_model_name_version:
                    print(f"Model used: {response.cloned_audio_message.audio_generator_model_name_version.value}")
                print(f"Generated audio size: {len(response.cloned_audio_message.audio_binary)} bytes")
                
                # Save the output
                output_path = "test_output.wav"
                with open(output_path, 'wb') as f:
                    f.write(response.cloned_audio_message.audio_binary)
                print(f"✓ Test audio saved as: {output_path}")
                
                return True
            else:
                print(f"✗ Request failed with status {response.processing_meta.status_code}")
                if response.processing_meta.error_message:
                    print(f"Error: {response.processing_meta.error_message}")
                return False
                
    except Exception as e:
        print(f"✗ Error during request: {e}")
        logger.exception("Request failed")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("VibeVA Client Test Suite")
    print("=" * 50)
    
    # Test 1: Connection
    if not test_connection():
        print("\n❌ Connection test failed. Please ensure the server is running.")
        return 1
    
    # Test 2: Clone request
    if not test_clone_request():
        print("\n❌ Clone request test failed.")
        return 1
    
    print("\n✅ All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
