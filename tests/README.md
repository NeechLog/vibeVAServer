# VibeVA Client Tests

This directory contains test clients for the VibeVA AudioClone service.

## Files

### `interactive_client.py`
An interactive client that allows users to:
- Select an audio file for voice cloning
- Enter text to be spoken in the cloned voice  
- Send requests to the VibeVA server via gRPC
- Save generated audio to files

### `test_client.py`
A simple test script to verify:
- Basic server connectivity
- Clone request functionality
- Audio generation and saving

## Usage

### Prerequisites
1. Make sure the VibeVA server is running on `localhost:50051` (or your configured address)
2. Install dependencies from the project root:
   ```bash
   uv sync
   ```

### Running the Interactive Client
```bash
cd /Users/gagan/projects/work/vibeVAServer
uv run tests/interactive_client.py
```

The client will prompt you for:
1. Server address (default: localhost:50051)
2. Path to an audio file containing the voice to clone
3. Text to be spoken in the cloned voice

The generated audio will be saved as `cloned_audio_*.wav` in the current directory.

### Running the Test Client
```bash
cd /Users/gagan/projects/work/vibeVAServer
uv run tests/test_client.py
```

This will run automated tests with synthetic audio data and save the output as `test_output.wav`.

## Supported Audio Formats
The client accepts common audio formats including:
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)
- AAC (.aac)

## Dependencies
The client uses the `audiocloneclient` package from `../packages/audiocloneclient`, which provides:
- gRPC client functionality
- Protocol buffer definitions
- Audio message handling

## Troubleshooting
1. **Connection failed**: Ensure the VibeVA server is running and accessible
2. **Audio file not found**: Check the file path and ensure the file exists
3. **Request failed**: Check server logs for detailed error messages
4. **Dependencies missing**: Run `uv sync` from the project root
