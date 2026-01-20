import torch
import soundfile as sf
import librosa
import numpy as np
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

def resample_to_24khz(audio_tensor, original_sr):
    """Resample audio tensor to 24kHz using librosa."""
    print(f"   🔧 Auto-resampling to 24kHz...")
    
    # Resample using librosa
    audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension for resampling
    audio_resampled = librosa.resample(
        audio_tensor.numpy(), 
        orig_sr=original_sr, 
        target_sr=24000
    )
    audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0)  # Add batch dimension back
    print(f"✅ Resampled to 24kHz. New shape: {audio_tensor.shape}")
    return audio_tensor, 24000


def validate_audio_input(audio_path):
    """
    Proves audio can be loaded and formatted for the model 
    without relying on torchaudio backends.
    """
    print(f"--- Proof Check for: {audio_path} ---")
    
    try:
        # 1. Load raw data using soundfile (more stable on remote servers)
        data, samplerate = sf.read(audio_path)
        print(f"✅ Raw Load: Success. Detected Sample Rate: {samplerate}Hz")
        
        # 2. Convert to Float32 Tensor (Model requirement)
        audio_tensor = torch.from_numpy(data).float()
        
        # 3. Handle Stereo to Mono (Crucial for VibeVoice)
        if len(audio_tensor.shape) > 1:
            print(f"⚠️  Detected {audio_tensor.shape[1]} channels. Merging to Mono...")
            audio_tensor = audio_tensor.mean(dim=1)
            
        # 4. Final Shape Check: Model expects [1, samples]
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        print(f"✅ Final Tensor Shape: {audio_tensor.shape} (Channels, Samples)")
        
        # 5. Model Compatibility Verification
        is_compatible = (samplerate == 24000)
        if is_compatible:
            print("✅ Compatibility: PERFECT (24kHz)")
        else:
            print(f"❌ Compatibility: FAILED. Model needs 24000Hz, got {samplerate}Hz.")
            audio_tensor, samplerate = resample_to_24khz(audio_tensor, samplerate)

        return audio_tensor, samplerate

    except Exception as e:
        print(f"❌ PROOF FAILED: Could not read file. Error: {e}")
        return None, None


# 1. Setup paths and device
# --- Configuration ---
# Use the official Hugging Face Repo ID
# MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
MODEL_ID = "microsoft/VibeVoice-1.5B"
model_path = "./models/VibeVoice-1.5B"

# Detection for Intel Mac (CPU) vs GPU Container (CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

# 2. Load the Processor and Model
print("Loading VibeVoice model...")
processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch_dtype, 
    device_map=device
)
#model.eval()
print("Model loaded")


ref_audio, sr = validate_audio_input("/home/jovyan/voice_assist/prod/voice_sample/test-124/converted_audio_voice_0.wav")
text_input = "Speaker 1: Hello, this is a test of the remote Vibe Voice implementation. Speak loudly and clearly.Do not hesitate to speak."

print(f"Audio tensor shape: {ref_audio.shape}")
print(f"Audio tensor dtype: {ref_audio.dtype}")
print(f"Sample rate: {sr}")

# Try different audio formats for the processor
print("Trying audio as numpy array...")
ref_audio_np = ref_audio.squeeze(0).numpy()  # Convert to numpy and remove batch dim
print(f"Audio numpy shape: {ref_audio_np.shape}")

inputs = processor(
    text=[text_input], 
    voice_samples=[ref_audio_np],  # Try with numpy array
    return_tensors="pt"
).to(device)

print(f"Processor inputs keys: {list(inputs.keys())}")
for key, value in inputs.items():
    if hasattr(value, 'shape'):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")

# --- Generation ---
print("Generating audio...")


with torch.no_grad():
    # max_new_tokens controls the length of the audio sequence
    print("🔧 Starting model generation...")
    print(f"📝 Input keys for generation: {list(inputs.keys())}")
    print(f"🎯 Device: {device}")
    print(f"⚙️  Generation params: max_new_tokens=512, cfg_scale=1.5")
    
    output = model.generate(
        **inputs, 
        tokenizer=processor.tokenizer,
        max_new_tokens=512,
        cfg_scale=1.5
    )
    
    print(f"✅ Generation complete!")
    print(f"🎵 Output type: {type(output)}")
    print(f"📊 Full output structure:")
    print(f"   - Type: {type(output)}")
    
    # Check for speech_outputs (the actual audio data)
    if hasattr(output, 'speech_outputs'):
        print(f"   - speech_outputs: {type(output.speech_outputs)}")
        if len(output.speech_outputs) > 0:
            for i, speech in enumerate(output.speech_outputs):
                if speech is not None:
                    print(f"     speech_outputs[{i}]: {speech.shape} ({speech.dtype})")
                else:
                    print(f"     speech_outputs[{i}]: None")
        else:
            print(f"     speech_outputs: Empty list")
    
    # Check for reach_max_step_sample
    if hasattr(output, 'reach_max_step_sample'):
        print(f"   - reach_max_step_sample: {output.reach_max_step_sample.shape} ({output.reach_max_step_sample.dtype})")
    
    # Check other potential attributes
    if hasattr(output, 'audio_spectrograms'):
        print(f"   - Audio spectrograms shape: {output.audio_spectrograms.shape}")
        print(f"   - Audio spectrograms dtype: {output.audio_spectrograms.dtype}")
    if hasattr(output, 'audio_waveforms'):
        print(f"   - Audio waveforms shape: {output.audio_waveforms.shape}")
        print(f"   - Audio waveforms dtype: {output.audio_waveforms.dtype}")
    
    # Show all attributes if it's a dataclass or similar
    if hasattr(output, '__dict__'):
        print(f"   - All attributes: {list(output.__dict__.keys())}")
    
    # Check if output has sequences attribute and log its properties
    audio_final = None
    if hasattr(output, 'sequences'):
        print(f"🔍 Output sequences analysis:")
        seq = output.sequences
        print(f"   - sequences type: {type(seq)}")
        print(f"   - sequence shape: {seq.shape if hasattr(seq, 'shape') else 'No shape attr'}")
        print(f"   - sequence dimensions: {seq.dim() if hasattr(seq, 'dim') else 'No dim attr'}")
        print(f"   - sequence dtype: {seq.dtype if hasattr(seq, 'dtype') else 'No dtype attr'}")
        print(f"   - sequence device: {seq.device if hasattr(seq, 'device') else 'No device attr'}")
        
        # Test CPU conversion
        try:
            seq = seq.squeeze().cpu()
            if seq.dim() == 1:
                audio_final = seq.unsqueeze(0)
                print(f"   - ✅ Squeezed and unsqueezed successfully")
            else:
                audio_final = seq[0]
                print(f"   - ✅ Squeezed and selected first element successfully")
            cpu_seq = audio_final
            print(f"   - ✅ CPU conversion successful")
            print(f"   - CPU sequence shape: {cpu_seq.shape}")
            print(f"   - CPU sequence dtype: {cpu_seq.dtype}")
            
            # Test numpy conversion
            try:
                np_seq = cpu_seq.numpy()
                print(f"   - ✅ Numpy conversion successful")
                print(f"   - Numpy sequence shape: {np_seq.shape}")
                print(f"   - Numpy sequence dtype: {np_seq.dtype}")
                
                # Test soundfile compatibility
                try:
                    # Check if values are in valid audio range
                    min_val = np_seq.min()
                    max_val = np_seq.max()
                    print(f"   - Value range: [{min_val:.4f}, {max_val:.4f}]")
                    
                    # Test dtype conversion for soundfile
                    if np_seq.dtype in ['float32', 'float64', 'int16', 'int32']:
                        print(f"   - ✅ Soundfile compatible dtype: {np_seq.dtype}")
                    else:
                        print(f"   - ⚠️  Converting to float32 for soundfile...")
                        np_seq = np_seq.astype('float32')
                        print(f"   - ✅ Converted dtype: {np_seq.dtype}")
                        
                except Exception as e:
                    print(f"   - ❌ Soundfile compatibility check failed: {e}")
                    
            except Exception as e:
                print(f"   - ❌ Numpy conversion failed: {e}")
                
        except Exception as e:
            print(f"   - ❌ CPU conversion failed: {e}")
    else:
        print(f"   - No 'sequence' attribute found in output")

# # 3. Configure generation settings
# # Lower steps = faster; Higher steps = better quality (range 10-50)
# model.set_ddpm_inference_steps(num_steps=20)


# --- Save Result ---
output_path = "output.wav"
# VibeVoice native sample rate is 24000Hz
output_audio = None

# Access the audio from speech_outputs
if hasattr(output, 'speech_outputs') and len(output.speech_outputs) > 0:
    # Find first non-None speech output
    audio_data = None
    for i, speech in enumerate(output.speech_outputs):
        if speech is not None:
            audio_data = speech
            print(f"🎵 Found valid audio at speech_outputs[{i}]: {audio_data.shape}")
            break
    
    if audio_data is not None:
        print(f"🔍 Audio data before conversion: {audio_data.dtype}, shape: {audio_data.shape}")
        # 1. Ensure audio is Float32 and on CPU
        # audio_data is the tensor from your model output
        audio_numpy = audio_data.to(torch.float32).cpu().numpy()

        # Convert to float32 to avoid BFloat16 issues
        if audio_data.dtype == torch.bfloat16:
            audio_data = audio_data.float()
            print(f"   - ✅ Converted speech audio from BFloat16 to Float32")
        
        print(f"🔍 Audio data after conversion: {audio_data.dtype}")
        
        # Move to CPU first, then convert to numpy
        audio_cpu = audio_data.cpu()
        print(f"🔍 Audio data on CPU: {audio_cpu.dtype}")
        
        output_audio = audio_cpu.numpy().astype('float32')
        
        # Ensure audio is in correct format for WAV file
        if len(output_audio.shape) > 1:
            output_audio = output_audio.flatten()  # Flatten to 1D for WAV
        
        print(f"🔍 Final audio shape for saving: {output_audio.shape}, dtype: {output_audio.dtype}")
        print(f"🔍 Audio value range: [{output_audio.min():.4f}, {output_audio.max():.4f}]")


        # 2. Reshape if necessary
        # soundfile expects [samples, channels]. VibeVoice usually gives [batch, channels, samples]
        if audio_numpy.ndim == 3: # [1, 1, samples]
            audio_numpy = audio_numpy[0, 0, :]
        elif audio_numpy.ndim == 2: # [1, samples]
            audio_numpy = audio_numpy[0, :]

        # 3. Simple write command (this handles the headers automatically)
        sf.write('output_numpy.wav', audio_numpy, 24000, subtype='PCM_16')
        sf.write(output_path, output_audio, 24000)
        print(f"✅ Audio generated successfully and saved to {output_path}")
    else:
        print("❌ No valid audio data found in speech_outputs (all None)")
else:
    print("❌ No speech_outputs found in generation result")
print("🔄 Trying to use audio_final from sequence...")

# Try to use audio_final from sequence analysis
if (output_audio is None or isinstance(output_audio, np.ndarray)) and 'audio_final' in locals() and audio_final is not None and not isinstance(audio_final, np.ndarray):
    # Convert to float32 to avoid BFloat16 issues
    if hasattr(audio_final, 'dtype') and audio_final.dtype == torch.bfloat16:
        audio_final = audio_final.float()
        print(f"   - ✅ Converted audio_final from BFloat16 to Float32")
    
    print(f"🎵 Using audio_final from sequence: {audio_final.shape}")
    output_audio = audio_final.numpy().astype('float32')
    
    # Ensure audio is in correct format for WAV file
    if len(output_audio.shape) > 1:
        output_audio = output_audio.flatten()  # Flatten to 1D for WAV
    
    print(f"🔍 Final audio shape for saving: {output_audio.shape}, dtype: {output_audio.dtype}")
    print(f"🔍 Audio value range: [{output_audio.min():.4f}, {output_audio.max():.4f}]")
    
    sf.write("output_seq.wav", output_audio, 24000)
    print(f"✅ Audio generated successfully and saved to output_seq.wav")
else:
    print("❌ No valid audio data found in sequence either")

def main():
    print("Hello from vibemodel!")


if __name__ == "__main__":
    main()
