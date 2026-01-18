import torch
import torchaudio
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# 1. Setup paths and device
# --- Configuration ---
# Use the official Hugging Face Repo ID
MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
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
model.eval()

# --- Generation ---
print("Generating audio...")
text_input = "Speaker 1: Hello, this is a test of the remote Vibe Voice implementation."
inputs = processor(text=[text_input], return_tensors="pt").to(device)

with torch.no_grad():
        # max_new_tokens controls the length of the audio sequence
            output = model.generate(
                      **inputs, 
                      max_new_tokens=512,
                      cfg_scale=1.5
                                                )

# --- Save Result ---
output_path = "output.wav"
# VibeVoice native sample rate is 24000Hz
torchaudio.save(output_path, output[0].cpu(), 24000)
print(f"✅ Audio generated successfully and saved to {output_path}")



# 3. Configure generation settings
# Lower steps = faster; Higher steps = better quality (range 10-50)
model.set_ddpm_inference_steps(num_steps=20) 

# 4. Prepare Input
# Note: For zero-shot cloning, you would load a reference .wav here.
# For demo purposes, we'll assume a standard text input.
script = "Speaker 1: Hello! This is a test of the VibeVoice library interface."
inputs = processor(text=[script], return_tensors="pt").to(device)

# 5. Generate Audio
print("Generating...")
with torch.no_grad():
        outputs = model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        cfg_scale=1.5  # Controls how strictly it follows the prompt
                       )

# 6. Save the Result
# The output is a tensor of audio samples
audio_output = outputs[0].cpu()
torchaudio.save("library_output.wav", audio_output, sample_rate=24000)
print("Done! Audio saved to library_output.wav")


def main():
    print("Hello from vibemodel!")


if __name__ == "__main__":
    main()
