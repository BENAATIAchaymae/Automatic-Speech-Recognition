import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import torchaudio
from torchaudio.transforms import Resample

# Load the model and processor from Hugging Face
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

# Specify the path to your audio file
audio_filepath = "audio-test.wav"  # Replace with your file path

# Load your audio file
waveform, original_sampling_rate = torchaudio.load(audio_filepath)

# Resample the audio to 16000 Hz if necessary
target_sampling_rate = 16000
if original_sampling_rate != target_sampling_rate:
    resampler = Resample(orig_freq=original_sampling_rate, new_freq=target_sampling_rate)
    waveform = resampler(waveform)

# Ensure the audio is in the expected format (mono channel)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Check the number of samples in the waveform and its shape
num_samples = waveform.size(1)

# Minimum samples required for the window size (for processing)
min_samples_required = 400  # Adjust based on the model requirements or audio processing constraints

if num_samples < min_samples_required:
    print(f"Warning: The audio file has only {num_samples} samples, which is less than the minimum required {min_samples_required}.")
    # Pad the waveform to the minimum required length
    padding_needed = min_samples_required - num_samples
    waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
    print(f"Padded waveform shape: {waveform.shape}")

# Process the audio with the processor
inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=target_sampling_rate, return_tensors="pt")
input_features = inputs["input_features"]
attention_mask = inputs["attention_mask"]

# Generate the transcription
generated_ids = model.generate(input_features, attention_mask=attention_mask)

# Decode the generated ids to get the transcription
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print("Transcription:", transcription)
