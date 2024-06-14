import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio

# Load the Whisper model and processor
model_name = "openai/whisper-large-v3"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

# Function to transcribe audio
def transcribe_audio(audio_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Print original waveform shape
    print(f"Original waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
    
    # Resample the audio if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
        print(f"Resampled waveform shape: {waveform.shape}, New sample rate: {sample_rate}")
    
    # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print(f"Mono waveform shape: {waveform.shape}")
    
    # Check waveform shape
    print(f"Final waveform shape before processing: {waveform.shape}")
    
    # Preprocess the audio
    inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")
    
    # Check the shape of the processed inputs
    print(f"Processed inputs shape: {inputs['input_features'].shape}")
    
    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    # Decode the transcription
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)
    
    return transcription[0]

# Provide the path to your audio file
audio_file_path = './audio-test.wav'

# Transcribe the audio file
transcription = transcribe_audio(audio_file_path)

# Print the transcription
print("Transcription:", transcription)
