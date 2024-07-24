import torch
import torchaudio
import pandas as pd

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        # Get the most probable indices
        indices = torch.argmax(emission, dim=-1)
        # Remove consecutive duplicates and blank tokens
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        # Convert indices to labels
        return "".join([self.labels[i] for i in indices])

def transcribe_audio(audio_path: str, model, decoder, device):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure waveform is in the expected format (mono channel)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to match the model's expected sample rate
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Move waveform to the same device as the model
    waveform = waveform.to(device)
    
    with torch.no_grad():
        # Get the emission probabilities from the model
        emission, _ = model(waveform)
        transcript = decoder(emission[0])
    
    return transcript

if __name__ == "__main__":
    # Path to the audio file
    audio_path = 'audio-test.wav'  # Replace with your file path

    # Initialize the Wav2Vec2 model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.get_model().to(device)
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())

    # Transcribe the audio file
    transcript = transcribe_audio(audio_path, model, decoder, device)
    print("Transcription:", transcript)
