
# Audio Transcription with OpenAI Whisper

This repository contains a script for transcribing audio files using the OpenAI Whisper model. The script leverages the Hugging Face `transformers` library along with `torchaudio` to process and transcribe audio data.

## Features
- **Automatic Speech Recognition (ASR)**: Uses the `openai/whisper-large-v3` model for high-quality transcriptions.
- **Audio Preprocessing**: Automatically resamples audio to 16kHz and converts stereo to mono if necessary.
- **Easy to Use**: Simply provide the path to your audio file, and the script handles the rest.

## Requirements
- `torch`
- `transformers`
- `torchaudio`

## Usage
1. Clone the repository and navigate to the directory.
2. Install the required packages:
   ```bash
   pip install torch transformers torchaudio
