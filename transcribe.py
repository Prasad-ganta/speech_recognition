import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

# Load fine-tuned model
processor = Wav2Vec2Processor.from_pretrained("asr_model")
model = Wav2Vec2ForCTC.from_pretrained("asr_model")

# 📂 Load test audio file (MP3 or WAV)
filename = "path/to/audio.wav"
speech_array, sampling_rate = torchaudio.load(filename)
resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
input_audio = resampler(speech_array).squeeze().numpy()

# ✨ Predict
inputs = processor(input_audio, sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]

print("Transcription:", transcription)
