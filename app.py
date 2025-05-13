import torch
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

app = Flask(__name__)

# ✅ Load pretrained model from Hugging Face
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['file']
    file_path = "temp.wav"
    file.save(file_path)

    # ✅ Load and preprocess audio
    speech, rate = librosa.load(file_path, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

    # ✅ Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)
