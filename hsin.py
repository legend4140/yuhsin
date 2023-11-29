import streamlit as st
import sounddevice as sd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play as pydub_play

def load_bert_model():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def record_audio(duration=5, sample_rate=44100):
    st.write("Click the button below and start speaking...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    return audio_data.flatten()

def preprocess_text(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

def predict_sentiment(inputs, model):
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def play_audio(audio_data, sample_rate=44100):
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )
    pydub_play(audio)

def analyze_emotion_bert(text, tokenizer, model):
    inputs = preprocess_text(text, tokenizer)
    predicted_class = predict_sentiment(inputs, model)

    # Based on model output, perform sentiment classification
    if predicted_class == 0:
        return "Oh, seems like you're feeling a bit down today."
    elif predicted_class == 1:
        return "Wow, your mood seems really positive today!"
    elif predicted_class == 2:
        return "Wow, it seems you're really excited right now!"

def main():
    st.title("Live Emotion Analysis")

    # Load the BERT model
    tokenizer, model = load_bert_model()

    if st.button("Start Recording"):
        # Record and process the audio
        audio_data = record_audio()

        if len(audio_data):
            # Convert audio to text
            text = " ".join(map(str, audio_data))

            # Perform sentiment analysis
            emotion = analyze_emotion_bert(text, tokenizer, model)
            st.write(f"Detected Emotion: {emotion}")

            # Speak the detected emotion
            speak_text(emotion)

            # Play the recorded audio
            play_audio(audio_data)

if __name__ == "__main__":
    main()
