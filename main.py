import streamlit as st
from huggingface_hub import notebook_login
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment, notebook
import whisper

# Web app title
st.title('音声書き起こしアプリ')

# OpenAI API Key and Huggingface token input
api_key = st.secrets["openai_api_key"]
huggingface_token = st.secrets["huggingface_token"]

# Huggingface login 
notebook_login(token=huggingface_token)

# Load models
speaker_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=huggingface_token)
whisper_model = whisper.load_model("large")

# Upload the audio file
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください（.wav、.mp3）", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save the uploaded file
    with open('uploaded_audio.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display processing message
    processing_message = st.empty()
    processing_message.text('音声ファイルを処理中です。しばらくお待ちください...')

    # Apply speaker diarization
    who_speaks_when = speaker_diarization('uploaded_audio.wav')
    
    # Reset notebook visualization
    notebook.reset()
    
    # Rename speakers
    who_speaks_when = who_speaks_when.rename_labels()
    
    # Transcribe each speaker's speech
    st.write('音声書き起こし結果：')
    audio = Audio(sample_rate=16000, mono=True)
    first_minute = Segment(0, 5600)
    for segment, _, speaker in who_speaks_when.crop(first_minute).itertracks(yield_label=True):
        waveform, sample_rate = audio.crop('uploaded_audio.wav', segment)
        text = whisper_model.transcribe(waveform.squeeze().numpy(), language='ja')["text"]
        speaker_name = st.text_input(f'話者{speaker}の名前を入力してください', f'話者{speaker}')
        st.write(f'{speaker_name}: {text}')
    
    # Remove processing message
    processing_message.empty()

# Error handling (TODO: implement specific error handling logic)
try:
    # ...
    pass
except Exception as e:
    st.write(f'エラーが発生しました: {e}')

# Display privacy policy and terms of service (TODO: provide actual privacy policy and terms of service)
st.write('プライバシーポリシー')
st.write('利用規約')
