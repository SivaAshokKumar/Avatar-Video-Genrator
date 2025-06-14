import streamlit as st
import os
import subprocess
from gtts import gTTS
import base64
import shutil
from PIL import Image
import sys

# Set page config first for proper initialization
st.set_page_config(page_title="Wav2Lip Demo", layout="wide")

# Title
st.title("Wav2Lip Lip-Sync Demo")
st.write("Upload an image and audio/text to generate lip-synced video")

# Verify Python version
if sys.version_info >= (3, 11):
    st.error("Python 3.10 required - make sure you have runtime.txt with 'python-3.10.13'")
    st.stop()

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("Wav2Lip/checkpoint", exist_ok=True)

def install_ffmpeg():
    """Install FFmpeg if not available"""
    try:
        # Check if FFmpeg is already installed
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        with st.spinner("Installing FFmpeg (this may take a minute)..."):
            try:
                subprocess.run(["apt-get", "update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["apt-get", "install", "-y", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                st.error(f"Failed to install FFmpeg: {str(e)}")
                return False
    return True

# Initialize FFmpeg
if not install_ffmpeg():
    st.error("FFmpeg installation failed. The app cannot continue.")
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    static = st.checkbox("Static Image", True)
    pads = st.text_input("Padding (top bottom left right)", "0 0 0 0")
    st.markdown("---")
    if st.button("Clear temporary files"):
        if os.path.exists("temp"):
            shutil.rmtree("temp")
            os.makedirs("temp")
        st.success("Temporary files cleared!")

# File upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Input")
    img_file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])
    if img_file:
        img_path = f"temp/{img_file.name}"
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        st.image(img_path, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("Audio Input")
    audio_option = st.radio("Audio source", ["Upload WAV", "Text-to-Speech"])
    
    if audio_option == "Upload WAV":
        audio_file = st.file_uploader("Upload WAV audio", type=["wav"])
        if audio_file:
            audio_path = f"temp/{audio_file.name}"
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            st.audio(audio_path)
    else:
        text_input = st.text_area("Enter text for TTS")
        if text_input:
            audio_path = "temp/tts_output.wav"
            try:
                tts = gTTS(text_input)
                tts.save(audio_path)
                st.audio(audio_path)
            except Exception as e:
                st.error(f"TTS failed: {str(e)}")

# Process button
if st.button("Generate Lip-Sync Video"):
    if not img_file or not (audio_file if audio_option == "Upload WAV" else text_input):
        st.warning("Please upload both image and audio/text first!")
    else:
        with st.spinner("Setting up Wav2Lip (first time may take several minutes)..."):
            try:
                # Clone Wav2Lip if not exists
                if not os.path.exists("Wav2Lip"):
                    subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git"], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Patch the code for CPU compatibility
                def patch_code():
                    """Patch Wav2Lip code for CPU compatibility"""
                    with open("Wav2Lip/inference.py", "r") as f:
                        code = f.read()
                    code = code.replace("torch.cuda.is_available()", "False")
                    with open("Wav2Lip/inference.py", "w") as f:
                        f.write(code)
                
                patch_code()
                
                # Download models if not exists
                models = {
                    "wav2lip_gan.pth": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth",
                    "wav2lip.pth": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth",
                    "mobilenet.pth": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/mobilenet.pth"
                }
                
                for model_name, url in models.items():
                    if not os.path.exists(f"Wav2Lip/checkpoint/{model_name}"):
                        subprocess.run(["wget", url, "-O", f"Wav2Lip/checkpoint/{model_name}"], 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Run inference
                output_path = "temp/output.mp4"
                cmd = [
                    "python", "Wav2Lip/inference.py",
                    "--checkpoint_path", "Wav2Lip/checkpoint/wav2lip_gan.pth",
                    "--face", img_path,
                    "--audio", audio_path,
                    "--outfile", output_path,
                    "--static", str(static),
                    "--pads", *pads.split()
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if os.path.exists(output_path):
                    # Display video
                    st.success("Lip-sync complete!")
                    st.video(output_path)
                    
                    # Download link
                    with open(output_path, "rb") as f:
                        bytes = f.read()
                        b64 = base64.b64encode(bytes).decode()
                        href = f'<a href="data:video/mp4;base64,{b64}" download="lip_sync_output.mp4">Download Video</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("Processing failed. Check logs below.")
                    st.code(result.stderr)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
