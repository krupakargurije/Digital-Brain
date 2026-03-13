import sounddevice as sd
import numpy as np
import torch
import torchaudio
import base64
import tempfile
import os
import soundfile as sf

# Monkey-patch torchaudio to bypass SpeechBrain version compatibility issue
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: []

from speechbrain.inference.speaker import EncoderClassifier

class VoiceModule:
    """
    Handles recording audio from the microphone and generating speaker embeddings.
    """
    def __init__(self, duration_sec=3, sample_rate=16000):
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        # Load the pre-trained ECAPA-TDNN model from HuggingFace via SpeechBrain
        print("Loading voice model... (this may take a moment on first run to download the model)")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"}
        )
        print("VoiceModule initialized.")

    def capture_audio(self):
        """
        Captures audio from the default microphone for a specified duration.
        Returns:
            audio_data: A flat numpy array of audio samples, or None if failed.
        """
        print(f"Recording {self.duration_sec} seconds of audio... Please speak now.")
        try:
            # Record audio: channels=1 (mono)
            myrecording = sd.rec(
                int(self.duration_sec * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1, 
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            print("Recording complete.")
            
            # Flatten to 1D array
            audio_data = np.squeeze(myrecording)
            return audio_data
            
        except Exception as e:
            print(f"Error capturing audio: {e}")
            return None

    def generate_embedding(self, audio_data) -> list:
        """
        Generates a speaker embedding for the given audio data.
        Args:
            audio_data: Numpy array of audio samples.
        Returns:
            embedding: A list of floats representing the speaker embedding.
        """
        if audio_data is None or len(audio_data) == 0:
            return None

        # SpeechBrain expects a PyTorch tensor with shape [batch, time]
        signal = torch.from_numpy(audio_data).unsqueeze(0)
        
        # We don't need to compute gradients
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(signal)
            
        # The output is [batch, 1, embedding_dim]. We squeeze and convert to list.
        embedding_list = embeddings.squeeze().cpu().numpy().tolist()
        return embedding_list

    def generate_embedding_from_base64(self, base64_string) -> list:
        """
        Decodes a base64 encoded audio string (usually passing from Web), 
        loads it, and generates a speaker embedding.
        Args:
            base64_string: The base64 string (e.g., from MediaRecorder).
        Returns:
            embedding: A list of floats.
        """
        if not base64_string:
            return None
            
        try:
            # Strip standard data URI prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                
            audio_data = base64.b64decode(base64_string)
            
            # Write out to a temporary file
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(audio_data)
                    
                # Load with soundfile to bypass torchcodec/ffmpeg issues on Windows
                try:
                    signal_np, fs = sf.read(temp_path)
                except sf.LibsndfileError:
                    print("Soundfile failed to read webm. Attempting to read as raw bytes or fallback.")
                    return None
                    
                # Convert numpy array to torch tensor
                # soundfile returns [time, channels], torchaudio expects [channels, time]
                if len(signal_np.shape) == 1:
                    # Mono
                    signal = torch.from_numpy(signal_np).float().unsqueeze(0)
                else:
                    # Stereo
                    signal = torch.from_numpy(signal_np).float().transpose(0, 1)
                
                # If stereo, average to mono.
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0, keepdim=True)
                
                # Ensure it's 16kHz (SpeechBrain's default model rate)
                if fs != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                    signal = resampler(signal)
                    
                # Signal is [1, time], pass directly
                with torch.no_grad():
                    embeddings = self.classifier.encode_batch(signal)
                    
                embedding_list = embeddings.squeeze().cpu().numpy().tolist()
                return embedding_list
            finally:
                # Cleanup
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except PermissionError:
                    pass
                    
        except Exception as e:
            print(f"Failed to process base64 audio: {e}")
            return None

# Simple test block
if __name__ == "__main__":
    voice = VoiceModule()
    audio = voice.capture_audio()
    if audio is not None:
        print("Generating voice embedding...")
        emb = voice.generate_embedding(audio)
        if emb:
            print(f"Successfully generated voice embedding of length {len(emb)}")
        else:
            print("Failed to generate voice embedding.")
