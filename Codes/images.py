import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf

def load_and_preprocess_audio(file_path):
    # 1. Original Audio Loading
    y, sr = librosa.load(file_path, sr=16000)
    fig1, ax1 = plt.subplots()
    ax1.plot(y)
    ax1.set_title('Original Audio Waveform')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')

    # 2. Normalization
    y_normalized = (y - np.mean(y)) / np.std(y)
    fig2, ax2 = plt.subplots()
    ax2.plot(y_normalized)
    ax2.set_title('Normalized Audio')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Normalized Amplitude')

    # 3. Noise Reduction (Spectral Gating)
    def spectral_noise_gate(y, sr, thresh_factor=2):
        # Compute short-time Fourier transform
        D = librosa.stft(y)
        
        # Compute the noise threshold
        noise_thresh = thresh_factor * np.mean(np.abs(D))
        
        # Apply noise gate
        D_masked = D * (np.abs(D) > noise_thresh)
        
        # Inverse STFT
        y_denoised = librosa.istft(D_masked)
        return y_denoised

    y_denoised = spectral_noise_gate(y, sr)
    fig3, ax3 = plt.subplots()
    ax3.plot(y_denoised)
    ax3.set_title('Noise Reduced Audio')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Amplitude')

    # 4. Mel Spectrogram (Original)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=128, 
        n_fft=2048, 
        hop_length=512
    )
    fig4, ax4 = plt.subplots()
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram), 
        ax=ax4, 
        x_axis='time', 
        y_axis='mel'
    )
    ax4.set_title('Original Mel Spectrogram')

    # 5. Pre-Emphasis Filtering
    def pre_emphasis(signal, coeff=0.95):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    y_pre_emphasized = pre_emphasis(y)
    fig5, ax5 = plt.subplots()
    ax5.plot(y_pre_emphasized)
    ax5.set_title('Pre-Emphasized Audio')
    ax5.set_xlabel('Sample')
    ax5.set_ylabel('Amplitude')

    # 6. Enhanced Mel Spectrogram (with Pre-Emphasis)
    mel_spectrogram_enhanced = librosa.feature.melspectrogram(
        y=y_pre_emphasized, 
        sr=sr, 
        n_mels=128, 
        n_fft=2048, 
        hop_length=512
    )
    fig6, ax6 = plt.subplots()
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_enhanced), 
        ax=ax6, 
        x_axis='time', 
        y_axis='mel'
    )
    ax6.set_title('Enhanced Mel Spectrogram')

    plt.show()

# Example usage
audio_file_path = 'D:\Digital Audio Forensics\Dataset\DS_10283_3055\ASVspoof2017_V2_train\ASVspoof2017_V2_train\T_1000981.wav'
load_and_preprocess_audio(audio_file_path)