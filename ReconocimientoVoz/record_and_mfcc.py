import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# Parámetros
SR = 16000          # muestreo (Hz)
DUR = 8.0           # segundos a grabar
NFFT = 512          # puntos FFT
FRAME_MS = 25       # tamaño de ventana (ms)
HOP_MS = 10         # salto entre ventanas (ms)
NUM_FILTERS = 40    # filtros Mel
NUM_CEPS = 13       # coeficientes MFCC
PRE_EMPH = 0.97     # pre-énfasis

def framesig(sig, frame_len, hop_len):
    num_frames = 1 + int(np.floor((len(sig) - frame_len) / hop_len))
    shape = (num_frames, frame_len)
    strides = (sig.strides[0]*hop_len, sig.strides[0])
    frames = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides).copy()
    return frames

def hamming(M):
    n = np.arange(M)
    return 0.54 - 0.46*np.cos(2*np.pi*n/(M-1))

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f/700.0)

def mel_to_hz(m):
    return 700.0 * (10**(m/2595.0) - 1.0)

def mel_filterbank(num_filters, nfft, sr, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr/2
    mmin, mmax = hz_to_mel(fmin), hz_to_mel(fmax)
    mpoints = np.linspace(mmin, mmax, num_filters + 2)
    hzs = mel_to_hz(mpoints)
    bins = np.floor((nfft+1) * hzs / sr).astype(int)
    fbanks = np.zeros((num_filters, nfft//2 + 1))
    for m in range(1, num_filters+1):
        f_m_minus, f_m, f_m_plus = bins[m-1], bins[m], bins[m+1]
        if f_m_minus == f_m: f_m = f_m_minus + 1
        for k in range(f_m_minus, f_m):
            fbanks[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbanks[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbanks

def compute_mfcc(x, sr, frame_ms, hop_ms, nfft, num_filters, num_ceps, pre_emph):
    # 1) pre-énfasis
    x = np.append(x[0], x[1:] - pre_emph*x[:-1])

    # 2) framing
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    frames = framesig(x, frame_len, hop_len)

    # 3) ventana Hamming
    win = hamming(frame_len)
    frames = frames * win

    # 4) espectro de potencia
    ps = (1.0/nfft) * np.abs(np.fft.rfft(frames, n=nfft, axis=1))**2  # (num_frames, nfft//2+1)

    # 5) banco de filtros Mel
    fb = mel_filterbank(num_filters, nfft, sr)
    mel_energies = np.maximum(ps @ fb.T, 1e-10)  # evitar log(0)

    # 6) log-energía
    log_mel = np.log(mel_energies)

    # 7) DCT-II para obtener cepstrales
    N = log_mel.shape[1]
    n = np.arange(N)
    k = np.arange(num_ceps).reshape(-1,1)
    basis = np.cos(np.pi*(n + 0.5)*k/N)
    mfcc = log_mel @ basis.T  # (num_frames, num_ceps)

    return mfcc

def main():
    print(f"Grabando {DUR} s a {SR} Hz... habla ahora.")
    audio = sd.rec(int(DUR*SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    audio = audio.squeeze()

    # Guardar WAV (16-bit PCM)
    sf.write("grabacion.wav", audio, SR, subtype='PCM_16')
    print("Guardado grabacion.wav")

    # MFCC
    mfcc = compute_mfcc(
        x=audio, sr=SR, frame_ms=FRAME_MS, hop_ms=HOP_MS,
        nfft=NFFT, num_filters=NUM_FILTERS, num_ceps=NUM_CEPS, pre_emph=PRE_EMPH
    )

    # Visualizar
    plt.figure(figsize=(9,4))
    plt.imshow(mfcc.T, aspect='auto', origin='lower')
    plt.xlabel("Frames (tiempo)")
    plt.ylabel("Coeficientes MFCC")
    plt.title("MFCC del audio grabado")
    plt.tight_layout()
    plt.savefig("mfcc_from_mic.png")
    plt.show()
    print("Listo. Se generó mfcc_from_mic.png")

if __name__ == "__main__":
    main()