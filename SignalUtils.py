import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
from os import listdir, remove
from pydub import AudioSegment
import random
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq

def prepare_simple_feedforward_data(X, look_back = 5):
    prepared_data = []
    for x in X:
        prepared_data.extend([x for x in X])
    
    prepared_data = X.flatten()#np.asarray(prepared_data, dtype=float).flatten()
    new_data = []
    step = int(X[0].size)
    start_point = 0
    end_point = step * look_back
    while(end_point < prepared_data.shape[0]):
        new_data.extend([s for s in prepared_data[start_point:end_point]])
        start_point += step
        end_point += step

    new_data = np.asarray(new_data, dtype=float)
    print(new_data.shape)
    return new_data.reshape(int(len(new_data) / (step * look_back)), step * look_back)

def prepare_feedforward_data(X, look_back = 5):
    prepared_data = []
    for x in X:
        mean = 0
        i = 0
        for value in x:
            if i == 7:
                prepared_data.append(mean / 7)
                mean = 0
                i = 0
            mean += value
            i += 1

    new_data = []
    step = int(X[0].size / 7)
    start_point = 0
    end_point = step * look_back
    while(end_point < len(prepared_data)):
        new_data.extend([s for s in prepared_data[start_point:end_point]])
        start_point += step
        end_point += step

    return np.reshape(new_data, (int(len(new_data) / (step * look_back)), step * look_back))

def prepare_lstm_data(X, batch_size=32):
    print(X.shape)
    rows = X.shape[0]
    cols = X.shape[1]

    result = X.flatten()
    result = X.reshape(rows, 1, cols)
    return result[0:32]

def get_samples():
    paths_mapping = [(DATA_SET_COMBINE_VOICE, 1), (DATA_SET_NOISE_PATH, 0), (DATA_SET_FOLDER_PATH, 1)]
    files_mapping = []
    for path_mapping in paths_mapping:
        files_mapping.extend([(path_mapping[0] + file, path_mapping[1]) for file in listdir(path_mapping[0])])
    random.shuffle(files_mapping)
    test_index = int(0.6 * len(files_mapping))
    train_samples = files_mapping[0:test_index]
    test_samples = files_mapping[test_index:len(files_mapping)]
    return train_samples, test_samples

def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window ** 2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    try:
        assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])
    except:
        return None, None

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs

def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    try:
        sample_rate, audio = wavfile.read(filename)
    except:
        return None
    if audio.ndim >= 2:
        audio = np.mean(audio, 1)
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        return None
    if step > window:
        raise ValueError("step size must not be greater than window size")
    hop_length = int(0.001 * step * sample_rate)
    fft_length = int(0.001 * window * sample_rate)
    pxx, freqs = spectrogram(audio, fft_length=fft_length, sample_rate=sample_rate,
        hop_length=hop_length)
    if pxx is None:
        return None
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))

def combine_waves(voice_dir, noise_dir, combine_dir):
    for file in listdir(path=voice_dir):
        for i in range(10):
            try:
                sound1 = AudioSegment.from_wav(voice_dir + file)
                noise_wav = random.choice(listdir(noise_dir))
                sound2 = AudioSegment.from_wav(noise_dir + noise_wav)
                newone = sound1.overlay(sound2, loop=True)
                newone.export(combine_dir + str(i) +  file, format='wav')
            except:
                continue

def clear_white_noise(filename, old_path, new_path):
    middle_point = 128
    threshold = 1
    frame_rate, audio_bytes = wavfile.read(old_path + filename)
    new_audio = []
    for byte in audio_bytes:
        if byte > threshold + middle_point or byte < middle_point - threshold:
            new_audio.append(byte)
    wav_array = np.array(new_audio)
    wav_array = _take_max_sec(frame_rate, wav_array)
    wavfile.write(new_path + filename, frame_rate, wav_array)

def shorten(filename, old_path, new_path):
    frame_rate, audio_bytes = wavfile.read(old_path + filename)
    if audio_bytes.shape[0] < frame_rate:
        remove(old_path + filename)
        return
    print(audio_bytes.shape)
    wav_array = _take_max_sec(frame_rate, audio_bytes)
    try:
        wavfile.write(new_path + filename, frame_rate, wav_array)
    except:
        remove(old_path + filename)

def _take_max_sec(frame_rate, audio_bytes):
    max_value = audio_bytes.max()
    index = audio_bytes.tolist().index(max_value)
    start_point = int(index/2)
    if start_point >= 0 and start_point + frame_rate < audio_bytes.shape[0]:
        wav_array = np.array(audio_bytes[start_point:start_point + frame_rate])
    elif index + frame_rate <= audio_bytes.shape[0]:
        wav_array = np.array(audio_bytes[index: index + frame_rate])
    else:
        return None
    return wav_array

def delete_if_unsuitable(file):
    frame_rate, X = wavfile.read(file)
    if frame_rate != 16000:
        remove(file)
        return True
    return False

def take_single(X_test):
    single_sample = []
    for x in X_test[0]:
        single_sample.append(x)
    return np.reshape(single_sample, 1, len(single_sample))