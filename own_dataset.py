import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import struct
import scipy.io.wavfile as wavefile

CHUNK = 16000 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 16000                 # samples per second

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# binary data
data = stream.read(CHUNK)  

len(data)

# convert data to integers, make np array, then offset it by 127
data_int = struct.unpack(str(2 * CHUNK) + 'B', data)

test_int = np.array(data_int)

plt.plot(test_int)
plt.show()
    
# create np array and offset by 128
data_np = np.array(data_int, dtype='b')[::2] + 128

wavefile.write('d:/test.wav', 16000, test_int)