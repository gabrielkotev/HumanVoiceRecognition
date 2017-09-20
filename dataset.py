import SignalUtils as su
from os import listdir
import os.path

combined_dir = "D:/dataset/combine/"
fixed_noise_dir = "D:/dataset/fixednoise/"
fixed_voice_dir = "D:/dataset/fixedvoice/"
voice_dir = "D:/dataset/voice/"
noise_dir = "D:/dataset/other/"

def get_mapping_paths():
    return [(combined_dir, 1), (fixed_noise_dir, 0), (fixed_voice_dir, 1)]


'''

'''
for file in listdir(voice_dir):   
    deleted = su.delete_if_unsuitable(voice_dir + file)
    if deleted:
        continue

    if os.path.isfile(fixed_voice_dir + file):
        continue
    
    su.shorten(file, voice_dir, fixed_voice_dir)

for file in listdir(noise_dir):
    deleted = su.delete_if_unsuitable(noise_dir + file)
    if deleted:
        continue

    if os.path.isfile(fixed_noise_dir + file):
        continue
    
    su.shorten(file, noise_dir, fixed_noise_dir)

# combine the files to have with noise and without
su.combine_waves(fixed_voice_dir, fixed_noise_dir, combined_dir)

