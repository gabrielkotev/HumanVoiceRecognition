combined_dir = "D:/dataset/combine/"
fixed_noise_dir = "D:/dataset/fixednoise/"
fixed_voice_dir = "D:/dataset/fixedvoice/"
voice_dir = "D:/dataset/voice/"
noise_dir = "D:/dataset/other/"

def get_mapping_paths():
    return [(combined_dir, 1), (fixed_noise_dir, 0), (fixed_voice_dir, 1)]