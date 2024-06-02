from glob import glob
import librosa
import soundfile as sf
from tqdm import tqdm
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pdb

datapath = 'data/maestro-v3.0.0'
files = [Path(f) for f in glob(f'{datapath}/**/*.wav', recursive=True)]

def create_training_text():    
    with open('train-maestro.txt', 'w') as f:
        lines = [f'{file[:-4]}' for file in files]
        f.write('|\n'.join(lines))

# def convert_training_audio(fs:int=24000):
#     out_directory = 'data/maestro24k'

#     for file in tqdm(files):
#         #ensure the output directory exists
#         out_dir = os.path.join(out_directory, file.parent)
#         os.makedirs(out_dir, exist_ok=True)
        
#         y, sr = librosa.load(file, sr=fs)
#         outfile = os.path.join(out_dir, file.name)
#         sf.write(outfile, y, fs)

def convert_single_file(file, fs, out_directory):
    out_dir = os.path.join(out_directory, file.parent)
    os.makedirs(out_dir, exist_ok=True)
    
    y, sr = librosa.load(file, sr=fs)
    outfile = os.path.join(out_dir, file.name)
    sf.write(outfile, y, fs)
    print(f'Converted {file.name} to {outfile}')

def convert_training_audio(fs:int=24000):
    out_directory = 'data/maestro24k'
    
    # Create a pool of worker processes
    num_processes = cpu_count()  # This will get the number of CPU cores on your machine
    with Pool(num_processes) as pool:
        # Use starmap to pass multiple arguments to the worker function
        list(tqdm(pool.starmap(convert_single_file, [(file, fs, out_directory) for file in files]), total=len(files)))

if __name__ == '__main__':
    convert_training_audio()