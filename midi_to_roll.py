import pretty_midi
import torch
import numpy as np
from matplotlib import pyplot as plt


def midi_to_piano_roll_with_pedals(midi_file_path, fs=100):
    # Load the MIDI file using pretty_midi
    midi = pretty_midi.PrettyMIDI(midi_file_path)

    # Get the piano roll for notes
    piano_roll = midi.get_piano_roll(fs) / 127.0

    # Initialize pedal data arrays
    sustain_pedal_data = np.zeros(piano_roll.shape[1])
    sostenuto_pedal_data = np.zeros(piano_roll.shape[1])
    soft_pedal_data = np.zeros(piano_roll.shape[1])

    # Extract pedal control changes
    for control_change in midi.instruments[0].control_changes:
        start_time = int(control_change.time * fs)
        pedal_value = control_change.value / 127.0

        if control_change.number == 64:  # Sustain Pedal
            sustain_pedal_data[start_time:] = pedal_value
        elif control_change.number == 66:  # Sostenuto Pedal
            sostenuto_pedal_data[start_time:] = pedal_value
        elif control_change.number == 67:  # Soft Pedal
            soft_pedal_data[start_time:] = pedal_value

    # Append pedal data to the piano roll
    extended_piano_roll = np.vstack([piano_roll, sustain_pedal_data, sostenuto_pedal_data, soft_pedal_data])

    return extended_piano_roll


# Test the function
midi_file_path = "data/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
piano_roll_with_pedal = midi_to_piano_roll_with_pedals(midi_file_path)
print(piano_roll_with_pedal.shape)
