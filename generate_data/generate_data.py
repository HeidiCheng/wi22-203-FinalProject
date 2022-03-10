import numpy as np
import random
import glob

from pydub import AudioSegment

dirpath = "/Users/heidicheng/Desktop/heidi/Convex_Optimization/project/data/"
DATA = 1000

def gen_music(paths, notes_number, chords_number):

    chords = AudioSegment.empty()
    for i in range(chords_number):
        samples = random.choices(guitar_path, k=notes_number)

        output_sound = AudioSegment.from_wav(samples[0])

        for i, s in enumerate(samples):
            if i == 0:
                continue
            output_sound = output_sound.overlay(AudioSegment.from_wav(s))
        chords = chords + output_sound
    return chords

def gen_chords(paths, instruments, num_of_notes, num_of_data):

    for n in range(num_of_data):

        samples = random.sample(range(0, len(paths[0])), k=num_of_notes)

        for ins, p in enumerate(paths):
            
            new_chord = AudioSegment.from_wav(p[samples[0]])

            for i, s in enumerate(samples):
                if i == 0:
                    continue
                new_chord = new_chord.overlay(AudioSegment.from_wav(p[s]))
            
            file_handle = new_chord.export(dirpath  + instruments[ins] + "_chords/"+ str(num_of_notes) + "/" + str(n).zfill(4) + ".wav", format="wav")
            
    return
    
    
def main():
    instruments = ["guitar1", "piano1"]
    guitar_path = []
    piano_path = []
    paths = []
    notes_range = [2, 3, 4, 5, 6]

    #guitar_sounds = []
    #piano_sounds = []

    # guitar
    for path in glob.glob(dirpath + "guitar1/*.wav"):
        guitar_path.append(path)
        #guitar_sounds.append(path)
        
    paths.append(guitar_path)

    # piano
    for path in glob.glob(dirpath + "piano1/*.wav"):
        piano_path.append(path)
        #piano_sounds.append(AudioSegment.from_wav(path))
        
    paths.append(piano_path)

    for notes in notes_range:
        gen_chords(paths, instruments, notes, DATA)

 
if __name__ == "__main__":
    main()


