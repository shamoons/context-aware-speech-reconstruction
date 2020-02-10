import glob
import torchaudio
from soundfile import SoundFile
from python_speech_features import mfcc
from torch.utils.data import Dataset


class CorruptedAudioDataset(Dataset):
    def __init__(self, corrupted_path, train_set=False, test_set=False):
        audio_file_paths = list(
            sorted(glob.iglob(corrupted_path + '**/*.flac', recursive=True)))

        cutoff_index = int(len(audio_file_paths) * 0.9)

        if train_set == True:
            self.file_paths = audio_file_paths[0: cutoff_index]
        if test_set == True:
            self.file_paths = audio_file_paths[cutoff_index:]

    def __getitem__(self, index):
        corrupted_sound_file = SoundFile(self.file_paths[index])
        corrupted_samplerate = corrupted_sound_file.samplerate
        corrupted_signal_audio_array = corrupted_sound_file.read()

        clean_path = self.file_paths[index].split('/')
        # print(self.file_paths[index], clean_path)
        clean_sound_file = SoundFile(self.file_paths[index])
        clean_samplerate = clean_sound_file.samplerate
        clean_signal_audio_array = clean_sound_file.read()


        # corrupted_mfcc = mfcc(corrupted_signal_audio_array, samplerate=corrupted_samplerate)
        # clean_mfcc = mfcc(clean_signal_audio_array, samplerate=clean_samplerate)

        # print(corrupted_signal_audio_array, clean_signal_audio_array)
        print('return', corrupted_signal_audio_array.shape, clean_signal_audio_array.shape)
        return corrupted_signal_audio_array, clean_signal_audio_array

    def __len__(self):
        return len(self.file_paths)
