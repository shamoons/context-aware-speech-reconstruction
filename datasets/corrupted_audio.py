import glob
import torchaudio
import torchaudio
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
        corrupted_signal = torchaudio.load(
            self.file_paths[index], out=None, normalization=True)
        corrupted_sound_data = corrupted_signal[0].permute(1, 0)
        # corrupted_sound_file = SoundFile(self.file_paths[index])
        # corrupted_samplerate = corrupted_sound_file.samplerate
        # corrupted_signal_audio_array = corrupted_sound_file.read()

        clean_path = self.file_paths[index].split('/')
        # print(self.file_paths[index], clean_path)
        # clean_sound_file = SoundFile(self.file_paths[index])
        # clean_samplerate = clean_sound_file.samplerate
        # clean_signal_audio_array = clean_sound_file.read()
        clean_signal = torchaudio.load(
            self.file_paths[index], out=None, normalization=True)
        clean_sound_data = clean_signal[0].permute(1, 0)

        # corrupted_mfcc = mfcc(corrupted_signal_audio_array, samplerate=corrupted_samplerate)
        # clean_mfcc = mfcc(clean_signal_audio_array, samplerate=clean_samplerate)

        # print(corrupted_signal_audio_array, clean_signal_audio_array)
        print(corrupted_sound_data.size(), clean_sound_data.size(), '\n')
        return corrupted_sound_data, clean_sound_data

    def __len__(self):
        return len(self.file_paths)
