import glob
import torchaudio
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
        signal, samplerate = torchaudio.load(
            self.file_paths[index], out=None, normalization=True)
        mfcc_features = mfcc()
        print('getitem', signal, samplerate)
        return signal, samplerate

    def __len__(self):
        return len(self.file_paths)
