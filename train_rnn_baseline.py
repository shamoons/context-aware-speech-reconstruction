import argparse
import torchaudio
import torch
from datasets import CorruptedAudioDataset

torch.manual_seed(0)


def get_args():
    parser = argparse.ArgumentParser(description='Train an RNN Baseline.')
    parser.add_argument('--clean_path', default='../speech-enhancement-asr/data/LibriSpeech/dev-clean/',
                        help='Path to clean files')

    parser.add_argument('--corrupted_path', default='../speech-enhancement-asr/data/LibriSpeech/dev-noise-subtractive-250ms-1/',
                        help='Path to corrupted files')

    return parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    corrupted_path = args.corrupted_path

    train_set = CorruptedAudioDataset(corrupted_path, train_set=True)
    test_set = CorruptedAudioDataset(corrupted_path, test_set=True)

    print("Train set size: " + str(len(train_set)))
    print("Test set size: " + str(len(test_set)))

    # needed for using datasets on gpu
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=True, **kwargs)

    my_item = train_set.__getitem__(1)


if __name__ == '__main__':
    main()
