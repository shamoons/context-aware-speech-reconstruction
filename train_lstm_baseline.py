import argparse
import torchaudio
import torch
import numpy
from datasets import CorruptedAudioDataset
from models import BaselineLSTM

torch.manual_seed(0)
numpy.random.seed(0)

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

    train(train_loader)

def train(train_loader):
    model = BaselineLSTM()
    for epoch in range(300):
        for inputs, outputs in train_loader:
            print('inputs', inputs)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            # targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            output_scores = model(inputs)
            print('output_scores', output_scores)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # loss = loss_function(tag_scores, targets)
            # loss.backward()
            # optimizer.step()

if __name__ == '__main__':
    main()
