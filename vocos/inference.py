import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import torchaudio

from vocos import Vocos


vocos = Vocos.from_pretrained("./vocos-mel-24khz")


def main(args):
    if args.wav is not None:
        y, sr = torchaudio.load(args.wav)
        if y.size(0) > 1:  # mix to mono
            y = y.mean(dim=0, keepdim=True)
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
        mel = vocos.encode(y)
        # print(mel.shape)
    else:
        mel = torch.load(args.mel)
        mel = mel.unsqueeze(0)
        # mel = torch.randn(1, 100, 256)  # B, C, T
    y_hat = vocos.decode(mel)
    print("save wave to vocos_save.wav")
    torchaudio.save('vocos_save.wav', y_hat, 24000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, help="Path of wave")
    parser.add_argument('--mel', type=str, help="Path of mel")
    args = parser.parse_args()

    print(args.wav)
    print(args.mel)

    if args.wav is None:
        assert os.path.isfile(args.mel)

    main(args)
