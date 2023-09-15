import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import torchaudio

from tqdm import tqdm
from grad_extend.spec import MelExtractor


extractor = MelExtractor()


def extract_mel(wav_in, mel_out):
    y, sr = torchaudio.load(wav_in)
    assert y.size(0) == 1
    assert sr == 24000 
    mel = extractor(y)
    mel = torch.squeeze(mel, 0)
    torch.save(mel, mel_out)


def process_files(wavPath, outPath):
    files = [f for f in os.listdir(f"./{wavPath}") if f.endswith(".wav")]
    for file in tqdm(files, desc="Extract Mel"):
        file = file[:-4]
        extract_mel(f"{wavPath}/{file}.wav", f"{outPath}/{file}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="wav", dest="wav", required=True)
    parser.add_argument("--out", help="out", dest="out", required=True)

    args = parser.parse_args()
    print(args.wav)
    print(args.out)

    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out

    process_files(wavPath, outPath)
