import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import datetime as dt
import torchaudio
import torch
import params

from text import text_to_sequence
from text.symbols import symbols
from text.pinyin import TTS_PinYin
from grad import GradTTS
from vocos import Vocos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-d', '--diffusion', type=int, required=True, default=1, help='1 for open, 0 for close')
    parser.add_argument('-s', '--timesteps', type=int, required=False, default=4, help='number of timesteps of reverse diffusion')
    parser.add_argument('-t', '--temperature', type=float, required=False, default=1.15, help='controls variance of terminal distribution')
    args = parser.parse_args()
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pinyin
    tts_front = TTS_PinYin("./bert", device)

    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols), params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc)['model'])
    generator.eval()
    generator.to(device)

    # torch.save({'model': generator.state_dict(),}, "grad_tts.pt")
    print(f'Number of parameters: {generator.nparams}')

    print('Initializing Vocos...')
    vocos = Vocos.from_pretrained("./vocos-mel-24khz")
    vocos.to(device)

    os.makedirs("inference_out", exist_ok=True)

    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')

            phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
            print(phonemes)
            input_ids = text_to_sequence(phonemes)

            x = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            x_lengths = torch.LongTensor([len(input_ids)]).to(device)

            bert = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)

            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, bert,
                                                   n_timesteps=args.timesteps, 
                                                   temperature=args.temperature,
                                                   use_diff=args.diffusion,
                                                   stoc=False, spk=None, length_scale=1)

            audio = vocos.decode(y_dec)
            torchaudio.save(f'./inference_out/sample_{i}.wav', audio.cpu(), 24000)

            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 24000 / (y_dec.shape[-1] * 256)}')


    print('Done. Check out `out` folder for samples.')
