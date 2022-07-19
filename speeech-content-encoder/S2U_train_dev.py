import h5py
import numpy as np
import joblib
import torchaudio
from tqdm import tqdm
import os
import torch
import glob

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000

tags = ['line_nr', 'dialog_id', 'turn_id']


def parse_key(key_str):
    data = {}
    for idx, tag in enumerate(tags):
        value = ''
        if key_str.find(tag) >= 0:
            if idx < len(tags) - 1:
                value = key_str[key_str.find(tag) + len(tag) + 1:key_str.find(tags[idx + 1])]
            else:
                value = key_str[key_str.find(tag) + len(tag) + 1:]
            value = value.strip()
        data[tag] = value
    return data


class ApplyKmeans(object):
    def __init__(self, km_path, return_diff=False):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.return_diff = return_diff
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = torch.sqrt(
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            min_dist = dist.detach().min(dim=1)
            if self.return_diff:
                return min_dist.indices.cpu().numpy(), min_dist.values.cpu().numpy()
            else:
                return min_dist.indices.cpu().numpy()
        else:
            dist = np.sqrt(
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            if self.return_diff:
                return np.argmin(dist, axis=1), np.min(dist, axis=1)
            else:
                return np.argmin(dist, axis=1)


tmp_file = 'tmp.wav'


def reader(fname):
    wav_list = []
    try:
        hd5_file = h5py.File(fname, 'r')
    except OSError:
        print(f'{fname} file is abnormal')
        return wav_list

    for key in hd5_file.keys():
        wav = hd5_file[key]['audio'][:]
        parsed_key = parse_key(str(key))

        # create temporary wav file to get torch audio vector
        import soundfile as sf
        sf.write(file=tmp_file, data=wav, samplerate=16000)
        wav, ori_sr = torchaudio.load(tmp_file)

        if ori_sr != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
        os.remove(path=tmp_file)
        wav_list.append({
            'wav': wav.squeeze(),
            'key': parsed_key['line_nr'] + '_' + parsed_key['dialog_id'].removesuffix('.json') + '_' + parsed_key['turn_id']
        })
    hd5_file.close()
    return wav_list




train_folders = ['tpa', 'tpb', 'tpc', 'tpd']
for folder in train_folders:
    print(f'{folder} start')

    # train
    train_file_list = glob.glob(f'data/base/train/{folder}/*.hd5')

    output_dir = f'data/SQA_code/train/{folder}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    extractor = torch.hub.load('s3prl/s3prl', 'hubert_large_ll60k')
    extractor.eval()
    extractor = extractor.cuda()

    apply_kmeans = ApplyKmeans(
        'ext/DUAL-textless-SQA/speeech-content-encoder/km_100h_c128/km_feat_layer_22')

    for train_file in tqdm(train_file_list, desc='transforming passage to discrete code'):
        file = os.path.basename(train_file)
        waves = reader(train_file)

        for idx, wav in enumerate(waves):
            if len(wav['wav']) > 20 * SAMPLE_RATE:
                continue

            feature = extractor([wav['wav'].cuda()])

            code = apply_kmeans(feature['hidden_state_22'].squeeze().cuda())
            code = torch.tensor(code)

            merged_code, counts = torch.unique_consecutive(code, return_counts=True)
            np.savetxt(os.path.join(output_dir, f'{wav["key"]}.code'), merged_code.long(), fmt='%i')
            np.savetxt(os.path.join(output_dir, f'{wav["key"]}.cnt'), counts.long(), fmt='%i')

    print(f'{folder} end')

# dev

output_dir = 'data/SQA_code/dev'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

dev_file_list = glob.glob('data/base/dev/*.hd5')

print('dev folder start')
for dev_file in tqdm(dev_file_list, desc='transforming passage to discrete code'):
    file = os.path.basename(dev_file)
    waves = reader(dev_file)

    for idx, wav in enumerate(waves):

        if len(wav['wav']) > 20 * SAMPLE_RATE:
            print(f'wav is too long {dev_file}_{idx}')
            continue

        feature = extractor([wav['wav'].cuda()])

        code = apply_kmeans(feature['hidden_state_22'].squeeze().cuda())
        code = torch.tensor(code)

        merged_code, counts = torch.unique_consecutive(code, return_counts=True)
        np.savetxt(os.path.join(output_dir, f'{wav["key"]}.code'), merged_code.long(), fmt='%i')
        np.savetxt(os.path.join(output_dir, f'{wav["key"]}.cnt'), counts.long(), fmt='%i')
