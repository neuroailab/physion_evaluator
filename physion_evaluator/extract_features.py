import glob

import numpy as np
import argparse
import h5py
import importlib
import json
import tqdm
from physion_evaluator.dataloader_mp4s import Physion, custom_collate
import sys
from physion_evaluator.download_dataset import download_and_extract_dataset
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_class', type=str)

    parser.add_argument('--data_root_path', type=str)

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--dir_for_saving', type=str)

    parser.add_argument('--mode', type=str)

    args = parser.parse_args()

    # data_root_json = '../physion_evaluator_config.json'
    #
    # with open(data_root_json) as json_file:
    #     data_root_path = json.load(json_file)['dataset_directory']
    data_root_path = args.data_root_path

    train_mp4s = glob.glob(data_root_path + '/physion_mp4s/train/*.mp4')
    test_mp4s = glob.glob(data_root_path + '/physion_mp4s/test/*.mp4')

    # print(f"train_mp4s: {len(train_mp4s)}, test_mp4s: {len(test_mp4s)}", os.path.exists(data_root_path))

    if (not os.path.exists(data_root_path))  or (len(train_mp4s) != 5608 or len(test_mp4s) != 1035):
        print("Dataset not found. Downloading dataset from https://storage.googleapis.com/physion-dataset/physion_dataset.zip...")
        download_and_extract_dataset(data_root_path)
    else:
        print(f"Dataset already exists in {data_root_path}. Skipping download.")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, '.')

    module_name, class_name = args.model_class.rsplit(".", 1)
    module = importlib.import_module(module_name)

    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

    model = getattr(module, class_name)
    model = model().to(device).eval()

    # if args.mode == 'ocd':
    #     args.batch_size = 1

    dir_for_saving = args.dir_for_saving

    if not os.path.exists(dir_for_saving +  '/' + args.mode):
        os.makedirs(dir_for_saving +  '/' + args.mode)

    datasets = {
        'train': data_root_path + '/physion_mp4s/train/*.mp4', \
        'test': data_root_path + '/physion_mp4s/test/*.mp4'}


    transform, frame_gap, num_frames = model.transform()

    for phase in datasets.keys():

        d_path = datasets[phase]

        labels_path = os.path.join(data_root_path, 'physion_mp4s', phase, 'all_video_labels.json')

        physion_datset = Physion(transform=transform,
                                 mp4_paths=d_path,
                                 labels_path=labels_path,
                                 mode=phase,
                                 task=args.mode,
                                 frame_gap=frame_gap,
                                 num_frames=num_frames)

        physion_dataloader = DataLoader(physion_datset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.batch_size, collate_fn=custom_collate)

        all_combined_features = []

        all_labels = []

        all_filenames = []

        all_frame_labels = []

        all_scenarios = {'collision': [], 'drop': [], 'towers': [], 'link': [], 'roll': [], 'contain': [], 'dominoes': []}

        scenario_map = {}

        it = 0

        for ctx, inp in enumerate(tqdm.tqdm(physion_dataloader)):

            with torch.no_grad():

                features = model.extract_features(inp['video'].cuda())

            all_combined_features.append(features.cpu().detach().numpy())

            for ct, fname in enumerate(inp['filename']):
                for key in all_scenarios.keys():
                    if key in fname:
                        if (key == 'collision' and 'roll' not in fname) or (key != 'collision'):
                            all_scenarios[key].append(it + ct)
                            scenario_map[fname] = it + ct

            it += len(inp['filename'])
            all_filenames.append(inp['filename'])
            all_labels.append(inp['label'])
            all_frame_labels.append(inp['frame_label'])

        all_combined_features = np.concatenate(all_combined_features)
        all_labels = torch.cat(all_labels).numpy()
        all_frame_labels = torch.cat(all_frame_labels).numpy()

        all_filenames = np.concatenate(all_filenames)

        dt = h5py.special_dtype(vlen=str)

        with h5py.File(
                dir_for_saving + '/' + args.mode + '/' + phase + '_features' + '.hdf5',
                'w') as hf:

            isn = np.isnan(all_combined_features)

            if isn.any():
                print("NANs in features")
                exit(0)

            hf.create_dataset("features", data=all_combined_features)

            print("shape of saved features is:", all_combined_features.shape)

            hf.create_dataset("label", data=all_labels.astype('float'))
            hf.create_dataset("filenames", data=list(all_filenames), dtype=dt)
            hf.create_dataset("frame_labels", data=all_frame_labels)

        # File path to save the JSON data
        file_path = dir_for_saving +  '/' + args.mode + '/' + phase + '_json' + '.json'

        # Saving the dictionary as JSON data in a file
        with open(file_path, 'w') as json_file:
            json.dump(all_scenarios, json_file)

        # File path to save the JSON data
        file_path = dir_for_saving + '/' + args.mode + '/' + phase + '_scenario_map' + '.json'

        # Saving the dictionary as JSON data in a file
        with open(file_path, 'w') as json_file:
            json.dump(scenario_map, json_file)


if __name__ == "__main__":
    main()
