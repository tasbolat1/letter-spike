import argparse
import numpy as np
import glob
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import Parallel, delayed
from pathlib import Path
import time

from utils.utils import read_timing_info, read_letter_file, LetterData, letters

NUMBER_SAMPLES = 50

parser = argparse.ArgumentParser(description="Tactile data preprocessor.")

parser.add_argument("--save_path", type=str, help="Location to save data to.", required=True)
parser.add_argument("--data_path", type=str, help="Path to dataset.", required=True)
parser.add_argument("--threshold", type=int, help="Threshold for tactile.", default=1)
# parser.add_argument("--seed", type=int, help="Random seed to use", default=100)

parser.add_argument(
    "--selection",
    type=str,
    help="Timing information for preprocessing. Read explanation below",
    choices=["fixed_time", "moving", "moving_and_touching", "touching"],
    default='fixed_time',
)
parser.add_argument("--bin_duration", type=float, help="Binning duration.", required=True)
parser.add_argument("--total_duration", type=float, help="Total duration of trainable data (valid for fixed time selection).", default=2.5)
parser.add_argument(
	"--offset",
	type=float, 
	help="Offset to compensate touch movement before 0.2N (note: this will be added to total_duration).",
	default=0.15)
# parser.add_argument("--test_size", type=float, help="test size in percentage", default=20)

# parser.add_argument(
#     "--k_folds",
#     type=int,
#     help="Number of splits for stratified K-folds.",
#     default=4,
# )

args = parser.parse_args()

selections = {'fixed_time': [1, args.total_duration+args.offset, args.offset],
              'moving': [1, 0.7+args.offset, args.offset],
              'moving_and_touching': [1, 1.7+args.offset, args.offset],
              'touching':[2, 1.0, 0]}

selection = selections[args.selection]

# Convert tactile into big matrix
def tact_bin_save(letter, sample_id, count, save_path=None):
    tac_data = LetterData(letter=letter,
     					  sample_id=sample_id,
     					  selection=selection,
     					  dataset_path=args.data_path,
     					  threshold=args.threshold)
    tacData = tac_data.binarize(args.bin_duration).astype(np.uint8)

    if save_path is not None:
    	f = save_path / f"{letter}_{sample_id}.npy"
    	print(f"Writing {f}...")
    	np.save(f, tacData)

    return tacData, letter, count, sample_id

start_time = time.time()
print(f'Starting preprocess of {len(letters)} letters with following configs:')
print(f'selection: {args.selection}')
print(f'bin_duration: {args.bin_duration}')
# print(f'test_size: {args.test_size}')
# print(f'k_folds: {args.k_folds}')


# check save_dir existence

if not os.path.exists(Path(args.save_path)):
    print(f'Creating save_path at {args.save_path}')
    os.makedirs(Path(args.save_path))
else:
    print(f'save_path already exists: {args.save_path} -> The directory will be rewritten with new files.')

big_list_tact = []
count = 0
for letter in letters:
    for sample_id in range(NUMBER_SAMPLES):
        big_list_tact.append([letter, sample_id, count])
        count += 1

# run data in parallel
data_with_labels = Parallel(n_jobs=18)(delayed(tact_bin_save)(*zz) for zz in big_list_tact)

all_data = []
#labels = []
#indices = []
#sample_ids = []
info = []
for _data, _label, _idx, _sample_id in data_with_labels:
    all_data.append(_data)
    #labels.append(_label)
    #indices.append(_idx)
    #sample_ids.append(_sample_id)
    info.append([_label, _sample_id, _idx])
    
all_data = np.stack(all_data, axis=0)

print('Done with binning')
print('Data shape: ', all_data.shape)
np.save(Path(args.save_path)/'data.npy', all_data)

np.savetxt(
    Path(args.save_path) / f"info.txt",
    np.array(info),
    fmt="%s",
    delimiter=",",
)

# # Prepare samples for k-Fold
# test_split_size = int( args.test_size*len(letters)*NUMBER_SAMPLES/100 )
# train_split_size = len(letters)*NUMBER_SAMPLES - test_split_size


# # Total samples
# total_samples = train_split_size+test_split_size
# print(f'Train test split sizes: {train_split_size}, {test_split_size}')

# x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(all_data, labels, indices, test_size=test_split_size, stratify=labels, random_state=args.seed)

# np.savetxt(Path(args.save_path)/'test.txt',
#            np.stack([idx_test, y_test], axis=0).astype(np.int).T,
#            fmt='%i',
#            delimiter=',' )

# skf = StratifiedKFold(n_splits=args.k_folds, random_state=args.seed, shuffle=True)

# train_indices = []
# val_indices = []
# for _train_indices, _val_indices in skf.split(idx_train, y_train):
#     _temp_val_indices = []
#     for _v in _val_indices:
#         _temp_val_indices.append([idx_train[_v], y_train[_v]])
#     val_indices.append(_temp_val_indices)
    
#     _temp_train_indices = []
#     for _t in _train_indices:
#         _temp_train_indices.append([idx_train[_t], y_train[_t]])
#     train_indices.append(_temp_train_indices)
    
# for fold in range(args.k_folds):
#     np.savetxt(
#         Path(args.save_path) / f"train_{fold+1}.txt",
#         np.array(train_indices[fold], dtype=int),
#         fmt="%i",
#         delimiter=",",
#     )
#     np.savetxt(
#         Path(args.save_path) / f"val_{fold+1}.txt",
#         np.array(val_indices[fold], dtype=int),
#         fmt="%i",
#         delimiter=",",
#     )
    
print('Done!')
# Save files
print(f'Preprocess finished in {time.time()-start_time:.2f} seconds')

