import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

letters = {'A':0,
           'B':1,
           'C':2,
           'D':3,
           'E':4,
           'F':5,
           'G':6,
           'H':7,
           'I':8,
           'J':9,
           'K':10,
           'L':11, 
           'M':12, 
           'N':13, 
           'O':14, 
           'P':15, 
           'Q':16, 
           'R':17, 
           'S':18, 
           'T':19, 
           'U':20, 
           'V':21, 
           'W':22, 
           'X':23, 
           'Y':24, 
           'Z':25,
          'space':26}

def read_timing_info(fpath):
    '''
    Reads a tactile timing data
    Input: file path
    Output: vector of 4:
    1. End of callibration for ATI
    2. "moving" starts at 0.2 N
    3. End of "moving" at 2.5 N and start of "staying" in static contact
    4. Start of lifting
    '''
    
    with open(fpath) as myfile:
        timings = [float(next(myfile)) for x in range(4)]
            
    return timings

# convert to spiking representation
# Assumption: threshold for derivative is 1
_last_adc = 1024
def convert_to_spikes(current_adc):
    global _last_adc
    delta_adc = current_adc - _last_adc
    _last_adc = current_adc
    if delta_adc < 0:
        return 1
    return 0

def read_letter_file(fpath, spiking=False):
    '''Reads a tactile file from path. Returns a pandas dataframe.
    Converts adc to spiking if spiking=True, if left false isPos has no meaning
    '''
    df = pd.read_csv(
        fpath,
        header=0,
        dtype={"timestamp": float, "isNeg": int, "taxel_id": int, "adc": int}
    )
    
    # isNeg is incorrect, shall be isPos
    df = df.rename(columns = {'isNeg':'isPos'})
    
    # remove some hardware noises
    df = df[df.taxel_id != -1]
            
    # Note: increasing adc means decreasing force
    if spiking:
        df.isPos = df.adc.apply(convert_to_spikes)
    
    return df

taxel_orders = np.array([36, 37, 38, 39, 40, 56, 57, 58,
                      59, 60, 31, 32, 33, 34, 35, 51,
                      52, 53, 54, 55, 26, 27, 28, 29,
                      30, 46, 47, 48, 49, 50, 21, 22,
                      23, 24, 25, 41, 42, 43, 44, 45,
                       5,  4,  3,  2,  1, 65, 64, 63,
                      62, 61, 10,  9,  8,  7,  6, 70,
                      69, 68, 67, 66, 15, 14, 13, 12,
                      11, 75, 74, 73, 72, 71, 20, 19,
                      18, 17, 16, 80, 79, 78, 77, 76]) - 1

def get_taxel_locations():
    return taxel_orders.reshape((8,10))

class LetterData:
    def __init__(self, letter, sample_id, selection, dataset_path="data/", threshold = 1):
        
        # initialize variables
        self.fpath_csv = Path(dataset_path) / f'{letter}/{letter}_{sample_id}.csv'
        self.fpath_txt = Path(dataset_path) / f'{letter}/{letter}_{sample_id}.txt'
        self.letter = letter
        self.sample_id = sample_id
        self.threshold = threshold
        
        # load data
        self.t_info = read_timing_info(self.fpath_txt)
        self.df = read_letter_file(self.fpath_csv, spiking=True)
        
        # assign selections
        _t_idx, self.T, offset = selection
        self.start_t = self.t_info[_t_idx]-offset
        self.end_t = self.start_t + self.T

    def binarize(self, bin_duration):
        bin_number = int(np.floor(self.T/bin_duration))
        data_matrix = np.zeros([80, 2, bin_number])
        
        pos_df = self.df[self.df.isPos == 1]
        neg_df = self.df[self.df.isPos == 0]
        
        count = 0
        
        init_t = self.start_t
        end_t = init_t + bin_duration
        
        while end_t <= self.T + init_t:
            _pos_count = pos_df[
                (
                    (pos_df.timestamp >= self.start_t)
                    & (pos_df.timestamp < end_t)
                )
            ]
            _pos_selective_cells = (
                _pos_count.taxel_id.value_counts() >= self.threshold
            )
            if len(_pos_selective_cells) > 0:
                
                data_matrix[
                    _pos_selective_cells[_pos_selective_cells].index.values-1,
                    0,
                    count,
                ] = 1

            _neg_count = neg_df[
                (
                    (neg_df.timestamp >= self.start_t)
                    & (neg_df.timestamp < end_t)
                )
            ]
            
            _neg_selective_cells = (
                _neg_count.taxel_id.value_counts() >= self.threshold
            )
            
            if len(_neg_selective_cells):
                data_matrix[
                    _neg_selective_cells[_neg_selective_cells].index.values-1,
                    1, 
                    count,
                ] = 1
            
            self.start_t = end_t
            end_t += bin_duration
            count += 1
            
        return data_matrix
                