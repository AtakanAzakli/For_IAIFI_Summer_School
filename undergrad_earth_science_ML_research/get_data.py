import obspy
import numpy as np
import os, shutil
import numpy as np
import pandas as pd
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow import keras

class LoadDataSets:
    """
    A class to load and process seismic datasets stored in HDF5 format.
    """
    def __init__(self, flat_it=False, data_type='spec'):
        self.flat_the_data = flat_it
        self.data_type = data_type

    def ready_to_use(self, hdf5_path, data_shape, sample_size):
        """
        Loads and prepares the dataset for training and evaluation.
        """
        all_data, labels = self.get_dataset(hdf5_path, data_shape, sample_size)

        if 0 < sample_size < 4000:
            x_train, y_train, x_test, y_test, eval_set, eval_label = self.partition_data(all_data, labels, eval_size=sample_size//10+1)
        else:
            x_train, y_train, x_test, y_test, eval_set, eval_label = self.partition_data(all_data, labels)

        return x_train, y_train, x_test, y_test, eval_set, eval_label

    def get_dataset(self, hdf5_path, data_shape, sample_size):
        """
        Retrieves the full dataset and corresponding labels.
        """
        all_data, sample_num = self.get_all_data(hdf5_path, data_shape, sample_size)
        labels = self.create_labels(sample_num)
        return shuffle(all_data, labels)

    def get_all_data(self, hdf5_path, data_shape, sample_size):
        """
        Extracts seismic data from the HDF5 file.
        """
        with h5py.File(hdf5_path, 'r') as dtfl:
            sample_num = self.get_sample_num(dtfl['Quarry'], data_shape, sample_size)
            quarry_data = self.get_data(dtfl['Quarry'], data_shape, sample_num)
            eq_data = self.get_data(dtfl['Earthquake'], data_shape, sample_num)

        all_data = np.concatenate((quarry_data, eq_data), axis=0)
        return all_data, sample_num

    def get_sample_num(self, dtfl_object, data_shape, sample_size=0):
        """
        Determines the sample size from the dataset.
        """
        if sample_size == 0:
            sample_num = sum(len(dtfl_object[key1].keys()) for key1 in dtfl_object.keys())
        else:
            sample_num = sample_size
        data_shape[0] = sample_num
        return sample_num

    def get_data(self, dtfl_object, data_shape, sample_num):
        """
        Retrieves data samples from the HDF5 file.
        """
        data = np.zeros(tuple(data_shape))
        flat_data = np.zeros([data_shape[0], data_shape[2], 3 * data_shape[1]]) if self.flat_the_data else None

        count = 0
        for event_instance in dtfl_object.keys():
            for instrument_recording in dtfl_object[event_instance].keys():
                if count >= sample_num:
                    break
                data[count] = np.array(dtfl_object[event_instance][instrument_recording])
                if self.flat_the_data:
                    flat_data[count] = self.flatten_data(data[count])
                count += 1

        return flat_data if self.flat_the_data else data

    def flatten_data(self, data):
        """
        Flattens spectrogram or waveform data.
        """
        return np.transpose(np.concatenate((data[:, :, 0], data[:, :, 1], data[:, :, 2]), axis=0)) if self.data_type == 'spec' else data

    def create_labels(self, size):
        """
        Creates binary labels for the dataset.
        """
        labels = np.concatenate((np.ones((size, 1)), np.zeros((size, 1))), axis=0)
        return labels

    def partition_data(self, all_data, labels, eval_size=3000):
        """
        Partitions the dataset into training, testing, and evaluation sets.
        """
        eval_size = int(eval_size)
        eval_set, eval_label = all_data[:eval_size], labels[:eval_size]
        all_data, labels = all_data[eval_size:], labels[eval_size:]

        x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2)
        y_train, y_test = keras.utils.to_categorical(y_train, 2), keras.utils.to_categorical(y_test, 2)
        return x_train, y_train, x_test, y_test, eval_set, eval_label

class ThreeChannelDatasets(LoadDataSets):
    """
    Handles different dataset configurations for seismic data.
    """
    def waveform_data(self, hdf5_path='Hdf5/KOERI_waveform_united.hdf5', sample_size=0):
        return self.ready_to_use(hdf5_path, [0, 6000, 3], sample_size)

    def spec_60s_data(self, hdf5_path='Hdf5/KOERI_spectrogram.hdf5', sample_size=0):
        return self.ready_to_use(hdf5_path, [0, 52, 26, 3], sample_size)

    def spec_90s_data(self, hdf5_path='Hdf5/KOERI_90s_spectrogram.hdf5', sample_size=0):
        return self.ready_to_use(hdf5_path, [0, 52, 40, 3], sample_size)

class FlattenedDatasets(LoadDataSets):
    """
    Handles flattened dataset configurations for seismic data.
    """
    def __init__(self, flat_it=True, data_type='spec'):
        super().__init__(flat_it, data_type)

    def spec_60s_data(self, hdf5_path='Hdf5/KOERI_spectrogram.hdf5', sample_size=0):
        return self.ready_to_use(hdf5_path, [0, 52, 26, 3], sample_size)

    def spec_90s_data(self, hdf5_path='Hdf5/KOERI_90s_spectrogram.hdf5', sample_size=0):
        return self.ready_to_use(hdf5_path, [0, 52, 40, 3], sample_size)



class FlattenedDatasets(LoadDataSets):
    
    def __init__(self, flat_it=True, data_type='spec'):
        self.flat_the_data = flat_it
        self.data_type = data_type
    
    SPEC_60s_PATH = 'Hdf5/KOERI_spectrogram.hdf5'
    def spec_60s_data(self, hdf5_path = SPEC_60s_PATH, sample_size=0):
        DATA_SHAPE = [0, 52, 26, 3]
        x_train, y_train, x_test, y_test, eval_set, eval_label = self.ready_to_use(hdf5_path, DATA_SHAPE, sample_size)
    
        return x_train, y_train, x_test, y_test, eval_set, eval_label
    
    
    SPEC_90s_PATH = 'Hdf5/KOERI_90s_spectrogram.hdf5'
    def spec_90s_data(self, hdf5_path = SPEC_90s_PATH, sample_size=0):
        DATA_SHAPE = [0, 52, 40, 3]
        x_train, y_train, x_test, y_test, eval_set, eval_label = self.ready_to_use(hdf5_path, DATA_SHAPE, sample_size)
    
        return x_train, y_train, x_test, y_test, eval_set, eval_label


