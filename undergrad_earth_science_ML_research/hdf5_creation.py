import os
import obspy
import numpy as np
import h5py
from scipy import signal

class HDF5DataProcessor:
    """
    A class to process seismic waveform data and store it in HDF5 format.
    """
    def __init__(self, hdf5_file_name):
        self.hdf5_file_name = os.path.join(os.getcwd(), hdf5_file_name)
    
    def create_hdf5(self, event_type, file_path, preprocess=True, take_spec=False, seconds=60):
        """
        Creates an HDF5 dataset for seismic events.
        """
        with h5py.File(self.hdf5_file_name, 'a') as hdf5_file:
            hdf5_group = hdf5_file.create_group(event_type)
            waveform_samples = os.listdir(file_path)
            for file_name in waveform_samples[:20]:
                self.add_data_to_hdf5(file_path, file_name, take_spec, hdf5_group, seconds, preprocess)
        print("========Closed the hdf5 file============")
    
    def add_data_to_hdf5(self, path, file_name, take_spec, hdf5_group, seconds, preprocess):
        """Processes and adds waveform data to HDF5."""
        print("Working on:", file_name)
        st = obspy.read(os.path.join(path, file_name))
        if preprocess:
            st = self.get_preprocessed_stream_object(st)
        
        station_list = self.get_list_of_stations(st)
        for station_name in station_list:
            station_stream = st.select(station=station_name)
            if len(station_stream) == 3:
                self.add_one_instrument_data(station_stream, station_name, file_name, take_spec, seconds, hdf5_group)
            elif len(station_stream) == 6:
                self.add_two_instrument_data(station_stream, station_name, file_name, take_spec, seconds, hdf5_group)
        st.clear()
    
    def get_preprocessed_stream_object(self, stream_object):
        """Preprocesses seismic waveform data."""
        stream_object.merge(fill_value=0)
        stream_object.detrend('demean')
        stream_object.filter('bandpass', freqmin=1.0, freqmax=20, corners=2, zerophase=True)
        stream_object.taper(max_percentage=0.001, type='cosine', max_length=2)
        stream_object.interpolate(100, method="linear")
        return stream_object
    
    def get_list_of_stations(self, st):
        """Extracts unique station names from the stream."""
        return list(set(tr.stats.station for tr in st))
    
    def add_one_instrument_data(self, station_stream, station_name, file_name, take_spec, seconds, hdf5_group):
        """Processes single instrument station data."""
        print("Working on:", station_stream[0].stats.station)
        try:
            numpy_data = self.get_numpy_data(station_stream, seconds)
            if numpy_data.shape == (seconds*100, 3):
                data_ = self.take_spectrogram(numpy_data) if take_spec else numpy_data
                hdf5_group.create_dataset(f"{file_name}/{station_name}", data=data_)
                print('Successful:', file_name, ';', station_name)
        except Exception as e:
            print(f'Failed: {file_name} ; {station_name}, Error: {e}')
    
    def get_numpy_data(self, station_stream, seconds):
        """Converts station stream data to a NumPy array."""
        try:
            sta_chanE = np.array(station_stream.select(component='E')[0].data)[:, np.newaxis][:seconds*100]
            sta_chanZ = np.array(station_stream.select(component='Z')[0].data)[:, np.newaxis][:seconds*100]
            sta_chanN = np.array(station_stream.select(component='N')[0].data)[:, np.newaxis][:seconds*100]
            return np.concatenate((sta_chanE, sta_chanZ, sta_chanN), axis=1)
        except IndexError:
            raise ValueError("Incomplete station data")
    
    def take_spectrogram(self, numpy_data):
        """Computes the spectrogram of the waveform data."""
        _, _, spec_E = signal.spectrogram(numpy_data[:, 0], 100, window=('tukey', 2.56))
        _, _, spec_Z = signal.spectrogram(numpy_data[:, 1], 100, window=('tukey', 2.56))
        _, _, spec_N = signal.spectrogram(numpy_data[:, 2], 100, window=('tukey', 2.56))
        return np.concatenate((spec_E[1:53, :, np.newaxis], spec_Z[1:53, :, np.newaxis], spec_N[1:53, :, np.newaxis]), axis=2)
    