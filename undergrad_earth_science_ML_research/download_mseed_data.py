import os
import pandas as pd
import numpy as np
from datetime import datetime
from obspy import read_inventory, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.iris import Client as Client2
import warnings

warnings.filterwarnings('ignore')

class SeismicDataProcessor:
    """
    Class for processing seismic waveform data from an FDSN web service.
    """
    def __init__(self, metadata_file='UUSS_metadata.csv', response_dir='Response_Files', output_dir='Waveform'):
        self.client = Client('IRIS')
        self.client2 = Client2()
        self.df = pd.read_csv(metadata_file)
        self.response_dir = response_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.unique_chans = {'EHE', 'EHN', 'EHZ', 'HHE', 'HHN', 'HHZ', 'BHE', 'BHN', 'BHZ', 'ENE', 'ENN', 'ENZ'}

    def process_events(self):
        """
        Processes seismic waveform data for each event in the dataset.
        """
        for _, row in self.df.iterrows():
            self.process_event(row)

    def process_event(self, row):
        """
        Process a single seismic event.
        """
        start = UTCDateTime(row.datetime) - 10
        end = UTCDateTime(row.datetime) + 120
        
        try:
            st = self.client.get_waveforms(network=row.net, station=row.station, location='*', channel='*', starttime=start, endtime=end)
            inv_path = os.path.join(self.response_dir, f"{row.net}.{row.station}.xml")
            inv = read_inventory(inv_path, format='STATIONXML')
            
            st = self.clean_channels(st)
            st = self.merge_channels(st)
            
            if len(st) > 0:
                st.trim(max([tr.stats.starttime for tr in st]), min([tr.stats.endtime for tr in st]))
                st.resample(100)
                st.detrend()
                st.taper(0.01, type='hann')
                
                for tr in st:
                    tr.remove_response(inventory=inv, output='ACC')
                
                st.filter('highpass', freq=1)
                
                baz = self.client2.distaz(stalat=row.stla, stalon=row.stlo, evtlat=row.evla, evtlon=row.evlo)
                st.rotate(method='NE->RT', back_azimuth=baz['backazimuth'])
                
                output_path = os.path.join(self.output_dir, f"{row.id}_{row.station}.mseed")
                st.write(output_path, format='MSEED')
                print(f"Successful: {row.datetime}")
        
        except Exception as e:
            print(f"Failed: {row.datetime}, Error: {e}")
            
    def clean_channels(self, stream):
        """
        Removes unwanted channels from the stream.
        """
        return Stream(traces=[tr for tr in stream if tr.stats.channel in self.unique_chans])

    def merge_channels(self, stream):
        """
        Merges duplicate channels.
        """
        tmp_chan = [tr.stats.channel for tr in stream]
        multi_chan = {chan for chan in tmp_chan if tmp_chan.count(chan) > 1}
        
        for dub_chan in multi_chan:
            st_tmp = stream.select(channel=dub_chan)
            stream = Stream(traces=[tr for tr in stream if tr.stats.channel != dub_chan])
            st_tmp.merge(method=1, fill_value=0)
            stream += st_tmp
        
        return stream

if __name__ == "__main__":
    processor = SeismicDataProcessor()
    processor.process_events()
