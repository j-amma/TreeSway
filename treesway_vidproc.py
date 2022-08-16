import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import video_pixel_timeseries_analysis as vptsa  # module with custom processing functions

# Processing constants
FREQMIN = 0.3  # lower bound of frequencies to consider, Hz
FREQMAX = 2    # upper bound of frequencies to consider, Hz

WINDOW = 'boxcar'  # window to apply to time series

REGION = None  # region of frame to read into array [ymin, ymax, xmin, xmax]

def main():
    ''' Computes and saves video pixel frequency spectrums'''
    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('outprefix')
    args = parser.parse_args()
    
    # parse command line args
    vid_fn = args.filename
    output_prefix = args.outprefix
    
    # add forward slash to output prefix if doesn't
    # already exist
    if output_prefix[-1] != '/':
        output_prefix = output_prefix + '/'
    
    
    # get vid name
    vid_name = vid_fn.split('/')[-1].split('.')[0]

    # read video into array (whole frame)
    vid, fps = vptsa.readvid_vreader(vid_fn, region=REGION)
    
    # compute pixel spectrums
    freq, pxx = vptsa.get_pixel_spectrums(vid, fps, FREQMIN, FREQMAX, window=WINDOW, conserve_mem=False)
    
    # save arrays
    np.save(f'{output_prefix}{vid_name}_freq', freq)
    np.save(f'{output_prefix}{vid_name}_pxx', pxx)
    

if __name__ == "__main__":
    main()
