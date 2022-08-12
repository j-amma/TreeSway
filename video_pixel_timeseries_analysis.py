import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy.signal
import scipy.stats
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

def readvid_cv(fn, region=None, verbose=True):
    """
    Uses opencv to read video region into array.
    
    Returns a 4-d array [num_frames, y, x, channel] and the video framerate
    
    fn : path to video
    region : defaults to None and reads whole frame, 
             when specified: list [ymin, ymax, xmin, xmax] 
    verbose : when true, prints helpful progress messages
    """
    
    vid_capture = cv2.VideoCapture(fn)
    print('Open Successful: ', vid_capture.isOpened())
    
    ymin, ymax, xmin, xmax = [0, 0, 0, 0]
    if region is None:
        ymin = 0 
        ymax = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        xmin = 0
        xmax = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        ymin, ymax, xmin, xmax = region[0], region[1], region[2], region[3]
    
    num_frame = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    
    outputdata = np.zeros((num_frame, ymax-ymin, xmax-xmin, 3), dtype=np.uint8)
    
    if (verbose):
        print('Reading video region into array')
    
    for i in range(num_frame):
        ret, frame = vid_capture.read()
        outputdata[i] = frame[ymin:ymax, xmin:xmax]
        if(verbose): 
            sys.stdout.write('\r{0}'.format(i))
            sys.stdout.flush()
    
    if (verbose):
        print('\n Finished reading video')
            
    return outputdata, fps

def isolate_channel(vid, channel='r'):
    """
    Isolates one channel of 4d video array.
    
    Returns 3d array [frames, y, x] where each value represents the channel
    brightness at that frame and coordinate.
    
    vid : np array storing uncompressed video (4d array [num_frames, y, x, channel])
    channel : string letter 'r', 'g', or 'b'
    """
    # by convention, opencv uses the bgr order
    if channel == 'b':
        return vid[:,:,:,0]
    elif channel == 'g':
        return vid[:,:,:,1]
    elif channel == 'r':
        return vid[:,:,:,2]
    else:
        print("did not pass 'b', 'g', or 'r'")
        
# Compute power of two greater than or equal to `n`
# https://www.techiedelight.com/round-next-highest-power-2/
def findNextPowerOf2(n):
    """
    Returns next power of 2 greater than or equal to n
    
    https://www.techiedelight.com/round-next-highest-power-2/
    """
    
    k = 1
    while k < n:
        k = k << 1
 
    return k
        
def get_pixel_spectrums(pixts, fps, freqmin, freqmax, channel='r', window='boxcar', detrend='constant'):
    """
    Computes spectrum for each pixel time series, performs frequency thresholding.
    
    Returns freq, 1d array representing frequency range, bounded by freqmin and freqmax
    Returns pxx, 3d array containing the power magnitudes at each pixel (for each frequency in freq), 
        bounded by freqmin and freqmax
        
    
    pixts : pixel time series, 3d array [num_frames, y, x]
    fps : frame rate of video
    freqmin : float, lower bound for frequency thresholding
    freqmax : float, upper bound for frequency thresholding
    channel : rgb channel to compute spectrum over
    window : window to apply to time series prior to fft
    detrend : string, see scipy.signal.periodogram for details
    """
    
    pixts_iso = isolate_channel(pixts, 'r')
    
    # get frequency of whole frame
    next_pow2 = findNextPowerOf2(len(pixts[:,0,0]))
    freq, pxx = scipy.signal.periodogram(pixts_iso, fps, window='boxcar', nfft=next_pow2, detrend='constant', axis=0)

    # filter by freq range
    low = np.where(freq > freqmin)[0][0]
    high = np.where(freq > freqmax)[0][0]
    pxx = pxx[low:high]
    freq = freq[low:high]
    
    return freq, pxx

def get_peak_freqs(freq, pxx):
    """
    Returns 2d array containing freq corresponding to max power for each pixel
    
    freq : 1d array with spectrum frequencies
    pxx : 3d array with power magnitudes at each pixel
    """
    peak_freq_idx = np.argmax(pxx, axis=0)
    peak_freq = freq[peak_freq_idx]
    return peak_freq

def mask_peak_freq(mask_approach, peak_freq, pxx):
    """
    Masks pixels in region according to filtering scheme.
    
    mask_approach : string, 'mean of max', 'mean of mean', 'median of max'
    peak_freq : 2d array with peak frequency at each pixel in region,
    pxx : 3d array with power magnitudes at each pixel
    """
    mask = None
    
    if mask_approach == 'mean of max':
        mask = pxx.max(axis=0) < (pxx.max(axis=0).ravel().mean())
    elif mask_approach == 'mean of mean':
        mask = pxx.max(axis=0) < (pxx.mean(axis=0).ravel().mean())
    elif mask_approach == 'median of max':
        mask = pxx.max(axis=0) < np.median(pxx.max(axis=0).ravel())
    elif mask_approach == 'peak amp large compated to spectrum':
        # TODO
        return None
    else:
        print('No valid threshold provided')
        return None
    
    peak_freq_masked = ma.array(peak_freq, mask=mask)
    return peak_freq_masked

def plot_vid_region(vid, peak_freq, peak_freq_masked, freqmin, freqmax, figsize):
    """
    Plots raw region, frequency map, and masked frequency map
    
    vid : 3D or 4d array, raw frame
    peak_freq : 2d array with peak frequency at each pixel
    peak_freq_masked : 2d masked peak frequency array
    freqmin : minimum frequency considered
    freqmax : maximum frequency considered
    """
    fig, axs = plt.subplots(1,3, figsize=figsize)
    
    axs[0].imshow(vid[0])
    axs[0].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    axs[0].set_title('Region of Interest')
    
    cmap = 'magma'
    
    im1 = axs[1].imshow(peak_freq, cmap=cmap, vmin=freqmin, vmax=freqmax)
    axs[1].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    divider = make_axes_locatable(axs[1])
    cax1 = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Hz')
    axs[1].set_title('Peak Frequencies')
    
    im2 = axs[2].imshow(peak_freq_masked, cmap=cmap, vmin=freqmin, vmax=freqmax)
    axs[2].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    divider = make_axes_locatable(axs[2])
    cax2 = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Hz')
    axs[2].set_title('Masked Peak Frequencies')
    
def get_freq_specs(peak_freq, peak_freq_masked):
    """
    Returns Pandas dataframe containing pixel aggregation statistics.
    
    Contains mean, median, and mode for masked and unmasked peak frequency arrays.
    
    peak_frq : 2d array with peak frequncy for each pixel
    peak_freq_masked : 2d masked peak frequency array
    """
    mean = peak_freq.ravel().mean()
    median = np.median(peak_freq.ravel())
    mode, x = scipy.stats.mode(peak_freq.ravel())
    unmasked = [mean, median, mode[0]]
    
    mean_ma = peak_freq_masked.ravel().mean()
    median_ma = ma.median(peak_freq_masked.ravel())
    mode_ma, x = scipy.stats.mode(peak_freq_masked.ravel())
    masked = [mean_ma, median_ma, mode_ma[0]]
    
    data = [unmasked, masked]
    
    stats_df = pd.DataFrame(data, ['Unmasked Peak Frequencies', 'Masked Peak Frequencies'], ['mean', 'median', 'mode'])

    return stats_df
