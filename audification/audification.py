import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.manifold import TSNE
import csv
import pyaudio
import time
import wave
import itertools
import argparse
import json
import requests
from scipy.io import wavfile
import sys
from pylab import *
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy import signal
import matplotlib.ticker as tick
from threading import Barrier, Thread


class ConfigReader:
    def __init__(self, method="pca"):
        self.method = method
        self.val = self.read_min_max_config(method)


    def read_min_max_config(self, method):
        data = {}
        with open("values.json") as json_data:
            data = json.load(json_data)
        return data[method]


calc_storage = []
global_times = []
audio_file = "false"

def audification(method, m, cr):
    start = time.time()

    twoD = window(m, 200, method)
    fcol = [row[0] for row in twoD]
    vcol = [row[1] for row in twoD]

    frequency = [min_max_scaling(20.00, 5000.00, cr["x_min"], cr["x_max"], x) for x in fcol]
    volume = [min_max_scaling(0, 1, cr["y_min"], cr["y_max"], x) for x in vcol]

    p = pyaudio.PyAudio()
    out = []
    for x in range(0, len(frequency)):
       prepare_Tones(frequency[x], volume[x], out)

    print("Write audio file")
    write_Audio_File(method+".wav", out, p)
    end = time.time()

    return (end-start)


#Reads in the appropriate csv file
def read_in_File():
    m = []
    with open('final.csv', 'r') as f:
        reader = csv.reader(f, delimiter=";")
        cnt = 0
        for line in reader:
            if cnt > 0:
                del line[0] # delete service, its categorial
                m.append(list(map(float, line)))
            else:
                m.append(line)
            cnt += 1

    del m[0]  # remove header
    return m

#Window function
def window(iterable, size, method):
    #fill in the first n elements
    buf = []
    result = []
    for x in range(0, size-1):
        buf.append(iterable[x])

    #delete inital vectors
    iterable = iterable[size-1:len(iterable)]

    timearr = []
    #calculate sliding windows
    for v in iterable:
        buf.append(v)
        transformed = transform(buf, method, calc_storage)
        #pop first element
        iterable = iterable[1:len(iterable)-1]
        #take the last calculated element to the result vector
        result.append(transformed[len(transformed)-1])

    return result


def get_min_max(vector):
    result = {}
    v_min = min(vector)
    v_max = max(vector)

    result["min"] = v_min
    result["max"] = v_max

    return result


def transform(matrix, method, timearr):
    trans = None
    start = time.clock()
    if method == "pca":
        trans = PCA(n_components=2).fit_transform(matrix)
    elif method == "random":
        trans = random_projection.GaussianRandomProjection(n_components=2, eps=0.5, random_state=None).fit_transform(matrix)
    else:
        trans = TSNE(n_components=2, n_iter=250, method='barnes_hut').fit_transform(matrix)
    end = time.clock()

    timearr.append((end-start))
    return trans


def min_max_scaling(scaleMin, scaleMax, min, max, value):
    if value > max:
        #scale to max
        value = max

    if value < min:
        #scale to min
        value = min

    out = (value - min) / (max - min)
    range2 = scaleMax - scaleMin
    out = (out * range2) + scaleMin
    out = round(out, 2)

    return out


def prepare_Tones(f, volume, out):

    #volume = 0.5  # range [0.0, 1.0]
    fs = 44100  # sampling rate, Hz, must be integer
    duration = 0.1  # in seconds, may be float
    # generate samples, note conversion to float32 array
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
    out.append(volume * samples)


def write_Audio_File(fname, frames, p):
    if audio_file == "true":
        wf = wave.open(fname, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()

def plot_Audio_File(fpath,method):
    myaudio = AudioSegment.from_file(fpath, "wav")
    chunk_length_ms = 20  # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms)

    #first chunk
    f_chunk = np.frombuffer(chunks[0].get_array_of_samples(), dtype=np.int16)

    #midd chunk
    midd = int(len(chunks)/2)
    mid_chunk = np.frombuffer(chunks[midd].get_array_of_samples(), dtype=np.int16)

    #last chunk
    last_chunk = np.frombuffer(chunks[len(chunks)-1].get_array_of_samples(), dtype=np.int16)

    time = np.arange(0, len(f_chunk), 1)
    time = (time / 44100) * 1000 / 2

    #clear old stuff in plot
    plt.clf()
    plt.rcParams["figure.figsize"] = [10, 5]
    fig = plt.figure(1)
    f, arr = plt.subplots(2, 3)
    #plt.subplot(211)
    arr[0, 0].plot(time, f_chunk, color='k', linewidth=0.5)
    arr[0, 0].set_title("[a]")
    arr[0, 0].set_ylabel("Amplitude")
    y_fmt = tick.FuncFormatter(downscale)
    arr[0, 0].yaxis.set_major_formatter(y_fmt)

    arr[0, 1].plot(time, mid_chunk, color='k', linewidth=0.5)
    arr[0, 1].set_title("[b]")
    arr[0, 1].yaxis.set_visible(False)
    arr[0, 1].set_xlabel("Time (ms)")
    arr[0, 2].plot(time, last_chunk, color='k', linewidth=0.5)
    arr[0, 2].set_title("[c]")
    arr[0, 2].yaxis.set_visible(False)

    #plot frequency and power
    n = len(f_chunk)
    nUniquePts = int(ceil((n + 1) / 2.0))
    freqArray = arange(0, nUniquePts, 1.0) * (44100 / n)

    f_p = fft(f_chunk)  # take the fourier transform
    m_p = fft(mid_chunk)
    l_p = fft(last_chunk)

    f_p = f_p[0:nUniquePts]
    m_p = m_p[0:nUniquePts]
    l_p = l_p[0:nUniquePts]

    f_p = abs(f_p)
    m_p = abs(m_p)
    l_p = abs(l_p)

    f_p = f_p / float(n)  # scale by the number of points so that
    m_p = m_p / float(n)
    l_p = l_p / float(n)

    # the magnitude does not depend on the length
    # of the signal or on its sampling frequency
    f_p = f_p ** 2  # square it to get the power
    m_p = m_p ** 2
    l_p = l_p ** 2
    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0:  # we've got odd number of points fft
        f_p[1:len(f_p)] = p[1:len(f_p)] * 2
        m_p[1:len(m_p)] = p[1:len(m_p)] * 2
        l_p[1:len(l_p)] = p[1:len(l_p)] * 2
    else:
        f_p[1:len(f_p) - 1] = f_p[1:len(f_p) - 1] * 2  # we've got even number of points fft
        m_p[1:len(m_p) - 1] = m_p[1:len(m_p) - 1] * 2
        l_p[1:len(l_p) - 1] = l_p[1:len(l_p) - 1] * 2


    arr[1, 0].plot(freqArray / 1000, 10 * log10(f_p), color='k', linewidth=0.5)
    arr[1, 0].set_ylabel("Power (dB)")
    arr[1, 0].set_title("[d]")
    arr[1, 1].plot(freqArray / 1000, 10 * log10(m_p), color='k', linewidth=0.5)

    arr[1, 1].set_xlabel("Frequency (kHz)")
    arr[1, 1].set_title("[e]")
    arr[1, 1].yaxis.set_visible(False)

    arr[1, 2].plot(freqArray / 1000, 10 * log10(l_p), color='k', linewidth=0.5)
    arr[1, 2].set_title("[f]")
    arr[1, 2].yaxis.set_visible(False)

    f.subplots_adjust(hspace=0.3, wspace=2.3)
    f.tight_layout()
    plt.savefig(method + "_audio_representation.png", dpi=300, format="png")


def downscale(x, pos):
    converted = x / 1000
    if converted != 0:
        return "%d k" % converted
    else:
        return "%d" % converted


def plot_scatter(x_vec, y_vec, method):
    fig = plt.scatter(x_vec, y_vec, color='k')
    #plt.title(method+" 2D Transformation")
    plt.savefig(method + "_transformation.png", dpi=300, format="png")


def do_Performance_Test(howMany, method, m, config):
    #do the calculation in a thread
    for i in range(0, howMany):
        global_times.append(audification(method,m, config))


def main_func():
    parser = argparse.ArgumentParser(description="Audification of Network Data")
    parser.add_argument('-m', '--method', help='Method of Transformation [pca, random or tsne]', required=True)
    parser.add_argument('-r', '--rounds', help='How often the audification is invoked', default=1)
    parser.add_argument('-n', '--numThreads', help='Number of threads', default=1)
    parser.add_argument('-fr', '--fileWriting', help='Decide, whether audiofile will be written or not', default="false")
    args = vars(parser.parse_args())

    method = args["method"]
    numThreads = args["numThreads"]
    howMany = args["rounds"]
    audio_file = args["fileWriting"]

    print("Reading in config file and data file")
    config = ConfigReader(method=args["method"]).val
    m = read_in_File()

    rounds = int(howMany/numThreads)
    threads = []

    print("Start audification process")
    for i in range(0, numThreads):
        threads.append(Thread(target=do_Performance_Test, args=(rounds, method, m, config)))
        threads[i].start()

    #barrier... wait for threads to finish their work
    for t in threads:
        t.join()

    all_avg = round(average(global_times), 4)
    all_min = round(min(global_times), 4)
    all_max = round(max(global_times), 4)
    all_median = round(median(global_times), 4)
    all_variance = round(var(global_times), 4)
    all_sdev = round(std(global_times), 4)

    print("Audification lasts for", howMany, "rounds:")
    print("Min ", all_min, "Max ", all_max, "Avg ", all_avg, "Median ", all_median, "Variance", all_variance, "SDEV",
      all_sdev)

    window_avg = round(average(calc_storage), 4)
    window_min = round(min(calc_storage), 4)
    window_max = round(max(calc_storage), 4)
    window_median = round(median(calc_storage), 4)
    window_variance = round(var(calc_storage), 4)
    window_sdev = round(std(calc_storage), 4)

    print("Calculation for each flow lasts:")
    print("Min ", window_min, "Max", window_max, "Avg ", window_avg, "Median ", window_median, "Variance",
      window_variance, "SDEV", window_sdev)


main_func()