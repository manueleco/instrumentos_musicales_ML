# ================== SCRIPT FOR MACOS ==================

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import scipy
import glob
import os
from numba import jit, cuda
from numba import vectorize

import pandas as pd

from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers
import keras
from keras.models import Sequential
import warnings

audios = []
listaAudios = []
datas = []
srs = []

files = []

audiosLen = []

posDatas = []

noDir = 'Audios/'

# os.chdir("Audios")


def audioDatabase():
    for file in glob.glob("Audio_res/*.wav"):
        audios.append(file)
        # print(file)

    print("Los audios son: ", audios)


# ================================== FUNCIÓN PARA EXTRAER CARACTERÍSTICAS DEL AUDIO ===========================

def audioFeatures(audioFile):
    data, sr = librosa.load(audioFile, sr=44100)
    dataRms, ind = librosa.effects.trim(data, top_db=25)

    rmsShape = dataRms.shape
    dataShape = data.shape
    #print(type(data), type(sr))
    # print(data)
    # print(sr)
    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x ** 2)

    return data, sr, dataRms, rmsShape, dataShape, ind


def listaLimpia():
    for aud in audios:  # iterating on a copy since removing will mess things up
        new_string = aud.replace(noDir, "")
        listaAudios.append(new_string)
    print("Hola somos la lista simplificada: ", listaAudios)
    # print("Hola somos la lista normal: ",audios)


# ================================== FUNCIÓN PARA AGREGAR CARACTERÍSTICAS DEL AUDIO A LISTA ===========================

def appenDataAndSr():
    for lesAudios in audios:
        laData, elSr, dataRms, rmsShape, daShape, splitIndex = audioFeatures(
            lesAudios)

        # laPreData1 = librosa.effects.trim(laData)
        # S = librosa.magphase(librosa.stft(laData, window=np.ones, center=False))[0]
        # laPreData2 = librosa.feature.rms(y=laData)

        datas.append(laData)
        posDatas.append(dataRms)
        srs.append(elSr)
        # print("Hola somos shape ",rmsShape)
        # print("Hola somos shape normal ",daShape)

    print("Hola, somos datas: ", datas)
    print("Hola, somos posDatas: ", posDatas)
    print("Hola, somos srs: ", srs)


def audioLen(audioName, theData, theSr):
    durAudio = librosa.get_duration(theData, theSr)
    texto = audioName + " dura: "+str(durAudio)
    audiosLen.append(texto)

# ================================== FUNCIONES PARA MOSTRAR WAVEPLOTS Y ESPECTROGRAMAS ===========================


def saveWaveplots(audioName, theData, theSr):
    # plt.figure(figsize=(10, 4))
    # forPiano = librosa.effects.split(theData, frame_length=100, hop_length=50)
    librosa.display.waveplot(theData, sr=theSr)
    # plt.show()
    filename = 'Images/Waveplots/' + str(audioName) + '.png'
    # files.append(filename)
    plt.savefig(filename)
    plt.clf()

# @jit(target ="cuda")


def saveSpectrograms(audioName, theData, theSr):
    X = librosa.stft(theData)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=theSr, x_axis='time', y_axis='hz')
    # plt.colorbar()
    # plt.show()
    filename = 'Images/Espectrogramas/Hz/' + str(audioName) + '.png'
    plt.savefig(filename)
    plt.clf()
    librosa.display.specshow(Xdb, sr=theSr, x_axis='time', y_axis='log')
    # plt.show()
    filename2 = 'Images/Espectrogramas/Log/' + str(audioName) + '.png'
    plt.savefig(filename2)
    plt.clf()

# @jit(target ="cuda")


def melSpectrograms(audioName, theData, theSr):

    pianosong, trIndex = librosa.effects.trim(theData)

    # forPiano, forTrIndex = librosa.effects.split(pianosong)
    # Transformada de Fourier
    # === Short time fourier transform ===
    n_fft = 2048
    # D = np.abs(librosa.stft(pianosong[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
    # plt.plot(D)
    # plt.show()
    hop_length = 512
    D = np.abs(librosa.stft(pianosong, n_fft=n_fft,  hop_length=hop_length))
    # D = np.abs(librosa.stft(y))**2
    # librosa.display.specshow(D, sr=theSr, x_axis='time', y_axis='linear')
    # plt.colorbar()
    # plt.show()
    # filename1 = 'Images/Mel_Spectograms/y_axis_linear/' + str(audioName) +'.png'
    # plt.savefig(filename1)

    # Espectrograma de Mel
    DB = librosa.amplitude_to_db(D, ref=np.max)

    librosa.display.specshow(
        DB, sr=theSr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    filename2 = 'Espectrogramas/' + str(audioName) + '.png'
    plt.savefig(filename2)
    plt.clf()


def createCsv(audioName, theData, theSr):
    filename = audioName
    y = theData
    sr = theSr
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    file = open('dataset.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


def main():
    # ========== Preparar la librería de audios ==========
    audioDatabase()
    appenDataAndSr()
    listaLimpia()

    # for losAudios, datos, senales in zip(listaAudios, posDatas, srs):
    #     audioLen(losAudios, datos, senales)
    # print("La duracion de los audios es: ")
    # print(audiosLen)

    # for losAudios, datos, senales in zip(listaAudios, posDatas, srs):
    #     saveWaveplots(losAudios, datos, senales)
    # print("Waveplots generados")
    # print("Generando Espectrogramas")
    # for losAudios, datos, senales in zip(listaAudios, posDatas, srs):
    #     saveSpectrograms(losAudios, datos, senales)
    # print("Espectrogramas generados")
    print("Generando Espectrogramas tipo mel")
    for losAudios, datos, senales in zip(listaAudios, posDatas, srs):
        melSpectrograms(losAudios, datos, senales)
    print("Espectrogramas tipo mel generados")

    print("Generando CSV ...")

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('dataset.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    for losAudios, datos, senales in zip(listaAudios, posDatas, srs):
        createCsv(losAudios, datos, senales)


if __name__ == "__main__":
    main()


