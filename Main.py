import pygame.mixer
from pygame.mixer import Sound
import time

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator
import math
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk

pygame.mixer.init()

dataset = []

def loadDataset(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    print("Dataset loaded.")

loadDataset("my.dat")

def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def predict_genre(audio_file):
    try:
        (rate, sig) = wav.read(audio_file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, 0)
        pred = nearestClass(getNeighbors(dataset, feature, 5))
        return results[pred]
    except ValueError:
        print(f"Error reading file: {audio_file}")
        return None

results = defaultdict(int)
i = 1
for folder in os.listdir("GTZAN\\genres_original\\"):
    results[i] = folder
    i += 1

print("Genres:", results)


root = tk.Tk()
root.title("Genre Prediction")
root.geometry("600x600")
bg_color = "#DCDCDC"

label_bg_color = bg_color

root.configure(bg=bg_color)

selected_file = tk.StringVar()
selected_genre = tk.StringVar()
selected_genre.set("Select Genre")

audio_file_path = None

def browse_file():
    global audio_file_path
    audio_file_path = filedialog.askopenfilename(initialdir="./", title="Select an audio file")
    if audio_file_path:
        genre = predict_genre(audio_file_path)
        selected_file.set(os.path.basename(audio_file_path))  # Set only the file name
        # Load the selected audio file
        pygame.mixer.music.load(audio_file_path)
        result_label.config(text="", fg="#333333")
        result_label.config(text=f"Predicted genre: {genre}", fg="green")

def play_audio():
    if audio_file_path:
        pygame.mixer.music.play()

def pause_audio():
    if audio_file_path:
        pygame.mixer.music.pause()

def stop_audio():
    if audio_file_path:
        pygame.mixer.music.stop()

def calculate_score():
    genre_folder = selected_genre.get()
    if genre_folder != "Select Genre":
        predictions = []
        total_files = 0
        processed_files = 0
        correct_predictions = 0
        genre_path = os.path.join("GTZAN\\genres_original\\", genre_folder)
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith(".wav"):
                total_files += 1
        print(f"Processing genre: {genre_folder}")
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith(".wav"):
                audio_file_path = os.path.join(genre_path, audio_file)
                prediction = predict_genre(audio_file_path)
                if prediction is not None:
                    predictions.append((audio_file, prediction))
                    if genre_folder in prediction:
                        correct_predictions += 1
                processed_files += 1
                print(f"Processed {processed_files}/{total_files} files")

        score = (correct_predictions / len(predictions)) * 100
        result_label.config(text=f"Prediction score for '{genre_folder}' genre: {score:.2f}%", fg="green")
    else:
        result_label.config(text="Please select a genre.", fg="red")

def predict():
    if audio_file_path:
        genre = predict_genre(audio_file_path)
        if genre is not None:
            result_label.config(text=f"Predicted genre: {genre}", fg="green")
        else:
            result_label.config(text="Error reading the audio file.", fg="red")
    else:
        result_label.config(text="Please select an audio file first.", fg="red")

title_label = tk.Label(root, text="Genre Prediction", font=("Arial", 24, "bold"), bg=bg_color, fg="#333333")
title_label.pack(pady=40)

file_frame = tk.Frame(root, bg=bg_color)
file_frame.pack(pady=10)

file_label = tk.Label(file_frame, text="Selected file:", font=("Arial", 12), bg=bg_color, fg="#333333")
file_label.grid(row=0, column=0, padx=10, pady=5)

file_entry = tk.Entry(file_frame, textvariable=selected_file, width=40, font=("Arial", 12))
file_entry.grid(row=0, column=1)

browse_button = tk.Button(file_frame, text="Browse", command=browse_file, font=("Arial", 12), bg="#4CAF50", fg="white")
browse_button.grid(row=0, column=2, padx=10)

genre_label = tk.Label(file_frame, text="Select Genre:", font=("Arial", 12), bg=bg_color, fg="#333333")
genre_label.grid(row=1, column=0, padx=10, pady=5)

genre_options = ["Select Genre", "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
genre_menu = tk.OptionMenu(file_frame, selected_genre, *genre_options)
genre_menu.grid(row=1, column=1, padx=10,pady=10)

score_button = tk.Button(root, text="Calculate Prediction Score", command=calculate_score, font=("Arial", 14, "bold"), bg="#9C27B0", fg="white", padx=20, pady=10)
score_button.pack()
result_label = tk.Label(root, text="", font=("Arial", 14), bg=bg_color, fg="#333333")
result_label.pack(pady=(20,0))
play_button = tk.Button(root, text="Play", command=play_audio, font=("Arial", 14, "bold"), bg="#2196F3", fg="white", padx=20, pady=10)
play_button.pack(side=tk.LEFT, padx=(120,0))

pause_button = tk.Button(root, text="Pause", command=pause_audio, font=("Arial", 14, "bold"), bg="#FFC107", fg="white", padx=20, pady=10)
pause_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(root, text="Stop", command=stop_audio, font=("Arial", 14, "bold"), bg="#F44336", fg="white", padx=20, pady=10)
stop_button.pack(side=tk.LEFT, padx=10)



root.mainloop()
