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