# Genre Prediction

This project is a Python application that uses machine learning techniques to predict the genre of an audio file. It employs the Mel-Frequency Cepstral Coefficients (MFCC) feature extraction method and the Gaussian Mixture Model (GMM) for classification.

## Features

- Load a pre-trained dataset for genre prediction
- Browse and select an audio file
- Display the predicted genre for the selected audio file
- Play, pause, and stop the selected audio file
- Calculate the prediction score for a specific genre using the entire dataset
- User-friendly GUI built with Tkinter

## Requirements

- Python 3.x
- Numpy
- Scipy
- Pygame
- python_speech_features
- Pillow

## Usage

1. Clone the repository or download the source code.
2. Install the required dependencies.
3. Run the `main.py` file.
4. The application window will open.
5. Click the "Browse" button to select an audio file.
6. The predicted genre will be displayed in the result label.
7. Use the "Play," "Pause," and "Stop" buttons to control audio playback.
8. To calculate the prediction score for a specific genre, select the genre from the dropdown menu and click the "Calculate Prediction Score" button.

## Dataset

The project uses the GTZAN dataset, which contains 1000 audio tracks divided into 10 genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock). Each genre has 100 audio tracks. The dataset is not included in the repository due to its size.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.
