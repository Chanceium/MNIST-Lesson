# MNIST Digit Recognizer

A beautiful, mobile-centric Flask web application for recognizing handwritten digits using three different MNIST models.

**Made by Chance Page**

## Features

- 🎨 **Apple-like smooth UI** with pastel blue and yellow theme
- 📱 **Mobile-first design** optimized for touch devices
- ✍️ **28x28 drawable canvas** with auto-prediction on pen lift
- 🤖 **Three model predictions** shown simultaneously:
  - Baseline Model
  - Augmented Model
  - Overfitted Model
- 🖥️ **CPU-optimized** for easy deployment on any server
- 🐳 **Docker ready** for containerized deployment

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python app.py
```

3. Open your browser to `http://localhost:5000`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t mnist-flask .
```

2. Run the container:
```bash
docker run -p 5000:5000 mnist-flask
```

3. Access the app at `http://localhost:5000`

## Usage

1. Draw a digit (0-9) on the canvas using your mouse or finger
2. Release your pen/finger to automatically trigger prediction
3. View predictions from all three models with confidence scores
4. See detailed probability distributions for each digit
5. Clear the canvas to try again

## Technical Details

- **Backend**: Flask with TensorFlow (CPU-only)
- **Frontend**: Vanilla JavaScript with smooth animations
- **Models**: Three Keras models (.h5 format)
- **Image Processing**: PIL for preprocessing
- **Production Server**: Gunicorn

## Project Structure

```
mnist-flask/
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Main UI
├── models/
│   ├── baseline_model.h5
│   ├── augmented_model.h5
│   └── overfitted_model.h5
├── requirements.txt
├── Dockerfile
└── README.md
```
