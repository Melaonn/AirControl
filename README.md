# ✋ Virtual Gesture-Based Drawing App

A **real-time virtual drawing application** that uses **hand gestures** detected through the webcam to draw on a digital canvas. Built using **PyQt5**, **OpenCV**, and **MediaPipe**, this project enables touchless interaction for artistic expression, presentations, or accessibility-focused UIs.


## 🚀 Features

- 🖐️ **Gesture Recognition with MediaPipe**
  - Track hand landmarks in real-time
  - Recognize gestures like "index-thumb pinch" to draw
  - Detect "fist" gesture to clear the canvas

- 🎨 **Virtual Drawing Canvas**
  - Draw using hand gestures without a physical stylus
  - Custom brush colors using color picker
  - Adjustable brush size with slider

- 🔄 **Undo / Redo Functionality**
  - Implemented using Python's `deque` for efficient drawing history

- 🧠 **Smooth UI with PyQt5**
  - Responsive GUI with buttons, sliders, toolbars, and dialogs
  - Multithreaded video processing for performance

- 💾 **Image Handling**
  - Save your drawings as images
  - Import background images for annotation

---

## 🧰 Tech Stack

| Component           | Technology             |
|--------------------|------------------------|
| Programming Language | Python 3.x            |
| GUI Framework        | PyQt5                 |
| Video Processing     | OpenCV                |
| Gesture Detection    | MediaPipe (Google)    |
| Math/Arrays          | NumPy                 |
| Data Structure       | collections.deque     |
| Multithreading       | QThread, QTimer       |

---

## 🧑‍💻 How It Works

1. Start the script – webcam feed is displayed.
2. MediaPipe tracks your hand in real-time.
3. Perform the "pinch" gesture (index + thumb) to draw on the canvas.
4. Use UI controls to:
   - Change color and brush size
   - Save or clear the canvas
   - Undo/redo your strokes

---

## 📦 Installation

### Prerequisites

Make sure Python 3.7+ is installed.
install all the packages and requires a gpu for better performance 
```bash
pip install pyqt5 opencv-python mediapipe numpy
