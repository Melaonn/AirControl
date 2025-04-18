from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap, QColor, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout,QLabel, QPushButton, QSlider, QFileDialog, QColorDialog, QDialog, QVBoxLayout, QComboBox, QDialogButtonBox, QMenu, QMenuBar, QAction, QWidgetAction, QHBoxLayout, QWidget, QToolBar
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
import sys

class VideoThread(QThread):
    frameCaptured = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.running = True

    def run(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                self.frameCaptured.emit(frame)
            QThread.msleep(16)  # 60fps

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()
        self.wait()

    def set_camera_index(self, camera_index):
        self.camera_index = camera_index
        self.cap.open(self.camera_index)

class CameraSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera")
        self.setGeometry(100, 100, 300, 100)

        layout = QVBoxLayout()

        self.combo_box = QComboBox(self)
        self.populate_combo_box()
        layout.addWidget(self.combo_box)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def populate_combo_box(self):
        self.combo_box.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            self.combo_box.addItem(f"Camera {index}")
            cap.release()
            index += 1

    def get_selected_camera(self):
        return self.combo_box.currentIndex()

class HandGestureSketchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Sketch Application")
        self.setGeometry(10, 100, 1920, 1080)
        
        self.brush_color = QColor(Qt.black)
        self.brush_size = 8
        self.brush_shape = Qt.RoundCap
        self.brush_active = False
        
        self.initUI()
        self.initMediaPipe()
        
        self.previous_position = None
        self.positions_deque = deque(maxlen=3)
        
        self.video_thread = VideoThread()
        self.video_thread.frameCaptured.connect(self.process_frame)
        self.video_thread.start()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS
        
        self.scaling_factor = 2  
        self.actions = []
        self.redo_stack = []

    def initUI(self):
        #sketchpad
        self.canvas = QLabel(self)
        self.canvas.setGeometry(0, 0, 1280, 720)
        self.canvas.setPixmap(QPixmap(1280, 720))
        #self.showFullScreen()
        self.canvas.pixmap().fill(Qt.white)
        
        self.cursor_label = QLabel(self)
        self.cursor_label.setGeometry(0, 0, 1280, 720)
        self.cursor_label.setStyleSheet("background:transparent;")
        self.cursor_label.setPixmap(QPixmap(1280, 720))
        self.cursor_label.pixmap().fill(Qt.transparent)
        
        self.video_label = QLabel(self)
        self.video_label.setGeometry(1300, 130, 640, 360)
        
        self.initToolBar()
        self.initMenuBar()
        #self.initShapeButtons()

    def initMenuBar(self):
        """Initialize the menu bar."""
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(
"""
QMenuBar
{
    background-color: #cbf2f0;
    color: #050505;
}
QMenuBar::item
{
    background-color: #cbf2f0;
    color: #050505;
}
QMenuBar::item::selected
{
    background-color: #768c8b;
    color: #050505;
}
QMenu
{
    background-color: #768c8b;
    color: #050505;
}
QMenu::item::selected
{
    background-color: #cbf2f0;
    color: #050505;
}
 """
)
        file_menu = menu_bar.addMenu('File')
        
        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_drawing)
        file_menu.addAction(save_action)
        
        load_action = QAction('Open', self)
        load_action.triggered.connect(self.load_drawing)
        file_menu.addAction(load_action)
        
        edit_menu = menu_bar.addMenu('Edit')
        
        undo_action = QAction('Undo', self)
        undo_action.triggered.connect(self.undo_last_action)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction('Redo', self)
        redo_action.triggered.connect(self.redo_last_action)
        edit_menu.addAction(redo_action)
        
        clear_action = QAction('Clear', self)
        clear_action.triggered.connect(self.clear_canvas)
        edit_menu.addAction(clear_action)
        
        camera_menu = menu_bar.addMenu('Camera')
        
        select_camera_action = QAction('Select Camera', self)
        select_camera_action.triggered.connect(self.select_camera)
        camera_menu.addAction(select_camera_action)
        
        brush_menu = menu_bar.addMenu('Brush Size')
        
        brush_size_slider_action = QWidgetAction(self)
        brush_size_slider = QSlider(Qt.Horizontal, self)
        brush_size_slider.setMinimum(1)
        brush_size_slider.setMaximum(20)
        brush_size_slider.setValue(self.brush_size)
        brush_size_slider.valueChanged.connect(self.change_brush_size)
        brush_size_slider_action.setDefaultWidget(brush_size_slider)
        brush_menu.addAction(brush_size_slider_action)

        colour_menu = menu_bar.addMenu('Colour')
        color_palette_widget = QWidget()
        color_palette_layout = QHBoxLayout(color_palette_widget)
        
        colors = [
            ('Black', QColor(Qt.black)),
            ('Red', QColor(Qt.red)),
            ('Green', QColor(Qt.green)),
            ('Blue', QColor(Qt.blue)),
            ('Yellow', QColor(Qt.yellow)),
            ('Cyan', QColor(Qt.cyan)),
            ('Magenta', QColor(Qt.magenta)),
            ('Gray', QColor(Qt.gray)),
        ]
        
        for name, color in colors:
            color_button = QPushButton(self)
            color_button.setStyleSheet(f"background-color: {color.name()}; width: 24px; height: 24px;")
            color_button.clicked.connect(lambda checked, col=color: self.set_brush_color(col))
            color_palette_layout.addWidget(color_button)
        
        colour_menu_action = QWidgetAction(self)
        colour_menu_action.setDefaultWidget(color_palette_widget)
        colour_menu.addAction(colour_menu_action)
        
        # Custom color picker
        colour_change_action = QAction('Custom', self)
        colour_change_action.triggered.connect(self.open_color_dialog)
        colour_menu.addAction(colour_change_action)

        brush_shape_menu = menu_bar.addMenu('Brush shape')
        round_shape_action = QAction('Round',self)
        round_shape_action.triggered.connect(self.set_round_brush_shape)
        brush_shape_menu.addAction(round_shape_action)

        square_shape_action = QAction('Square',self)
        square_shape_action.triggered.connect(self.set_square_brush_shape)
        brush_shape_menu.addAction(square_shape_action)
    def initToolBar(self):
        """Initialize the toolbar with basic color buttons and a custom color picker."""
        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        toolbar.setStyleSheet("""
        QToolBar {
            min-height: 100px;
            background-color: #cbf2f0;
        }
        """)

        color_palette_widget = QWidget()
        color_palette_layout = QGridLayout(color_palette_widget)
        
        colors = [
            ('Black', [QColor(0, 0, 0), QColor(50, 50, 50), QColor(100, 100, 100)]),
            ('Red', [QColor(255, 0, 0), QColor(255, 102, 102), QColor(255, 204, 204)]),
            ('Green', [QColor(0, 255, 0), QColor(102, 255, 102), QColor(204, 255, 204)]),
            ('Blue', [QColor(0, 0, 255), QColor(102, 102, 255), QColor(204, 204, 255)]),
            ('Yellow', [QColor(255, 255, 0), QColor(255, 255, 102), QColor(255, 255, 204)]),
            ('Cyan', [QColor(0, 255, 255), QColor(102, 255, 255), QColor(204, 255, 255)]),
            ('Magenta', [QColor(255, 0, 255), QColor(255, 102, 255), QColor(255, 204, 255)]),
            ('Gray', [QColor(128, 128, 128), QColor(169, 169, 169), QColor(211, 211, 211)]),
            ('Orange', [QColor(255, 165, 0), QColor(255, 191, 102), QColor(255, 218, 204)]),
            ('Purple', [QColor(128, 0, 128), QColor(191, 102, 191), QColor(218, 204, 218)]),
            ('Pink', [QColor(255, 192, 203), QColor(255, 218, 230), QColor(255, 240, 246)]),
            ('Brown', [QColor(165, 42, 42), QColor(186, 85, 85), QColor(205, 133, 133)]),
            ('White', [QColor(255, 255, 255), QColor(245, 245, 245), QColor(235, 235, 235)]),
            ('Dark Green', [QColor(0, 100, 0), QColor(34, 139, 34), QColor(46, 139, 87)]),
        ]

        total_shades = sum(len(shades) for _, shades in colors)
        rows = 3
        cols = (total_shades + rows - 1) // rows  # Calculate number of columns needed for 3 rows

        row, col = 0, 0
        for name, shades in colors:
            for shade in shades:
                color_button = QPushButton(self)
                color_button.setStyleSheet(f"background-color: {shade.name()}; width: 24px; height: 24px;")
                color_button.clicked.connect(lambda checked, col=shade: self.set_brush_color(col))
                color_palette_layout.addWidget(color_button, row, col)
                col += 1
                if col == cols:
                    col = 0
                    row += 1
                    if row == rows:
                        row = 0  # Reset row to 0 if it exceeds the number of rows

        colour_menu_action = QWidgetAction(self)
        colour_menu_action.setDefaultWidget(color_palette_widget)
        toolbar.addAction(colour_menu_action)
        
        # Custom color picker
        colour_change_action = QAction('Custom', self)
        colour_change_action.triggered.connect(self.open_color_dialog)
        toolbar.addAction(colour_change_action)


    def set_round_brush_shape(self):
        """Set the brush shape to round."""
        self.brush_shape = Qt.RoundCap
        
    def set_square_brush_shape(self):
        """Set the brush shape to square."""
        self.brush_shape = Qt.SquareCap
            

    def initMediaPipe(self):
        """Initialize MediaPipe components for hand tracking."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils

    def set_brush_color(self, color):
        """Set the brush color."""
        self.brush_color = color

    def open_color_dialog(self):
        """Open a color dialog to select a custom color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.set_brush_color(color)

    def set_brush_shape(self, shape):
        """Set the brush shape."""
        self.brush_shape = shape

    def change_brush_size(self, value):
        """Change the brush size."""
        self.brush_size = value

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.pixmap().fill(Qt.white)
        self.canvas.update()
        self.actions.clear()  # Clear the actions stack
        self.redo_stack.clear()  # Clear the redo stack

    def undo_last_action(self):
        """Undo the last action."""
        if self.actions:
            action = self.actions.pop()
            self.redo_stack.append(action)
            self.redraw_canvas()

    def redo_last_action(self):
        """Redo the last undone action."""
        if self.redo_stack:
            action = self.redo_stack.pop()
            self.actions.append(action)
            self.redraw_canvas()

    def redraw_canvas(self):
        """Redraw the canvas based on the actions stack."""
        self.canvas.pixmap().fill(Qt.white)
        painter = QPainter(self.canvas.pixmap())
        for action in self.actions:
            pen = QPen(action['color'], action['size'], Qt.SolidLine, action.get('shape', Qt.RoundCap), Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(action['start'], action['end'])
        painter.end()
        self.canvas.update()

    def save_drawing(self):
        """Save the current drawing to a file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(.png);;JPEG(.jpg .jpeg);;All Files(.*) ")
        if file_path:
            self.canvas.pixmap().save(file_path)

    def load_drawing(self):
        """Load a drawing from a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "PNG(.png);;JPEG(.jpg .jpeg);;All Files(.*) ")
        if file_path:
            self.canvas.setPixmap(QPixmap(file_path))
            self.actions.clear()
            self.redo_stack.clear()

    def process_frame(self, frame):
        """Process a frame from the video feed."""
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                try:
                    index_finger_tip = (int(lm[8].x * 640), int(lm[8].y * 360))
                    index_finger_tip = (index_finger_tip[0] * self.scaling_factor, index_finger_tip[1] * self.scaling_factor)
                    self.update_cursor(index_finger_tip[0], index_finger_tip[1])
                    if self.is_only_index_finger_extended(lm):
                        self.brush_active = True
                        self.paint(index_finger_tip[0], index_finger_tip[1])
                    else:
                        self.brush_active = False
                        self.previous_position = None
                    if self.is_only_pinky_finger_extended(lm):
                        self.clear_canvas()
                except Exception as e:
                    print(f"Error processing landmarks: {e}")
        img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(pix)

    def is_only_index_finger_extended(self, landmarks):
        """Check if only the index finger is extended."""
        try:
            index_tip = np.array([landmarks[8].x, landmarks[8].y])
            index_pip = np.array([landmarks[6].x, landmarks[6].y])
            index_extended = index_tip[1] < index_pip[1]
            other_fingers = [(landmarks[12], landmarks[10]), (landmarks[16], landmarks[14]), (landmarks[20], landmarks[18])]
            others_folded = all(tip.y > pip.y for tip, pip in other_fingers)
            return index_extended and others_folded
        except Exception as e:
            print(f"Error checking only index finger extension: {e}")
            return False

    def is_only_pinky_finger_extended(self, landmarks):
        """Check if only the pinky finger is extended."""
        try:
            pinky_tip = np.array([landmarks[20].x, landmarks[20].y])
            pinky_pip = np.array([landmarks[18].x, landmarks[18].y])
            pinky_extended = pinky_tip[1] < pinky_pip[1]
            other_fingers = [(landmarks[8], landmarks[6]), (landmarks[12], landmarks[10]), (landmarks[16], landmarks[14])]
            others_folded = all(tip.y > pip.y for tip, pip in other_fingers)
            return pinky_extended and others_folded
        except Exception as e:
            print(f"Error checking only pinky finger extension: {e}")
            return False

    def paint(self, x, y):
        """Paint on the canvas based on brush settings."""
        if self.brush_active:
            smoothed_position = self.smooth_position((x, y))
            if self.previous_position:
                x1, y1 = self.previous_position
                painter = QPainter(self.canvas.pixmap())
                pen = QPen(self.brush_color, self.brush_size, Qt.SolidLine, self.brush_shape, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(int(x1), int(y1), int(smoothed_position[0]), int(smoothed_position[1]))
                painter.end()
                self.canvas.update()
                self.actions.append({'color': self.brush_color, 'size': self.brush_size, 'start': QPoint(int(x1), int(y1)), 'end': QPoint(int(smoothed_position[0]), int(smoothed_position[1])), 'shape': self.brush_shape})
            self.previous_position = smoothed_position

    def smooth_position(self, position):
        """Smooth the position of the brush using a moving average."""
        self.positions_deque.append(position)
        avg_x = sum([pos[0] for pos in self.positions_deque]) / len(self.positions_deque)
        avg_y = sum([pos[1] for pos in self.positions_deque]) / len(self.positions_deque)
        return (avg_x, avg_y)

    def update_cursor(self, x, y):
        """Update the cursor position on the canvas."""
        cursor_size = 10
        smoothed_position = self.smooth_position((x, y))
        self.cursor_label.pixmap().fill(Qt.transparent)
        painter = QPainter(self.cursor_label.pixmap())
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawEllipse(QPoint(int(smoothed_position[0]), int(smoothed_position[1])), cursor_size // 2, cursor_size // 2)
        painter.end()
        self.cursor_label.update()

    def select_camera(self):
        """Open a dialog to select the camera."""
        dialog = CameraSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            camera_index = dialog.get_selected_camera()
            self.video_thread.stop()
            self.video_thread = VideoThread(camera_index)
            self.video_thread.frameCaptured.connect(self.process_frame)
            self.video_thread.start()

    def closeEvent(self, event):
        """Handle the close event to stop the video thread."""
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = HandGestureSketchApp()
    main_win.show()
    sys.exit(app.exec_())