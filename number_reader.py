import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QHBoxLayout
)
from PyQt5.QtCore import Qt
import numpy as np
import os

path = 'results/recent/nail_indices.txt'

with open(path) as f:
    numbers = [int(x.strip()) for x in f.read().split(',')]
    
class NumberReader(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Number Reader")
        
        # Load saved index if exists
        if os.path.exists("final_index.npy"):
            self.index = int(np.load("final_index.npy"))
        else:
            self.index = 0

        # Layouts
        main_layout = QVBoxLayout()
        nav_layout = QHBoxLayout()

        # Main numbers display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("font-size: 24px;")
        
        # Progress text
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 12px;")

        # Buttons
        self.forward_button = QPushButton("🔊 Forward (spacebar)")
        self.back_button = QPushButton("⏮ Back ('a')")

        # Add widgets
        main_layout.addWidget(self.display_label)
        main_layout.addWidget(self.progress_label)
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.forward_button)
        main_layout.addLayout(nav_layout)

        self.setLayout(main_layout)

        # Connect buttons
        self.back_button.clicked.connect(self.go_back)
        self.forward_button.clicked.connect(self.go_forward)

        # Show initial state
        self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.go_forward()
        if event.key() == Qt.Key_A:
            self.go_back()

    def go_back(self):
        self.index = max(self.index - 1, 0)
        self.update_display()

    def go_forward(self):
        self.index = min(self.index + 1, len(numbers) - 1)
        self.update_display()

    def update_display(self):
        start = max(self.index - 4, 0)
        end = min(self.index + 5, len(numbers)) 
        window = numbers[start:end]

        # Pad sides if near edges
        while len(window) < 9:
            if start > 0:
                start -= 1
                window = [numbers[start]] + window
            elif end < len(numbers):
                window.append(numbers[end])
                end += 1
            else:
                break

        display_text = "   ".join(
            f"<span style='font-weight:bold; font-size:36px;'>[{n}]</span>" if i + start == self.index else f"<span>{n}</span>"
            for i, n in enumerate(window)
        )
        self.display_label.setText(display_text)
        
        progress_text = f'{self.index}/{len(numbers)}      {self.index/len(numbers)*100:.1f} %'
        self.progress_label.setText(progress_text)
        
    def closeEvent(self, event):
        np.save("final_index.npy", self.index)
        print(f"Saved final index: {self.index}")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NumberReader()
    window.resize(1000, 400)
    window.show()
    sys.exit(app.exec_())