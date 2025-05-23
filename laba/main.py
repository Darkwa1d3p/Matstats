import sys
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
from logic import initialize_logic

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    initialize_logic(window)
    window.showMaximized()
    sys.exit(app.exec_())