import sys
from PyQt5.QtWidgets import QApplication
from gui import DataAnalyzerGUI

def main():
    app = QApplication(sys.argv)
    mainWin = DataAnalyzerGUI()
    mainWin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()