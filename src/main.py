from PyQt6 import QtWidgets
import sys
from .ui.app import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.resize(720, 520); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()