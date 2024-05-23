 # -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import Qt

from UI.Ui_FaceGrid import Ui_Dialog

if __name__ == '__main__':

    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    
    MainWindow = QDialog()
    ui = Ui_Dialog()
    #ui.var_dflRoot=os.path.abspath(os.path.join(os.getcwd(),"..",".."))

    ui.setupUi(MainWindow)
    
    MainWindow.show()
    #splash.finish(MainWindow)                   # 隐藏启动界面
    sys.exit(app.exec_())

