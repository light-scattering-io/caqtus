# Form implementation generated from reading ui file '.\sequence_widget.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_SequenceWidget(object):
    def setupUi(self, SequenceWidget):
        SequenceWidget.setObjectName("SequenceWidget")
        SequenceWidget.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(SequenceWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(parent=SequenceWidget)
        self.tabWidget.setObjectName("tabWidget")
        self.Constants = QtWidgets.QWidget()
        self.Constants.setObjectName("Constants")
        self.tabWidget.addTab(self.Constants, "")
        self.Iteration = QtWidgets.QWidget()
        self.Iteration.setObjectName("Iteration")
        self.tabWidget.addTab(self.Iteration, "")
        self.Shot = QtWidgets.QWidget()
        self.Shot.setObjectName("Shot")
        self.tabWidget.addTab(self.Shot, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.start_button = QtWidgets.QPushButton(parent=SequenceWidget)
        self.start_button.setObjectName("start_button")
        self.horizontalLayout.addWidget(self.start_button)
        self.interrupt_button = QtWidgets.QPushButton(parent=SequenceWidget)
        self.interrupt_button.setObjectName("interrupt_button")
        self.horizontalLayout.addWidget(self.interrupt_button)
        self.clear_button = QtWidgets.QPushButton(parent=SequenceWidget)
        self.clear_button.setObjectName("clear_button")
        self.horizontalLayout.addWidget(self.clear_button)
        self.progressBar = QtWidgets.QProgressBar(parent=SequenceWidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout.addWidget(self.progressBar)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(SequenceWidget)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(SequenceWidget)

    def retranslateUi(self, SequenceWidget):
        _translate = QtCore.QCoreApplication.translate
        SequenceWidget.setWindowTitle(_translate("SequenceWidget", "Form"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Constants), _translate("SequenceWidget", "Constants"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Iteration), _translate("SequenceWidget", "Iteration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Shot), _translate("SequenceWidget", "Shot"))
        self.start_button.setText(_translate("SequenceWidget", "Start"))
        self.interrupt_button.setText(_translate("SequenceWidget", "Interrupt"))
        self.clear_button.setText(_translate("SequenceWidget", "Clear"))
