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
        SequenceWidget.resize(424, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(SequenceWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(parent=SequenceWidget)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self.tabWidget.setObjectName("tabWidget")
        self.Constants = QtWidgets.QWidget()
        self.Constants.setObjectName("Constants")
        self.tabWidget.addTab(self.Constants, "")
        self.iteration_tab = QtWidgets.QWidget()
        self.iteration_tab.setObjectName("iteration_tab")
        self.tabWidget.addTab(self.iteration_tab, "")
        self.Timelanes = QtWidgets.QWidget()
        self.Timelanes.setObjectName("Timelanes")
        self.tabWidget.addTab(self.Timelanes, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(SequenceWidget)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(SequenceWidget)

    def retranslateUi(self, SequenceWidget):
        _translate = QtCore.QCoreApplication.translate
        SequenceWidget.setWindowTitle(_translate("SequenceWidget", "Form"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Constants), _translate("SequenceWidget", "Constants"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.iteration_tab), _translate("SequenceWidget", "Iteration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Timelanes), _translate("SequenceWidget", "Shot"))
