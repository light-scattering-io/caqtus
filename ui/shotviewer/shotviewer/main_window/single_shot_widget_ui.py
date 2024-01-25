# Form implementation generated from reading ui file '.\single_shot_widget.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_SingleShotWidget(object):
    def setupUi(self, SingleShotWidget):
        SingleShotWidget.setObjectName("SingleShotWidget")
        SingleShotWidget.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(parent=SingleShotWidget)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self._mdi_area = QtWidgets.QMdiArea(parent=self.centralwidget)
        self._mdi_area.setObjectName("_mdi_area")
        self.verticalLayout.addWidget(self._mdi_area)
        SingleShotWidget.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(parent=SingleShotWidget)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuWindow = QtWidgets.QMenu(parent=self.menuBar)
        self.menuWindow.setObjectName("menuWindow")
        self.menu_add_viewer = QtWidgets.QMenu(parent=self.menuBar)
        self.menu_add_viewer.setObjectName("menu_add_viewer")
        self.menuWorkspace = QtWidgets.QMenu(parent=self.menuBar)
        self.menuWorkspace.setObjectName("menuWorkspace")
        SingleShotWidget.setMenuBar(self.menuBar)
        self._shot_selector_dock = QtWidgets.QDockWidget(parent=SingleShotWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self._shot_selector_dock.sizePolicy().hasHeightForWidth())
        self._shot_selector_dock.setSizePolicy(sizePolicy)
        self._shot_selector_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self._shot_selector_dock.setObjectName("_shot_selector_dock")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self._shot_selector_dock.setWidget(self.dockWidgetContents)
        SingleShotWidget.addDockWidget(QtCore.Qt.DockWidgetArea(4), self._shot_selector_dock)
        self._sequence_hierarchy_dock = QtWidgets.QDockWidget(parent=SingleShotWidget)
        self._sequence_hierarchy_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self._sequence_hierarchy_dock.setObjectName("_sequence_hierarchy_dock")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self._sequence_hierarchy_view = QtWidgets.QTreeView(parent=self.dockWidgetContents_2)
        self._sequence_hierarchy_view.setObjectName("_sequence_hierarchy_view")
        self.verticalLayout_2.addWidget(self._sequence_hierarchy_view)
        self._sequence_hierarchy_dock.setWidget(self.dockWidgetContents_2)
        SingleShotWidget.addDockWidget(QtCore.Qt.DockWidgetArea(1), self._sequence_hierarchy_dock)
        self._action_cascade = QtGui.QAction(parent=SingleShotWidget)
        self._action_cascade.setObjectName("_action_cascade")
        self._action_tile = QtGui.QAction(parent=SingleShotWidget)
        self._action_tile.setObjectName("_action_tile")
        self.actionImage = QtGui.QAction(parent=SingleShotWidget)
        self.actionImage.setObjectName("actionImage")
        self.actionParameters = QtGui.QAction(parent=SingleShotWidget)
        self.actionParameters.setObjectName("actionParameters")
        self.actionAtoms = QtGui.QAction(parent=SingleShotWidget)
        self.actionAtoms.setObjectName("actionAtoms")
        self.actionSave = QtGui.QAction(parent=SingleShotWidget)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtGui.QAction(parent=SingleShotWidget)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionLoad = QtGui.QAction(parent=SingleShotWidget)
        self.actionLoad.setObjectName("actionLoad")
        self.menuWindow.addAction(self._action_cascade)
        self.menuWindow.addAction(self._action_tile)
        self.menuWorkspace.addAction(self.actionSave_as)
        self.menuWorkspace.addAction(self.actionLoad)
        self.menuBar.addAction(self.menuWorkspace.menuAction())
        self.menuBar.addAction(self.menu_add_viewer.menuAction())
        self.menuBar.addAction(self.menuWindow.menuAction())

        self.retranslateUi(SingleShotWidget)
        QtCore.QMetaObject.connectSlotsByName(SingleShotWidget)

    def retranslateUi(self, SingleShotWidget):
        _translate = QtCore.QCoreApplication.translate
        SingleShotWidget.setWindowTitle(_translate("SingleShotWidget", "MainWindow"))
        self.menuWindow.setTitle(_translate("SingleShotWidget", "Windows"))
        self.menu_add_viewer.setTitle(_translate("SingleShotWidget", "Add viewer"))
        self.menuWorkspace.setTitle(_translate("SingleShotWidget", "Workspace"))
        self._action_cascade.setText(_translate("SingleShotWidget", "Cascade"))
        self._action_tile.setText(_translate("SingleShotWidget", "Tile"))
        self.actionImage.setText(_translate("SingleShotWidget", "Image"))
        self.actionParameters.setText(_translate("SingleShotWidget", "Parameters"))
        self.actionAtoms.setText(_translate("SingleShotWidget", "Atoms"))
        self.actionSave.setText(_translate("SingleShotWidget", "Save"))
        self.actionSave_as.setText(_translate("SingleShotWidget", "Save as"))
        self.actionLoad.setText(_translate("SingleShotWidget", "Load"))
