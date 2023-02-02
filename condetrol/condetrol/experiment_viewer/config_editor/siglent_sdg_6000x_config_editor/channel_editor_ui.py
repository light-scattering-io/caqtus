# Form implementation generated from reading ui file '.\resources\channel_editor.ui'
#
# Created by: PyQt6 UI code generator 6.4.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ChannelEditor(object):
    def setupUi(self, ChannelEditor):
        ChannelEditor.setObjectName("ChannelEditor")
        ChannelEditor.resize(400, 224)
        self.verticalLayout = QtWidgets.QVBoxLayout(ChannelEditor)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.parameters_layout = QtWidgets.QGridLayout()
        self.parameters_layout.setObjectName("parameters_layout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.on_button = QtWidgets.QRadioButton(parent=ChannelEditor)
        self.on_button.setObjectName("on_button")
        self.horizontalLayout_3.addWidget(self.on_button)
        self.off_button = QtWidgets.QRadioButton(parent=ChannelEditor)
        self.off_button.setObjectName("off_button")
        self.horizontalLayout_3.addWidget(self.off_button)
        self.parameters_layout.addLayout(self.horizontalLayout_3, 0, 1, 1, 1)
        self.modulation_combobox = QtWidgets.QComboBox(parent=ChannelEditor)
        self.modulation_combobox.setObjectName("modulation_combobox")
        self.parameters_layout.addWidget(self.modulation_combobox, 3, 1, 1, 1)
        self.modulation_label = QtWidgets.QLabel(parent=ChannelEditor)
        self.modulation_label.setObjectName("modulation_label")
        self.parameters_layout.addWidget(self.modulation_label, 3, 0, 1, 1)
        self.waveform_label = QtWidgets.QLabel(parent=ChannelEditor)
        self.waveform_label.setObjectName("waveform_label")
        self.parameters_layout.addWidget(self.waveform_label, 2, 0, 1, 1)
        self.waveform_combobox = QtWidgets.QComboBox(parent=ChannelEditor)
        self.waveform_combobox.setObjectName("waveform_combobox")
        self.parameters_layout.addWidget(self.waveform_combobox, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(parent=ChannelEditor)
        self.label.setObjectName("label")
        self.parameters_layout.addWidget(self.label, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=ChannelEditor)
        self.label_4.setObjectName("label_4")
        self.parameters_layout.addWidget(self.label_4, 1, 0, 1, 1)
        self.output_load_combobox = QtWidgets.QComboBox(parent=ChannelEditor)
        self.output_load_combobox.setObjectName("output_load_combobox")
        self.parameters_layout.addWidget(self.output_load_combobox, 1, 1, 1, 1)
        self.horizontalLayout_2.addLayout(self.parameters_layout)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem1)

        self.retranslateUi(ChannelEditor)
        QtCore.QMetaObject.connectSlotsByName(ChannelEditor)

    def retranslateUi(self, ChannelEditor):
        _translate = QtCore.QCoreApplication.translate
        ChannelEditor.setWindowTitle(_translate("ChannelEditor", "Form"))
        self.on_button.setText(_translate("ChannelEditor", "On"))
        self.off_button.setText(_translate("ChannelEditor", "Off"))
        self.modulation_label.setText(_translate("ChannelEditor", "Modulation"))
        self.waveform_label.setText(_translate("ChannelEditor", "Waveform"))
        self.label.setText(_translate("ChannelEditor", "Enabled:"))
        self.label_4.setText(_translate("ChannelEditor", "Output load:"))
