# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'create_atom_viewer.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QDialog,
    QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_AtomDialog(object):
    def setupUi(self, AtomDialog):
        if not AtomDialog.objectName():
            AtomDialog.setObjectName(u"AtomDialog")
        AtomDialog.resize(400, 300)
        self.verticalLayout = QVBoxLayout(AtomDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.titleLabel = QLabel(AtomDialog)
        self.titleLabel.setObjectName(u"titleLabel")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.titleLabel)

        self._window_title = QLineEdit(AtomDialog)
        self._window_title.setObjectName(u"_window_title")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self._window_title)

        self.cameraLabel = QLabel(AtomDialog)
        self.cameraLabel.setObjectName(u"cameraLabel")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.cameraLabel)

        self._detector_line_edit = QLineEdit(AtomDialog)
        self._detector_line_edit.setObjectName(u"_detector_line_edit")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self._detector_line_edit)

        self.pictureLabel = QLabel(AtomDialog)
        self.pictureLabel.setObjectName(u"pictureLabel")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.pictureLabel)

        self._picture_line_edit = QLineEdit(AtomDialog)
        self._picture_line_edit.setObjectName(u"_picture_line_edit")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self._picture_line_edit)

        self.axesEqualLabel = QLabel(AtomDialog)
        self.axesEqualLabel.setObjectName(u"axesEqualLabel")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.axesEqualLabel)

        self.axesEqualCheckBox = QCheckBox(AtomDialog)
        self.axesEqualCheckBox.setObjectName(u"axesEqualCheckBox")
        self.axesEqualCheckBox.setChecked(True)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.axesEqualCheckBox)


        self.verticalLayout.addLayout(self.formLayout)

        self.buttonBox = QDialogButtonBox(AtomDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(AtomDialog)
        self.buttonBox.accepted.connect(AtomDialog.accept)
        self.buttonBox.rejected.connect(AtomDialog.reject)

        QMetaObject.connectSlotsByName(AtomDialog)
    # setupUi

    def retranslateUi(self, AtomDialog):
        AtomDialog.setWindowTitle(QCoreApplication.translate("AtomDialog", u"Dialog", None))
        self.titleLabel.setText(QCoreApplication.translate("AtomDialog", u"Title", None))
        self.cameraLabel.setText(QCoreApplication.translate("AtomDialog", u"Detector", None))
        self.pictureLabel.setText(QCoreApplication.translate("AtomDialog", u"Picture", None))
        self.axesEqualLabel.setText(QCoreApplication.translate("AtomDialog", u"Axes equal", None))
    # retranslateUi

