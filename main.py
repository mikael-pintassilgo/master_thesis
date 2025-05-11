import sys
import time
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from all_in_one import take_screenshot_on_press_key

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QComboBox,
    QLineEdit, QPushButton, QHBoxLayout, QKeySequenceEdit
)

# Mapping of languages to Hugging Face model names (examples)
LANGUAGE_MODELS = {
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "Chinese": "Helsinki-NLP/opus-mt-en-zh",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Arabic": "Helsinki-NLP/opus-mt-en-ar",
    "Portuguese": "facebook/nllb-200-distilled-600M",
    "Bengali": "Helsinki-NLP/opus-mt-en-bn",
    "Russian": "Helsinki-NLP/opus-mt-en-ru",
    "Japanese": "Helsinki-NLP/opus-mt-en-ja",
    "Punjabi": "Helsinki-NLP/opus-mt-en-pa",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Other...": ""
}

# Import the function from all_in_one.py

class TranslationDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Divertidos Games")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("<h2>Divertidos Games</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Application for translating text in games\nFrom English to other languages")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Add extra space between title and description
        layout.addSpacing(20)

        # Language selection
        lang_label = QLabel("Translate into")
        layout.addWidget(lang_label)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANGUAGE_MODELS.keys())
        self.lang_combo.currentTextChanged.connect(self.update_model_field)
        layout.addWidget(self.lang_combo)

        # Add extra space between title and description
        layout.addSpacing(20)

        # Model field
        model_label = QLabel("Model for translation from")
        layout.addWidget(model_label)
        link_label = QLabel('<a href="https://huggingface.co/">huggingface.co</a>')
        link_label.setOpenExternalLinks(True)
        layout.addWidget(link_label)

        self.model_field = QLineEdit()
        #self.model_field.setReadOnly(True)
        layout.addWidget(self.model_field)
        self.update_model_field(self.lang_combo.currentText())

        # Add extra space between title and description
        layout.addSpacing(20)

        # Key combination selection
        key_label = QLabel("Key combination to perform translation")
        layout.addWidget(key_label)

        self.key_edit = QKeySequenceEdit()
        self.key_edit.setKeySequence(QKeySequence("Ctrl+1"))
        layout.addWidget(self.key_edit)

        # Add extra space between title and description
        layout.addSpacing(8)
        
        # Key combination selection
        key_label = QLabel("Key combination to view previous translation")
        layout.addWidget(key_label)

        self.key_edit = QKeySequenceEdit()
        self.key_edit.setKeySequence(QKeySequence("Ctrl+2"))
        layout.addWidget(self.key_edit)

        # Add extra space between title and description
        layout.addSpacing(20)
        
        # Run button
        self.run_button = QPushButton("Run the translation program")
        self.run_button.clicked.connect(self.run_translation_program)
        layout.addWidget(self.run_button)
                
        self.setLayout(layout)

    def update_model_field(self, language):
        model_name = LANGUAGE_MODELS.get(language, "")
        self.model_field.setText(model_name)

    def run_translation_program(self):
        
        # Add label for "Press Esc to stop the program" (hidden by default)
        # Add label for "Press Ctrl+3 to stop the program" (shown only during run)
        if not hasattr(self, 'esc_label'):
            self.esc_label = QLabel("Press 'Ctrl+3' to stop the program")
            self.esc_label.setAlignment(Qt.AlignCenter)
            self.layout().addWidget(self.esc_label)
        self.esc_label.show()
        
        self.run_button.hide()
        self.repaint()
        self.update()
        
        output_folder = "figures"
        file_name = "img_to_translate.jpg" #png"
        model_name = self.model_field.text()
        print(f"model_name: {model_name}")
        take_screenshot_on_press_key(output_folder, file_name, model_name)
        
        self.run_button.show()
        self.esc_label.hide()
        self.repaint()
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TranslationDialog()
    window.show()
    sys.exit(app.exec_())