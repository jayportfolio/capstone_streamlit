from PIL import Image
from pytesseract import pytesseract
import numpy as np

# Defining paths to tesseract.exe
# and the image we would be using
path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image_path = r"csv\sample_text.png"
image_path = r"22260_BAR210108_FLP_00_0000.jpeg"


filename = 'image_01.png'
filename = image_path
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1)

print(text)