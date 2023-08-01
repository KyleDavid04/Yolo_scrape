```python
import cv2
import pytesseract

def extract_text(image_path):
    # Load the image from file
    img = cv2.imread(image_path, 0)

    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(img)

    return text
```