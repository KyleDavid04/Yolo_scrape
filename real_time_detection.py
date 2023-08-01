```python
import cv2
from selenium import webdriver
from object_detection_yolo import detect_objects
from ocr_extraction import extract_text
from latex_conversion import convert_to_latex
from csv_append import append_to_csv

# YOLO configuration
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
classes_path = "yolov3.txt"

# Initialize headless browser
options = webdriver.ChromeOptions()
options.add_argument('headless')
browser = webdriver.Chrome(chrome_options=options)

# Capture browser content
def capture_content(url):
    browser.get(url)
    screenshot = browser.get_screenshot_as_png()
    image = cv2.imdecode(np.frombuffer(screenshot, np.uint8), 1)
    return image

# Real-time detection
def real_time_detection(url):
    while True:
        # Capture browser content
        image = capture_content(url)

        # Detect mathematical formulas
        formulas = detect_objects(image, config_path, weights_path, classes_path)

        # Extract text and convert to LaTeX
        for formula in formulas:
            text = extract_text(formula)
            latex = convert_to_latex(text)

            # Append to CSV
            append_to_csv(latex)

# Run real-time detection
real_time_detection("http://www.example.com")
```