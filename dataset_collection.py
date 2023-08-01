```python
import os
import requests
from bs4 import BeautifulSoup

# Define the URL of the website from which we will scrape images
url = "http://www.website.com"

# Define the directory where we will save the images
image_dir = "math_formula_images"

# Create the directory if it doesn't exist
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Send a GET request to the website
response = requests.get(url)

# Parse the content of the request with BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all img tags in the HTML
img_tags = soup.find_all("img")

# Define a counter for the image filenames
counter = 0

# Loop over the img tags
for img in img_tags:
    # Try to download the image
    try:
        # Get the URL of the image
        img_url = img['src']

        # Send a GET request to the image URL
        img_response = requests.get(img_url, stream=True)

        # If the GET request is successful, save the image
        if img_response.status_code == 200:
            # Open a file in write-binary mode
            with open(os.path.join(image_dir, f"image_{counter}.jpg"), 'wb') as f:
                # Write the content of the image to the file
                f.write(img_response.content)

            # Increment the counter
            counter += 1

    # If an error occurs, print a message and continue with the next image
    except Exception as e:
        print(f"Failed to download image {img_url}: {e}")
```