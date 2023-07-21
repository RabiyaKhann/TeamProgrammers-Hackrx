from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import cv2

def capture_screenshot(url, output_path):
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    
    # Initialize the Chrome WebDriver with the specified options
    driver = webdriver.Chrome(executable_path='path/to/chromedriver', options=chrome_options)
    
    try:
        # Open the URL in the browser
        driver.get(url)
        
        # Wait for the page to load (you may need to adjust the waiting time based on your website)
        driver.implicitly_wait(10)
        
        # Capture a screenshot of the page
        driver.save_screenshot(output_path)
        print("Screenshot captured successfully.")
    except Exception as e:
        print("Error occurred:", e)
    finally:
        # Close the browser
        driver.quit()

def compare_images(image1_path, image2_path):
    # Read the images from file
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert the images to grayscale (required for some comparison methods)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Perform image comparison using structural similarity (SSIM) index
    ssim_index, diff_image = cv2.compareSSIM(gray_image1, gray_image2, full=True)

    # Threshold for considering images as similar or different (adjust this based on your needs)
    similarity_threshold = 0.95

    if ssim_index > similarity_threshold:
        print("Website layout is similar.")
    else:
        print("Website layout is different.")

if __name__ == "__main__":
    # URL of the website you want to test
    website_url = "https://www.example.com"

    # Output path to save the screenshots
    screenshot_output = "path/to/screenshot.png"

    # Capture the screenshot of the website
    capture_screenshot(website_url, screenshot_output)

    # Path to the reference screenshot for comparison
    reference_screenshot = "path/to/reference_screenshot.png"

    # Perform visual testing by comparing the screenshots
    compare_images(reference_screenshot, screenshot_output)


import cv2

def extract_layout_from_screenshot(image_path):
    # Read the screenshot
    screenshot = cv2.imread(image_path)

    # Convert the screenshot to grayscale
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Apply edge detection or other preprocessing as needed
    edges = cv2.Canny(gray_screenshot, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process the contours and gather layout information
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print("Element position (x, y):", x, y)
        print("Element size (width, height):", w, h)

if __name__ == "__main__":
    # Path to the screenshot image
    screenshot_path = "path/to/screenshot.png"

    # Extract layout information from the screenshot
    extract_layout_from_screenshot(screenshot_path)


import requests

def get_figma_file(figma_file_id, access_token):
    headers = {
        "X-Figma-Token": access_token
    }
    url = f"https://api.figma.com/v1/files/{figma_file_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        figma_data = response.json()
        return figma_data
    else:
        print("Error:", response.status_code)
        return None

if __name__ == "__main__":
    # Figma file ID and access token
    figma_file_id = "your_figma_file_id"
    access_token = "your_figma_access_token"

    # Get the Figma file data
    figma_data = get_figma_file(figma_file_id, access_token)

    # Process the Figma data to gather layout information
    # (depends on the specific data structure returned by the Figma API)
    # Extract position, size, and other layout details of design elements

import cv2
import numpy as np
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)
vk_size = (5, 5)  # Set the kernel size for blurring
image_blurred = cv2.GaussianBlur(image, k_size, 0)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
alpha = 1.5  # Increase or decrease for different levels of contrast
enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
beta = 50  # Increase or decrease for different levels of brightness
enhanced_image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


import cv2

def preprocess_image(image_path):
    # Read the image from the file path
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise (optional, depending on the task)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply thresholding to create a binary image (optional, depending on the task)
    _, binary_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)

    # Perform other preprocessing steps as needed, like resizing, cropping, etc.

    return binary_image

# Example usage
image_path = 'path_to_your_image.jpg'
preprocessed_image = preprocess_image(image_path)
cv2.imshow('Preprocessed Image', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np

def feature_matching(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize the ORB detector and descriptor
    orb = cv2.ORB_create()

    # Find keypoints and descriptors in both images
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Create a Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance (smaller distances mean better matches)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches on a new image
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matched image
    cv2.imshow("Feature Matching", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image1_path = "path/to/your/image1.jpg"
    image2_path = "path/to/your/image2.jpg"

    feature_matching(image1_path, image2_path)



import cv2
import numpy as np

# Load YOLO model and class names
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load image and get dimensions
image = cv2.imread("your_image.jpg")  # Replace with the path to your image
height, width, channels = image.shape

# Prepare image for YOLO model
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Get detections
outs = net.forward(output_layers)

# Process detections
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # You can adjust this threshold based on your needs
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression to remove duplicate detections
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the detections on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the output image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

def compare_images_opencv(image_path1, image_path2):
    # Read images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Compute the structural similarity index (SSIM) between the two images
    ssim_index, _ = cv2.compareSSIM(image1, image2, full=True)

    # The SSIM index ranges from -1 to 1, with 1 being a perfect match
    print(f"SSIM index: {ssim_index}")

if __name__ == "__main__":
    image_path1 = "path_to_image1.jpg"
    image_path2 = "path_to_image2.jpg"
    compare_images_opencv(image_path1, image_path2)




import requests
from bs4 import BeautifulSoup

def extract_layout(url):
    try:
        # Fetch the webpage content using requests
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")
            return None

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract layout information (e.g., tags and their attributes)
        layout_info = []
        for tag in soup.find_all():
            tag_name = tag.name
            attributes = {attr: tag[attr] for attr in tag.attrs}
            layout_info.append((tag_name, attributes))

        return layout_info

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    url_to_extract = "https://www.bajajfinserv.in/"  # Replace with the URL of the website you want to extract the layout from
    layout_data = extract_layout(url_to_extract)
    if layout_data:
        print(layout_data)


import requests
from bs4 import BeautifulSoup

def extract_layout(url):
    try:
        # Fetch the website's HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Initialize layout dictionary to store tag counts
        layout = {}

        # Extract layout by counting tags
        for element in soup.find_all(True):
            tag = element.name
            layout[tag] = layout.get(tag, 0) + 1

        return layout

    except requests.exceptions.RequestException as e:
        print("Error fetching the website:", e)
        return None

if __name__ == "__main__":
    website_url1 = "https://example.com"  # Replace with the URL of the first website
    website_url2 = "https://example2.com"  # Replace with the URL of the second website

    layout1 = extract_layout(website_url1)
    layout2 = extract_layout(website_url2)

    if layout1 and layout2:
        print("Layout of Website 1:")
        for tag, count in layout1.items():
            print(f"{tag}: {count}")

        print("\nLayout of Website 2:")
        for tag, count in layout2.items():
            print(f"{tag}: {count}")

        # Compare layout of both websites
        if layout1 == layout2:
            print("\nThe layouts are identical.")
        else:
            print("\nThe layouts are different.")
    else:
        print("Layout extraction failed.")




import requests
from bs4 import BeautifulSoup

# Function to fetch design specifications from Figma API
def get_figma_design(api_key, design_link):
    headers = {"X-Figma-Token": api_key}
    response = requests.get(design_link, headers=headers)
    figma_data = response.json()
    # Extract design specifications from figma_data
    # ... Implement your extraction logic here ...

# Function to extract layout from a website
def extract_layout(url):
    try:
        # Fetch the website's HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Initialize layout dictionary to store tag counts
        layout = {}

        # Extract layout by counting tags
        for element in soup.find_all(True):
            tag = element.name
            layout[tag] = layout.get(tag, 0) + 1

        return layout

    except requests.exceptions.RequestException as e:
        print("Error fetching the website:", e)
        return None

if __name__ == "__main__":
    # Replace with your Figma API key and design link
    figma_api_key = "YOUR_FIGMA_API_KEY"
    figma_design_link = "YOUR_FIGMA_DESIGN_LINK"

    website_url = "https://example.com"  # Replace with the URL of the website you want to compare

    # Get design specifications from Figma
    figma_specifications = get_figma_design(figma_api_key, figma_design_link)

    # Extract layout from the website
    website_layout = extract_layout(website_url)

    # Compare design specifications and website layout
    # ... Implement your comparison logic here ...

import requests

def get_figma_design(api_key, design_link):
    headers = {"X-Figma-Token": api_key}
    response = requests.get(design_link, headers=headers)
    figma_data = response.json()
    # Implement your extraction logic here to extract design specifications from figma_data
    # Return the extracted design specifications

# Replace with your Figma API key and design link
figma_api_key = "YOUR_FIGMA_API_KEY"
figma_design_link = "YOUR_FIGMA_DESIGN_LINK"

# Get design specifications from Figma
figma_specifications = get_figma_design(figma_api_key, figma_design_link)

from bs4 import BeautifulSoup
import requests

def extract_layout(url):
    try:
        # Fetch the website's HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Initialize layout dictionary to store tag counts
        layout = {}

        # Extract layout by counting tags
        for element in soup.find_all(True):
            tag = element.name
            layout[tag] = layout.get(tag, 0) + 1

        return layout

    except requests.exceptions.RequestException as e:
        print("Error fetching the website:", e)
        return None

# Replace with the URLs of the websites you want to compare
website_url1 = "https://example1.com"
website_url2 = "https://example2.com"

# Extract layout from the websites
layout1 = extract_layout(website_url1)
layout2 = extract_layout(website_url2)


def calculate_similarity(layout1, layout2):
    # Implement your comparison logic here to compare the design specifications and layouts
    # Return a similarity score/rating based on the comparison results

# Calculate similarity between Figma design and website layout
similarity_rating = calculate_similarity(figma_specifications, layout2)

print("Rating of Website based on Similarity:", similarity_rating)

