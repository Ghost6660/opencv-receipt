import cv2
import pytesseract
from matplotlib import pyplot as plt

image = 'sharpened_image.png'
image = cv2.imread(image,  cv2.IMREAD_GRAYSCALE)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
plt.figure(1)
plt.imshow(im_bw)
plt.title('Cropped Image')
plt.axis('on')
plt.show()
custom_config = r'-l eng+ara --oem 1 --psm 11'

extracted_text = pytesseract.image_to_string(im_bw, config=custom_config)
print("Extracted text:\n")
print(extracted_text)