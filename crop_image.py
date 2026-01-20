from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import cv2 as cv

# Read the captured image
image_path = 'scanned.png'
image = Image.open(image_path)

# # Detect document edges using OpenCV
# cv_image = cv.imread(image_path)
# grayscale = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
# blurred = cv.GaussianBlur(grayscale, (5, 5), 0)
# _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# canny = cv.Canny(binary, 50, 150)

# contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

# doc_contour = None
# for contour in contours:
#     peri = cv.arcLength(contour, True)
#     approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    
#     if len(approx) == 4:
#         doc_contour = approx
#         break

# if doc_contour is None:
#     print("No document detected, cropping entire image")
#     cropped_image = image
# else:
#     # Get bounding rectangle
#     x, y, w, h = cv.boundingRect(doc_contour)
#     box = (x, y, x + w, y + h)
    
#     # Crop the image
#     cropped_image = image.crop(box)
#     print(f"Document detected and cropped: {cropped_image.size}")
#     cropped_image.save('cropped_image.png')
#     print("Cropped image saved as 'cropped_image.png'")
    
    #enhance contrast and sharpen the image


sharp_img = ImageEnhance.Contrast(image)
sharp_img = sharp_img.enhance(4.2)
sharp_img = sharp_img.filter(ImageFilter.SHARPEN)

# Save cropped image

sharp_img.save('sharpened_image.png')
print("sharpened image saved as 'sharpened_image.png'")

# Display the images in separate figures
plt.figure(1)
plt.imshow(image)
plt.title('Cropped Image')
plt.axis('on')

plt.figure(2)
plt.imshow(sharp_img)
plt.title('Sharpened Image')
plt.axis('on')

plt.show()