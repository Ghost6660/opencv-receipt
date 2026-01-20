import cv2 as cv

def detect_edges(image_path):
    image = cv.imread(image_path)
    original = image.copy()

    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale', grayscale)

    blurred = cv.GaussianBlur(grayscale, (5, 5), 0)
    cv.imshow('blurred', blurred)

    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('binary', binary)

    canny = cv.Canny(binary, 50, 150)
    cv.imshow('Canny edge detection', canny)

    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:5]

    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            doc_contour = approx
            break

    else:
        print("no document detected")
        return None
    
    cv.drawContours(original, [doc_contour], -1, (0, 255, 0), 10)
    cv.imshow("Detected document contour", original)

    cv.waitKey(0)
    cv.destroyAllWindows()


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    press = cv.waitKey(1)
    if press == ord('q'):
        cap.release()
        cv.destroyAllWindows()
    
    elif press == ord('c'):
        cv.imwrite('captured_image.png', frame)
        cv.imshow('captured_image.png', gray)
        print("Image captured and saved as 'captured_image.png'")
        detect_edges('captured_image.png')



