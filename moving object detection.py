import cv2

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Compute the absolute difference between current and previous frame
    diff = cv2.absdiff(frame1, frame2)

    # Convert difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to highlight differences
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill gaps
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours (moving object outlines)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) < 800:  # Skip small movements
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Moving Object Detection", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Exit on pressing 'q'
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
