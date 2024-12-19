import cv2
import numpy as np

# Apply Canny edge detection
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert video to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise with Gaussian blur
    edges = cv2.Canny(blur, 50, 150)  # Apply Canny edge detection
    return edges

# Crop the image to focus on the region of interest (lane area)
def crop(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    triangle = np.array([[(150, height), (640, 330), (1200, height)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)  # Mask everything except the triangle
    cropped_image = cv2.bitwise_and(edges, mask)  # Apply the mask
    return cropped_image

# Detect lines using Hough Transform
def hough_lines(cropped_image):
    return cv2.HoughLinesP(
        cropped_image,
        rho=2,
        theta=np.pi/180,
        threshold=100,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

# Compute average slope and intercept for left and right lane lines
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit
            if slope < 0:
                left_fit.append((slope, intercept))  # Left lane (negative slope)
            else:
                right_fit.append((slope, intercept))  # Right lane (positive slope)

    left_line = make_points(image, np.mean(left_fit, axis=0) if left_fit else None)
    right_line = make_points(image, np.mean(right_fit, axis=0) if right_fit else None)

    return [left_line, right_line]

# Convert slope and intercept into coordinates for drawing
def make_points(image, fit_avg):
    if fit_avg is None:
        return None

    slope, intercept = fit_avg
    if abs(slope) < 1e-3:  # Avoid division by near-zero slopes
        return None

    height = image.shape[0]
    y1 = height  # Bottom of the image
    y2 = int(height * 3.0 / 5)  # Higher up in the image

    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        width = image.shape[1]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        return [[x1, y1, x2, y2]]
    except Exception as e:
        print(f"Error in make_points: {e}")
        return None

# Draw lane lines on a blank image
def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

# After returning the line ^, add some weight to the line so it's more visible
def add_weight(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# Main function to process the video and detect lanes
def main():
    video = input("Enter a mp4 file: ")
    video_file = cv2.VideoCapture(video)

    while video_file.isOpened():
        ret, frame = video_file.read()
        if not ret:
            break

        # Detect edges
        edges = canny(frame)

        # Crop to region of interest
        cropped_edges = crop(edges)

        # Detect lines
        lines = hough_lines(cropped_edges)

        # Compute average lane lines
        lane_lines = average_slope_intercept(frame, lines)

        # Debugging: Print lane lines
        print("Lane lines:", lane_lines)

        # Draw lines
        line_image = display_lines(frame, lane_lines)

        # Overlay lines on the original frame
        weighted_image = add_weight(frame, line_image)

        # Display the result
        cv2.imshow("Lane Detection", weighted_image)

        # Exit on 'Esc' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_file.release() # This holds our video
    cv2.destroyAllWindows() #Exits program


main()