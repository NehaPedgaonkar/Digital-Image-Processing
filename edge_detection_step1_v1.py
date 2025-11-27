import cv2
import numpy as np

def marr_hildreth_edge_detection(image_path, low_threshold, high_threshold):
    """
    Marr-Hildreth edge detection with Gaussian smoothing and hysteresis thresholding.

    Args:
        image_path (str): Path to the input image.
        low_threshold (float): Lower gradient magnitude threshold for edge candidates.
        high_threshold (float): Higher gradient magnitude threshold for seed points.
    
    Returns:
        np.ndarray: Binary edge image with well-localized edges.
    """
    # Step 1: Load the grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image could not be loaded. Please check the file path.")

    # Step 2: Apply Gaussian smoothing
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)  # Adjust kernel size and sigma as needed

    # Step 3: Compute the Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Step 4: Detect zero-crossings
    zero_crossings = np.zeros_like(laplacian, dtype=np.uint8)
    rows, cols = laplacian.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if laplacian[i, j] == 0:
                if ((laplacian[i - 1, j] > 0 and laplacian[i + 1, j] < 0) or
                    (laplacian[i - 1, j] < 0 and laplacian[i + 1, j] > 0) or
                    (laplacian[i, j - 1] > 0 and laplacian[i, j + 1] < 0) or
                    (laplacian[i, j - 1] < 0 and laplacian[i, j + 1] > 0)):
                    zero_crossings[i, j] = 255
            elif laplacian[i, j] > 0:
                if ((laplacian[i - 1, j] < 0) or (laplacian[i + 1, j] < 0) or
                    (laplacian[i, j - 1] < 0) or (laplacian[i, j + 1] < 0)):
                    zero_crossings[i, j] = 255

    # Step 5: Compute the gradient magnitude
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Step 6: Apply hysteresis thresholding
    strong_edges = (gradient_magnitude >= high_threshold)
    weak_edges = ((gradient_magnitude >= low_threshold) & (gradient_magnitude < high_threshold))

    final_edges = np.zeros_like(img, dtype=np.uint8)
    final_edges[strong_edges] = 255  # Seed points
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j] and np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                final_edges[i, j] = 255  # Add weak edges connected to strong edges

    return final_edges

if __name__ == "__main__":
    # Test parameters
    image_path = "cartoon_image.jpg"  # Replace with your image path
    low_threshold = 30
    high_threshold = 100

    # Perform Marr-Hildreth edge detection
    edges = marr_hildreth_edge_detection(image_path, low_threshold, high_threshold)

    # Save and display the results
    cv2.imwrite("marr_hildreth_edges.jpg", edges)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
