import cv2
import numpy as np
import subprocess
import os

# Step 1: Edge Detection
def detect_edges(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Compute Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Detect zero-crossings
    edges = (np.sign(laplacian) != np.sign(np.roll(laplacian, 1, axis=0))) | \
            (np.sign(laplacian) != np.sign(np.roll(laplacian, 1, axis=1)))

    return edges.astype(np.uint8) * 255  # Convert to binary image (0 or 255)

# Step 2: Save Edge Image in PBM Format
def save_as_pbm(binary_image, pbm_path):
    # OpenCV writes PBM in the correct format
    cv2.imwrite(pbm_path, binary_image)

# Step 3: Compress Using JBIG
def jbig_compress(pbm_path, output_path):
    # Run JBIG compression using jbig85
    subprocess.run(["jbig85.o", pbm_path, output_path], check=True)

# Step 4: Decompress Using JBIG
def jbig_decompress(jbig_path, pbm_path):
    # Run JBIG decompression using jbigdec
    subprocess.run(["jbigtopbm85", jbig_path, pbm_path], check=True)

# Step 5: Convert PBM Back to Image
def pbm_to_image(pbm_path, output_path):
    # Load PBM file and save it in a viewable format
    img = cv2.imread(pbm_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_path, img)

# Driver Code
if __name__ == "__main__":
    # Input image path
    input_image_path = "your_image.png"

    # Output paths
    edge_pbm_path = "edges.pbm"
    compressed_jbig_path = "edges.jbg"
    decompressed_pbm_path = "decompressed_edges.pbm"
    reconstructed_image_path = "reconstructed_edges.jpg"

    # Step 1: Detect edges
    edges = detect_edges(input_image_path)

    # Step 2: Save edges as PBM
    save_as_pbm(edges, edge_pbm_path)

    # Step 3: Compress using JBIG
    jbig_compress(edge_pbm_path, compressed_jbig_path)

    # Step 4: Decompress using JBIG
    jbig_decompress(compressed_jbig_path, decompressed_pbm_path)

    # Step 5: Convert decompressed PBM back to image
    pbm_to_image(decompressed_pbm_path, reconstructed_image_path)

    print(f"Compression and reconstruction completed.")
    print(f"Compressed JBIG file: {compressed_jbig_path}")
    print(f"Reconstructed image: {reconstructed_image_path}")

    # Cleanup intermediate files if needed
    os.remove(edge_pbm_path)
    os.remove(decompressed_pbm_path)
