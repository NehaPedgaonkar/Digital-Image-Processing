import cv2
import numpy as np
import subprocess
import os
def extract_pixel_values(binary_edge_image, original_image, subsampling_distance=1):
    """
    Extracts pixel values around edges and along the image boundary.

    Args:
        binary_edge_image (np.ndarray): Binary edge image (0 or 255).
        original_image (np.ndarray): Original grayscale or color image.
        subsampling_distance (int): Subsampling distance (d).
    
    Returns:
        np.ndarray: Extracted pixel values.
    """
    rows, cols = binary_edge_image.shape
    pixel_values = []

    # Extract pixel values from both sides of edges
    for i in range(rows):
        for j in range(cols):
            if binary_edge_image[i, j] == 255:  # Edge pixel
                # Get neighboring pixels
                neighbors = [
                    (i - 1, j), (i + 1, j),  # Top and bottom
                    (i, j - 1), (i, j + 1)   # Left and right
                ]
                for ni, nj in neighbors:
                    if 0 <= ni < rows and 0 <= nj < cols:
                        pixel_values.append(original_image[ni, nj])

    # Include boundary pixel values
    pixel_values.extend(original_image[0, :])  # Top boundary
    pixel_values.extend(original_image[-1, :])  # Bottom boundary
    pixel_values.extend(original_image[:, 0])  # Left boundary
    pixel_values.extend(original_image[:, -1])  # Right boundary

    # Subsample the pixel values
    pixel_values = np.array(pixel_values)[::subsampling_distance]

    return pixel_values

def quantize_pixel_values(pixel_values, q):
    """
    Uniformly quantizes pixel values to 2^q levels.

    Args:
        pixel_values (np.ndarray): Array of pixel values.
        q (int): Quantization parameter (number of bits per pixel).
    
    Returns:
        np.ndarray: Quantized pixel values.
    """
    levels = 2**q
    max_value = 255
    quantized_values = np.round(pixel_values / max_value * (levels - 1)) * (max_value / (levels - 1))
    return quantized_values.astype(np.uint8)

def compress_pixel_values_with_paq(pixel_values, output_file, compression_level=8):
    """
    Compresses pixel values using PAQ.

    Args:
        pixel_values (np.ndarray): Quantized pixel values.
        output_file (str): Path to the output compressed file.
        compression_level (int): Compression level (0-12, higher is slower but better compression).
    """
    # Save pixel values to a temporary file
    temp_file = "pixel_values.raw"
    pixel_values.tofile(temp_file)

    # Use PAQ for compression
    try:
        subprocess.run(["paq8px", f"-{compression_level}", temp_file, output_file], check=True)
        print(f"PAQ compression successful. Saved to {output_file}")
    except FileNotFoundError:
        raise FileNotFoundError("PAQ compression tool is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"PAQ compression failed with error: {e}")
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    # Input paths
    binary_edge_image_path = "edges.pbm"  # Binary edge image from Step 2
    original_image_path = "../../2_cartoon_image.jpg"  # Original grayscale or color image
    compressed_pixel_values_path = "pixel_values.paq"  # Compressed pixel values file

    # Parameters
    subsampling_distance = 5  # Subsample every 5th pixel
    quantization_bits = 4  # Quantize to 2^4 = 16 levels
    compression_level = 10  # Compression level for PAQ (0-12)

    # Load images
    binary_edge_image = cv2.imread(binary_edge_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Extract pixel values around edges
    pixel_values = extract_pixel_values(binary_edge_image, original_image, subsampling_distance)

    # Step 2: Quantize pixel values
    quantized_values = quantize_pixel_values(pixel_values, quantization_bits)

    # Step 3: Compress pixel values with PAQ
    compress_pixel_values_with_paq(quantized_values, compressed_pixel_values_path, compression_level)
