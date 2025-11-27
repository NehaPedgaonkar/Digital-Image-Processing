import cv2
import numpy as np
import subprocess
import os
from scipy.ndimage import laplace

def homogeneous_diffusion(edge_image, channel_image, max_iterations=1000, tolerance=1e-6):
    """
    Performs homogeneous diffusion to fill missing data in a single color channel.

    Args:
        edge_image (np.ndarray): Binary edge image (0 or 255).
        channel_image (np.ndarray): Channel image with known values at edges.
        max_iterations (int): Maximum number of iterations for diffusion.
        tolerance (float): Convergence tolerance.
    
    Returns:
        np.ndarray: Fully reconstructed channel image.
    """
    # Create a mask for known pixels (edges)
    known_mask = (edge_image == 255)
    
    # Initialize the reconstructed image with the given channel image
    reconstructed_image = channel_image.copy()

    for iteration in range(max_iterations):
        # Apply Laplace operator (finite difference approximation)
        laplace_image = laplace(reconstructed_image)
        next_image = reconstructed_image + laplace_image

        # Ensure known pixels remain fixed
        next_image[known_mask] = channel_image[known_mask]

        # Check for convergence
        max_diff = np.max(np.abs(next_image - reconstructed_image))
        if max_diff < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

        reconstructed_image = next_image

    return reconstructed_image
def reconstruct_color_image_with_diffusion(binary_edge_image, pixel_values_channels, image_shape):
    """
    Reconstructs a color image using pixel values assigned to edges and diffusion.

    Args:
        binary_edge_image (np.ndarray): Binary edge image (0 or 255).
        pixel_values_channels (list): List of pixel values for each channel (R, G, B).
        image_shape (tuple): Shape of the original color image (H, W, 3).
    
    Returns:
        np.ndarray: Fully reconstructed color image.
    """
    reconstructed_channels = []

    for channel_idx, channel_values in enumerate(pixel_values_channels):
        # Initialize channel image with zeros
        channel_image = np.zeros(image_shape[:2], dtype=np.float32)
        rows, cols = binary_edge_image.shape
        value_idx = 0

        # Place known pixel values at edge locations
        for i in range(rows):
            for j in range(cols):
                if binary_edge_image[i, j] == 255:  # Edge pixel
                    if value_idx < len(channel_values):
                        channel_image[i, j] = channel_values[value_idx]
                        value_idx += 1

        # Apply diffusion to fill missing pixels
        reconstructed_channel = homogeneous_diffusion(binary_edge_image, channel_image)
        reconstructed_channels.append(reconstructed_channel)

    # Merge reconstructed channels into a single color image
    return cv2.merge(reconstructed_channels).astype(np.uint8)


# Step 1: Extract Pixel Values Around Edges
def extract_pixel_values_color(binary_edge_image, color_image, subsampling_distance=1):
    channels = cv2.split(color_image)  # Split image into R, G, B channels
    pixel_values = []

    for channel in channels:
        rows, cols = binary_edge_image.shape
        channel_pixel_values = []

        # Extract pixel values from edges and boundaries
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
                            channel_pixel_values.append(channel[ni, nj])

        # Include boundary pixel values
        channel_pixel_values.extend(channel[0, :])  # Top boundary
        channel_pixel_values.extend(channel[-1, :])  # Bottom boundary
        channel_pixel_values.extend(channel[:, 0])  # Left boundary
        channel_pixel_values.extend(channel[:, -1])  # Right boundary

        # Subsample the pixel values
        channel_pixel_values = np.array(channel_pixel_values)[::subsampling_distance]
        pixel_values.append(channel_pixel_values)

    return pixel_values


# Step 2: Quantize Pixel Values
def quantize_pixel_values_color(pixel_values, q):
    quantized_values = []
    levels = 2**q
    max_value = 255

    for channel_values in pixel_values:
        quantized_channel = np.round(channel_values / max_value * (levels - 1)) * (max_value / (levels - 1))
        quantized_values.append(quantized_channel.astype(np.uint8))

    return quantized_values


# Step 3: Compress Pixel Values Using PAQ
def compress_pixel_values_color_with_paq(pixel_values, output_file_prefix, compression_level=8):
    """
    Compresses pixel values using PAQ for each color channel.

    Args:
        pixel_values (list): List of quantized pixel values for each channel (R, G, B).
        output_file_prefix (str): Prefix for the output compressed file (one per channel).
        compression_level (int): Compression level (0-12, higher is slower but better compression).
    """
    for idx, channel_values in enumerate(pixel_values):
        temp_file = f"channel_{idx}_values.raw"
        compressed_file = f"{output_file_prefix}_channel_{idx}.paq8px208fix1"
        channel_values.tofile(temp_file)

        # Use PAQ for compression
        try:
            subprocess.run(["paq8px", f"-{compression_level}", temp_file, compressed_file], check=True)
            print(f"PAQ compression successful for channel {idx}. Saved to {compressed_file}")
        except FileNotFoundError:
            raise FileNotFoundError("PAQ compression tool is not installed or not in PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"PAQ compression failed for channel {idx} with error: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)



def decompress_paq_to_pixel_values(paq_file, output_raw_file):
    """
    Decompresses PAQ data to reconstruct the quantized pixel values.

    Args:
        paq_file (str): Path to the PAQ-compressed file (must end with .paq8px208fix1).
        output_raw_file (str): Path to save the decompressed raw pixel values.
    
    Returns:
        np.ndarray: Decompressed pixel values.
    """
    try:
        subprocess.run(["paq8px", "-d", paq_file, output_raw_file], check=True)
        print(f"PAQ decompression successful. Raw data saved to {output_raw_file}")
    except FileNotFoundError:
        raise FileNotFoundError("PAQ compression tool is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"PAQ decompression failed with error: {e}")
    
    return np.fromfile(output_raw_file, dtype=np.uint8)



# Step 5: Reconstruct Image from Edge Pixels
def reconstruct_color_image(binary_edge_image, pixel_values_channels):
    rows, cols = binary_edge_image.shape
    reconstructed_channels = []

    for channel_values in pixel_values_channels:
        reconstructed_channel = np.zeros((rows, cols), dtype=np.uint8)
        value_idx = 0

        for i in range(rows):
            for j in range(cols):
                if binary_edge_image[i, j] == 255:  # Edge pixel
                    if value_idx < len(channel_values):
                        reconstructed_channel[i, j] = channel_values[value_idx]
                        value_idx += 1
        
        reconstructed_channels.append(reconstructed_channel)

    return cv2.merge(reconstructed_channels)  # Combine R, G, B channels into a color image


# Main Workflow
if __name__ == "__main__":
    # Input paths
    binary_edge_image_path = "edges.pbm"  # Binary edge image
    original_image_path = "../../2_cartoon_image.jpg"  # Original color image
    compressed_pixel_values_path_prefix = "pixel_values_color"  # Prefix for compressed files
    reconstructed_image_path = "reconstructed_color_image.jpg"  # Final reconstructed image

    # Parameters
    subsampling_distance = 5  # Subsample every 5th pixel
    quantization_bits = 4  # Quantize to 2^4 = 16 levels
    compression_level = 10  # Compression level for PAQ (0-12)

    # Load images
    binary_edge_image = cv2.imread(binary_edge_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path)  # Load color image
    image_shape = original_image.shape

    # Step 1: Extract pixel values for each channel
    pixel_values_channels = extract_pixel_values_color(binary_edge_image, original_image, subsampling_distance)

    # Step 2: Quantize pixel values for each channel
    quantized_values_channels = quantize_pixel_values_color(pixel_values_channels, quantization_bits)

    # Step 3: Compress pixel values with PAQ for each channel
    compress_pixel_values_color_with_paq(quantized_values_channels, compressed_pixel_values_path_prefix, compression_level)

    # Step 4: Decompress PAQ files and reconstruct pixel values
    pixel_values_channels_reconstructed = []
    for idx in range(3):  # For R, G, B channels
        paq_file = f"{compressed_pixel_values_path_prefix}_channel_{idx}.paq8px208fix1"
        raw_file = f"decompressed_channel_{idx}.raw"
        pixel_values = decompress_paq_to_pixel_values(paq_file, raw_file)
        pixel_values_channels_reconstructed.append(pixel_values)

    # Step 5: Reconstruct the color image with diffusion
    reconstructed_image = reconstruct_color_image_with_diffusion(binary_edge_image, pixel_values_channels_reconstructed, image_shape)

    # Save and display the reconstructed image
    cv2.imwrite(reconstructed_image_path, reconstructed_image)
    print(f"Reconstructed color image saved as {reconstructed_image_path}")

    cv2.imshow("Reconstructed Color Image", reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
