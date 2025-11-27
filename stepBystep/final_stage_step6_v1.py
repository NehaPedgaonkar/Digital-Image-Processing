import numpy as np
from scipy.ndimage import laplace
import cv2
def reconstruct_image_with_diffusion(edge_image, reconstructed_image, max_iterations=1000, tolerance=1e-6):
    """
    Reconstructs missing data using homogeneous diffusion to solve the Laplace equation (Δu = 0).

    Args:
        edge_image (np.ndarray): Binary edge image (0 or 255) indicating edge locations.
        reconstructed_image (np.ndarray): Initial image with known pixel values at edges and boundaries.
        max_iterations (int): Maximum number of iterations for diffusion.
        tolerance (float): Convergence tolerance for stopping the iteration.
    
    Returns:
        np.ndarray: Fully reconstructed image.
    """
    # Mask indicating known (boundary) pixel values
    known_mask = (edge_image == 255)
    
    # Initialize the inpainting region with the given pixel values
    prev_image = reconstructed_image.copy()

    for iteration in range(max_iterations):
        # Apply Laplace operator (finite difference approximation)
        next_image = laplace(prev_image)
        next_image[known_mask] = reconstructed_image[known_mask]  # Keep boundary values fixed

        # Check for convergence
        max_diff = np.max(np.abs(next_image - prev_image))
        if max_diff < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break

        prev_image = next_image

    return next_image

if __name__ == "__main__":
    # Input paths
    edge_image_path = "decoded_edges.pbm"  # Decoded edge image
    reconstructed_partial_path = "reconstructed_image.jpg"  # Partially reconstructed image (Step 1)

    # Load edge image and initial reconstructed image
    edge_image = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
    reconstructed_partial = cv2.imread(reconstructed_partial_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Perform homogeneous diffusion for inpainting
    final_image = reconstruct_image_with_diffusion(edge_image, reconstructed_partial)

    # Save the final reconstructed image
    cv2.imwrite("final_reconstructed_image.jpg", np.clip(final_image, 0, 255).astype(np.uint8))
    print("Final reconstructed image saved as 'final_reconstructed_image.jpg'.")
