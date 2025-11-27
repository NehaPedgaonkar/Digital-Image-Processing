import cv2
import subprocess
import os

def encode_with_jbig(input_binary_image_path, jbig_file_path):
    """
    Encodes a binary image using JBIG-KIT.

    Args:
        input_binary_image_path (str): Path to the input binary (bi-level) image in PBM format.
        jbig_file_path (str): Path to save the JBIG encoded file.
    """
    # Step 1: Use pbmtojbg to compress the binary image
    try:
        subprocess.run(["pbmtojbg85", input_binary_image_path, jbig_file_path], check=True)
        print(f"JBIG encoding successful. Saved to {jbig_file_path}")
    except FileNotFoundError:
        raise FileNotFoundError("The `pbmtojbg` tool is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"JBIG encoding failed with error: {e}")

def save_binary_image_as_pbm(binary_image, pbm_file_path):
    """
    Saves a binary image as a PBM file.

    Args:
        binary_image (np.ndarray): Binary image to save.
        pbm_file_path (str): Path to save the PBM file.
    """
    cv2.imwrite(pbm_file_path, binary_image)
    print(f"Binary image saved as PBM at {pbm_file_path}")

if __name__ == "__main__":
    # Input paths
    binary_edge_image_path = "marr_hildreth_edges.jpg"  # Binary edge image from Step 1
    pbm_file_path = "edges.pbm"  # Intermediate PBM file
    jbig_file_path = "edges.jbg"  # JBIG-compressed file

    # Load binary edge image
    binary_edge_image = cv2.imread(binary_edge_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary edge image is bi-level (0 and 255)
    _, binary_edge_image = cv2.threshold(binary_edge_image, 127, 255, cv2.THRESH_BINARY)

    # Step 1: Save binary image as PBM
    save_binary_image_as_pbm(binary_edge_image, pbm_file_path)

    # Step 2: Encode the PBM file using JBIG
    encode_with_jbig(pbm_file_path, jbig_file_path)
