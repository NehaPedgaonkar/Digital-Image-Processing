import os
import struct
import subprocess
import cv2
import numpy as np
def split_encoded_file(encoded_file_path):
    """
    Splits the encoded file into JBIG data and PAQ data.

    Args:
        encoded_file_path (str): Path to the encoded file.
    
    Returns:
        dict: Parsed header information and raw JBIG and PAQ data.
    """
    with open(encoded_file_path, "rb") as file:
        # Read the header
        header = file.read(9)  # Header size: 4 + 1 + 1 + 3 bytes
        jbig_size, num_channels, q, d = struct.unpack(">IBB3s", header)
        d = int.from_bytes(d, byteorder="big")  # Convert 3-byte sampling distance to integer

        # Read JBIG data
        jbig_data = file.read(jbig_size)

        # Read PAQ data
        paq_data = file.read()

    return {
        "jbig_size": jbig_size,
        "num_channels": num_channels,
        "q": q,
        "d": d,
        "jbig_data": jbig_data,
        "paq_data": paq_data
    }
def decode_jbig(jbig_data, output_pbm_path):
    """
    Decodes JBIG data to a PBM file.

    Args:
        jbig_data (bytes): JBIG encoded data.
        output_pbm_path (str): Path to save the decoded PBM file.
    """
    # Save JBIG data to a temporary file
    temp_jbig_file = "temp.jbg"
    with open(temp_jbig_file, "wb") as file:
        file.write(jbig_data)

    # Decode JBIG using jbgtopbm85
    try:
        subprocess.run(["jbgtopbm85", temp_jbig_file, output_pbm_path], check=True)
        print(f"JBIG decoding successful. Saved to {output_pbm_path}")
    except FileNotFoundError:
        raise FileNotFoundError("The `jbgtopbm85` tool is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"JBIG decoding failed with error: {e}")
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_jbig_file):
            os.remove(temp_jbig_file)
def decode_paq(paq_data, output_raw_path):
    """
    Decodes PAQ data to reconstruct the quantized pixel values.

    Args:
        paq_data (bytes): PAQ encoded data.
        output_raw_path (str): Path to save the reconstructed raw pixel values.
    """
    # Save PAQ data to a temporary file
    temp_paq_file = "temp.paq8px208fix1"
    with open(temp_paq_file, "wb") as file:
        file.write(paq_data)

    # Decode PAQ using paq8px
    try:
        subprocess.run(["paq8px", "-d", temp_paq_file, output_raw_path], check=True)
        print(f"PAQ decoding successful. Reconstructed data saved to {output_raw_path}")
    except FileNotFoundError:
        raise FileNotFoundError("The `paq8px` tool is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"PAQ decoding failed with error: {e}")
    # finally:
        # # Cleanup temporary file
        # if os.path.exists(temp_paq_file):
        #     os.remove(temp_paq_file)
def linear_interpolation_along_edges(edge_image, quantized_values, sampling_distance):
    """
    Performs linear interpolation along edges to reconstruct missing pixel values.

    Args:
        edge_image (np.ndarray): Bi-level edge image.
        quantized_values (np.ndarray): Decoded and quantized pixel values.
        sampling_distance (int): Sampling distance used during encoding.

    Returns:
        np.ndarray: Interpolated pixel values at edge positions.
    """
    reconstructed_image = np.zeros_like(edge_image, dtype=np.float32)
    rows, cols = edge_image.shape
    value_idx = 0

    # Iterate over the edge image
    for i in range(rows):
        for j in range(cols):
            if edge_image[i, j] == 255:  # Edge pixel
                if value_idx >= len(quantized_values):
                    print(f"Warning: Not enough quantized values. Stopping at index {value_idx}.")
                    return reconstructed_image
                reconstructed_image[i, j] = quantized_values[value_idx]
                value_idx += 1

    return reconstructed_image

if __name__ == "__main__":
    # Input paths
    encoded_file_path = "compressed_image.cmp"  # Encoded file
    decoded_pbm_path = "decoded_edges.pbm"  # Output PBM file for decoded JBIG data
    decoded_raw_path = "reconstructed_values.raw"  # Output file for reconstructed PAQ values

    # Decode the file
    parsed_data = split_encoded_file(encoded_file_path)

    # Step 1: Decode JBIG data
    decode_jbig(parsed_data["jbig_data"], decoded_pbm_path)

    # Step 2: Decode PAQ data
    decode_paq(parsed_data["paq_data"], decoded_raw_path)

    # Step 3: Load decoded data
    edge_image = cv2.imread(decoded_pbm_path, cv2.IMREAD_GRAYSCALE)
    quantized_values = np.fromfile(decoded_raw_path, dtype=np.uint8)

    # Step 4: Reconstruct pixel values
    reconstructed_image = linear_interpolation_along_edges(edge_image, quantized_values, parsed_data["d"])

    # Save the reconstructed image
    cv2.imwrite("reconstructed_image.jpg", np.clip(reconstructed_image, 0, 255).astype(np.uint8))
