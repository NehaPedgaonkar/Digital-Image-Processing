import struct

def write_final_compressed_file(jbig_file, paq_file, output_file, num_channels, q, d):
    """
    Combines header, JBIG data, and PAQ data into the final compressed file format.

    Args:
        jbig_file (str): Path to the JBIG encoded file.
        paq_file (str): Path to the PAQ encoded file.
        output_file (str): Path to the final compressed file.
        num_channels (int): Number of channels (1 for grayscale, 3 for RGB).
        q (int): Quantization parameter.
        d (int): Sampling distance.
    """
    # Step 1: Read JBIG and PAQ data
    with open(jbig_file, "rb") as jbig_fp:
        jbig_data = jbig_fp.read()
    with open(paq_file, "rb") as paq_fp:
        paq_data = paq_fp.read()

    # Step 2: Create the header
    # Header: [Size of JBIG (4 bytes), Num channels (1 byte), q (1 byte), d (3 bytes)]
    jbig_size = len(jbig_data)
    header = struct.pack(">I", jbig_size)  # 4 bytes for JBIG size (big-endian)
    header += struct.pack("B", num_channels)  # 1 byte for num_channels
    header += struct.pack("B", q)  # 1 byte for q
    header += struct.pack(">I", d)[1:]  # 3 bytes for sampling distance (drop 1 byte)

    # Step 3: Combine header and data
    with open(output_file, "wb") as output_fp:
        output_fp.write(header)  # Write header
        output_fp.write(jbig_data)  # Write JBIG data
        output_fp.write(paq_data)  # Write PAQ data

    print(f"Final compressed file saved to {output_file}")


# Example Usage
if __name__ == "__main__":
    # Input paths
    jbig_file_path = "edges.jbg"  # JBIG encoded file
    paq_file_path = "pixel_values.paq"  # PAQ encoded pixel values
    final_output_path = "compressed_image.cmp"  # Final compressed file

    # Compression parameters
    num_channels = 1  # Grayscale image
    quantization_parameter = 4  # q
    sampling_distance = 5  # d

    # Write the final compressed file
    write_final_compressed_file(
        jbig_file_path, paq_file_path, final_output_path, num_channels, quantization_parameter, sampling_distance
    )
