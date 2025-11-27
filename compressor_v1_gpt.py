import numpy as np
import cv2
import matplotlib.pyplot as plt
import heapq

# Define quantization matrix
QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Divide image into 8x8 blocks
def divide_into_blocks(image, block_size=8):
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):  # Padding for non-8x8 areas
                block = np.pad(block, ((0, block_size - block.shape[0]), (0, block_size - block.shape[1])), 'constant')
            blocks.append(block)
    return np.array(blocks)

# Apply 2D DCT
def dct2(block):
    return cv2.dct(block.astype(np.float32))

# Apply 2D IDCT
def idct2(block):
    return cv2.idct(block.astype(np.float32))

# Quantize the DCT coefficients
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

# Dequantize the coefficients
def dequantize(block, quant_matrix):
    return block * quant_matrix

# Huffman encoding (simplified using a heap-based tree)
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(tree, code="", code_dict={}):
    if tree is None:
        return

    if tree.symbol is not None:
        code_dict[tree.symbol] = code
    build_huffman_codes(tree.left, code + "0", code_dict)
    build_huffman_codes(tree.right, code + "1", code_dict)

    return code_dict

# Encode the blocks
def huffman_encode(blocks):
    flat_data = np.concatenate([block.flatten() for block in blocks])
    unique, counts = np.unique(flat_data, return_counts=True)
    freq_dict = dict(zip(unique, counts))

    huffman_tree = build_huffman_tree(freq_dict)
    huffman_codes = build_huffman_codes(huffman_tree)

    encoded_data = "".join([huffman_codes[val] for val in flat_data])
    return encoded_data, huffman_tree

# Decode Huffman-encoded data
def huffman_decode(encoded_data, huffman_tree, num_blocks, block_size=8):
    decoded = []
    node = huffman_tree

    for bit in encoded_data:
        node = node.left if bit == "0" else node.right

        if node.symbol is not None:
            decoded.append(node.symbol)
            node = huffman_tree

    # Reshape into blocks
    decoded = np.array(decoded)
    blocks = []
    for i in range(num_blocks):
        start_idx = i * (block_size * block_size)
        end_idx = start_idx + (block_size * block_size)
        block = decoded[start_idx:end_idx].reshape(block_size, block_size)
        blocks.append(block)

    return np.array(blocks)

def jpeg_compression(image, quant_matrix):
    blocks = divide_into_blocks(image)
    dct_blocks = np.array([dct2(block) for block in blocks])
    quantized_blocks = np.array([quantize(block, quant_matrix) for block in dct_blocks])

    encoded_data, huffman_tree = huffman_encode(quantized_blocks)
    return encoded_data, huffman_tree, len(quantized_blocks), image.shape

def jpeg_decompression(encoded_data, huffman_tree, num_blocks, image_shape, quant_matrix):
    decoded_blocks = huffman_decode(encoded_data, huffman_tree, num_blocks)
    dequantized_blocks = np.array([dequantize(block, quant_matrix) for block in decoded_blocks])
    idct_blocks = np.array([idct2(block) for block in dequantized_blocks])

    return recompose_image(idct_blocks, image_shape)


# Recompose image from blocks
def recompose_image(blocks, shape, block_size=8):
    h, w = shape
    image = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            image[i:i+block_size, j:j+block_size] = blocks[idx][:block_size, :block_size]
            idx += 1
    return np.clip(image, 0, 255).astype(np.uint8)

# Main
if __name__ == "__main__":
    # Load the grayscale PNG image
    image = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load image. Check the file path.")
        exit()

    # Define quality factors
    quality_factors = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 10 = high compression, 100 = low compression
    rmse_values = []
    bpp_values = []

    for qf in quality_factors:
        # Scale the quantization matrix based on the quality factor
        scaled_quant_matrix = np.clip((QUANTIZATION_MATRIX * (100 / qf)), 1, 255).astype(int)

        # Compress the image
        print(f"Compressing with Quality Factor: {qf}...")
        encoded_data, huffman_tree, num_blocks, image_shape = jpeg_compression(image, scaled_quant_matrix)

        # Save compressed data to a file (for size comparison)
        compressed_file = f'compressed_qf_{qf}.bin'
        with open(compressed_file, 'wb') as f:
            # Save image metadata
            f.write(np.array(image_shape, dtype=np.int32).tobytes())  # Image dimensions
            f.write(np.array(num_blocks, dtype=np.int32).tobytes())   # Number of blocks
            f.write(scaled_quant_matrix.flatten().astype(np.int32).tobytes())  # Quantization matrix
            # Save encoded data as binary
            encoded_bytes = bytearray(int(encoded_data[i:i+8], 2) for i in range(0, len(encoded_data), 8))
            f.write(encoded_bytes)

        # Decompress the image
        print(f"Decompressing image for Quality Factor: {qf}...")
        decompressed_image = jpeg_decompression(encoded_data, huffman_tree, num_blocks, image_shape, scaled_quant_matrix)

        # Save decompressed image
        decompressed_file = f'decompressed_qf_{qf}.png'
        cv2.imwrite(decompressed_file, decompressed_image)
        print(f"Decompressed image saved as '{decompressed_file}'.")

        # Calculate RMSE
        rmse = np.sqrt(np.mean((image.astype(np.float32) - decompressed_image.astype(np.float32)) ** 2))
        rmse_values.append(rmse)

        # Calculate BPP (bits per pixel)
        compressed_size = len(open(compressed_file, 'rb').read())
        bpp = compressed_size * 8 / (image.shape[0] * image.shape[1])  # Bits per pixel
        bpp_values.append(bpp)

        print(f"Quality Factor: {qf}, RMSE: {rmse:.2f}, BPP: {bpp:.2f}")

    # Plot RMSE vs. BPP
    plt.figure(figsize=(8, 6))
    plt.plot(bpp_values, rmse_values, marker='o')
    plt.title("RMSE vs. BPP")
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.grid()
    plt.show()

    print("Plot of RMSE vs. BPP generated.")
