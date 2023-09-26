import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from PIL import Image


def dct2(a):
    """
    Compute the 2D Discrete Cosine Transform of the input matrix.
    """
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


def idct2(a):
    """
    Compute the 2D Inverse Discrete Cosine Transform of the input matrix.
    """
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


def pad_image(img):
    """The pad_image function takes an image as input and pads it with zeros
    to ensure that its height and width are divisible by 8."""
    h, w = img.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8

    # Pad image with zeros
    padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), "constant", constant_values=0)
    return padded_img


def quantization_matrix_normal():
    """
    Generate a normal quantization matrix with ones for the top-left 5x5 section.
    """
    q = np.ones((8, 8)) * 99
    q[:5, :5] = 1
    return q


def quantization_matrix_low_freq():
    """
    Generate a quantization matrix that prioritizes low frequencies.
    """
    q = np.ones((8, 8))
    q[:3, :3] = 99
    return q


def quantization_matrix_high_low_freq():
    """
    Generate a quantization matrix that balances between high and low frequencies.
    """
    q = np.ones((8, 8)) * 1
    q[:3, :3] = 99
    q[5:, 5:] = 99
    return q


def compress(img, quantization_func):
    """
    The compress function takes an input image and a quantization
    function as arguments and compresses the image using the
    8x8 Discrete Cosine Transform (DCT) and quantization.
    """

    imsize = img.shape
    dct_transformed = np.zeros(imsize)

    # Apply 8x8 DCT on each block
    compressed = np.zeros(imsize)

    # apply 8x8 DCT on each block and quantize
    quant_matrix = quantization_func()

    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            dct_transformed[i : (i + 8), j : (j + 8)] = dct2(
                img[i : (i + 8), j : (j + 8)]
            )
            compressed[i : (i + 8), j : (j + 8)] = (
                np.round(dct_transformed[i : (i + 8), j : (j + 8)] / quant_matrix)
                * quant_matrix
            )

    return compressed


def decompress(compressed_img):
    """
    Decompress the image using IDCT.
    """
    imsize = compressed_img.shape
    decompressed = np.zeros(imsize)

    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            decompressed[i : (i + 8), j : (j + 8)] = idct2(
                compressed_img[i : (i + 8), j : (j + 8)]
            )
    return decompressed


# load image
img = Image.open("minions.jpg").convert("L")
img = np.array(img)
img = pad_image(img)

compressed_normal = compress(img, quantization_matrix_normal)
decompressed_normal = decompress(compressed_normal)
print(f"compressed_normal:\n {compressed_normal}")
print(f"decompressed_normal:\n {decompressed_normal}\n")

compressed_low_freq = compress(img, quantization_matrix_low_freq)
decompressed_low_freq = decompress(compressed_low_freq)
print(f"compressed_low_freq:\n {compressed_low_freq}")
print(f"decompressed_low_freq:\n {decompressed_low_freq}\n")

compressed_high_low_freq = compress(img, quantization_matrix_high_low_freq)
decompressed_high_low_freq = decompress(compressed_high_low_freq)
print(f"compressed_high_low_freq:\n {compressed_high_low_freq}")
print(f"decompressed_high_low_freq:\n {decompressed_high_low_freq}\n")

# displaying images
fig, axes = plt.subplots(1, 4, figsize=(20, 10))
axes[0].imshow(img, cmap="gray"), axes[0].set_title("Original Image")
axes[1].imshow(decompressed_normal, cmap="gray"), axes[1].set_title(
    "Normal Compression"
)
axes[2].imshow(decompressed_low_freq, cmap="gray"), axes[2].set_title(
    "Low Freq. Compression"
)
axes[3].imshow(decompressed_high_low_freq, cmap="gray"), axes[3].set_title(
    "High & Low Freq. Compression"
)
for ax in axes:
    ax.axis("off")
plt.show()
