from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    Converts uploaded image into MNIST-like format
    """

    # 1️⃣ Load image
    img = Image.open(image_path)

    # 2️⃣ Convert to grayscale
    img = img.convert('L')

    # 3️⃣ Resize to 28x28
    img = img.resize((28, 28))

    # 4️⃣ Convert to NumPy array
    img = np.array(img)

    # 5️⃣ Invert colors if background is white
    # MNIST digits are white on black background
    if img.mean() > 127:
        img = 255 - img

    # 6️⃣ Normalize pixel values
    img = img / 255.0

    # 7️⃣ Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    return img
