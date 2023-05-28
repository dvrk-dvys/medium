from PIL import Image
import numpy as np
import qrcode

# Create a QR code
data = "This is the data to be encoded into the QR code"
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(data)
qr.make(fit=True)
qr_img = qr.make_image(fill='black', back_color='white')

# Convert the QR code into a numpy array
qr_array = np.array(qr_img) * 255  # QR code is binary, so we multiply by 255 to get the same range as the image
qr_array = qr_array.astype('uint8')  # Cast to the same type as the image

# Load the image
img = Image.open('/Users/jordanharris/Code/PycharmProjects/medium/adversarial_ai/cat.jpg')
img_array = np.array(img)

# Resize the QR code to match the size of the image
qr_img_resized = qr_img.resize((img_array.shape[1], img_array.shape[0]))
qr_array = np.array(qr_img_resized) * 255  # QR code is binary, so we multiply by 255 to get the same range as the image
qr_array = qr_array.astype('float64')  # Cast to the same type as the image

# Convert the QR code to a 3-channel image
qr_array = np.stack([qr_array, qr_array, qr_array], axis=-1)

# Embed the QR code into the image
epsilon = 0.085 # This is the strength of the noise
img_array = img_array.astype('float64')  # Cast image array to float64
img_array_noise = img_array.copy()

# Generate a random index array
np.random.seed(0)  # For reproducibility
index_array = np.random.choice(a=[False, True], size=img_array.shape, p=[0.99, 0.01])  # 1% of pixels will be perturbed

# Apply the perturbations
img_array_noise[index_array] = img_array_noise[index_array] + epsilon * qr_array[index_array]

# Cast the noisy image array back to 'uint8'
img_array_noise = img_array_noise.astype('uint8')

# Save the image with the embedded QR code
img_noise = Image.fromarray(img_array_noise)
img_noise.save('path_to_save_your_noisy_image.jpg')

# extract

# Load the original and noisy images
img_path = '/Users/jordanharris/Code/PycharmProjects/medium/adversarial_ai/cat.jpg'
img_noise_path = '/Users/jordanharris/Code/PycharmProjects/medium/path_to_save_your_noisy_image.jpg'

img = Image.open(img_path)
img_noise = Image.open(img_noise_path)

# Convert the images to numpy arrays
img_array = np.array(img)
img_noise_array = np.array(img_noise)

# Subtract the original image from the noisy image to get the QR code
qr_array_ = (img_noise_array - img_array) / 0.01  # Divide by epsilon to reverse the scaling

# Convert the QR code array back to an 8-bit image
qr_array_ = qr_array_.astype('uint8')

# Save the QR code as a separate image
qr_img = Image.fromarray(qr_array_)
qr_img.save('path_to_save_your_qr_code.jpg')