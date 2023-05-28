import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import ToTensor

from PIL import Image
import json


# # Load the pretrained model
model = models.resnet50(pretrained=True)
#
# Set the model in evaluation mode
model.eval()

# Load the image
image = Image.open('cat.jpg')

# Calculate the transformations: CNNs expect their input images to be of a certain size, and so this transformation is often necessary.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Apply the transformations and add an extra dimension
image = transform(image).unsqueeze(0)

# Make sure the image is a PyTorch variable and can track gradients
# image = torch.autograd.variable(image, requires_grad=True)
image = torch.tensor(image, requires_grad=True)


# Define the target class (assuming the class for dog is 283)
# This line defines the target class for the adversarial attack.
# In your example, 283 corresponds to the class for a dog in the
# ImageNet dataset.
target_1 = torch.tensor([283])

# Define the loss function
loss_fn_1 = nn.CrossEntropyLoss()

# Forward pass through the model
output = model(image)

# Calculate the loss
loss_1 = loss_fn_1(output, target_1)

# Zero all existing gradients
model.zero_grad()

# Backward pass
loss_1.backward()

# Create adversarial image
epsilon = 0.01
sign_data_grad = image.grad.data.sign()
adversarial_image = image + epsilon * sign_data_grad

# Forward pass through the model using the adversarial image
output = model(adversarial_image)

# Print the top 5 classes predicted by the model
_, indices_1 = torch.topk(output, 5)
print(indices_1)

with open('imagenet_class_index.json') as f:
    idx_to_label = json.load(f)

# Convert the indices to labels
labels = [idx_to_label[str(int(idx))] for idx in indices_1[0]]

print(labels)


# //////////////////////////////////////////////////////////////////////////////

#
# # Load the pretrained model
# model_2 = models.resnet50(pretrained=True)
#
# # Set the model in evaluation mode
# model_2.eval()
#
# # Load the image
# image = Image.open('cat.jpg')
#
# # # Calculate the transformations: CNNs expect their input images to be of a certain size, and so this transformation is often necessary.
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # # Apply the transformations and add an extra dimension
# image = transform(image).unsqueeze(0).to(device)
#
# # # Make sure the image is a PyTorch variable and can track gradients
# image = torch.autograd.Variable(image, requires_grad=True)
# image = torch.tensor(image)
#
# # # Define the target class (assuming the class for dog is 283)
# # # This line defines the target class for the adversarial attack.
# # # In your example, 283 corresponds to the class for a dog in the
# # # ImageNet dataset.
#
# target_i = Image.open('/Users/jordanharris/Code/PycharmProjects/medium/adversarial_ai/csm_Giraffe_Frieda_Tierpark_Berlin__7__fbc128c4e3.jpg')
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to the size expected by your model
#     ToTensor(),  # Convert the PIL Image to a PyTorch tensor
# ])
#
# target_image_2 = transform(target_i).unsqueeze(0).to(device)
#
# target_image_2 = torch.autograd.Variable(target_image_2, requires_grad=True)
# target_image_2 = torch.tensor(target_image_2)
#
# # Define the loss function
# loss_fn_2 = nn.CrossEntropyLoss()
#
# # Forward pass through the model
# output = model_2(target_image_2)
#
# # Calculate the loss
# loss_2 = loss_fn_2(output, target_image_2)
#
# # Zero all existing gradients
# model_2.zero_grad()
#
# # Backward pass
# loss_2.backward()
#
# # Create adversarial image
# epsilon = 0.01
# sign_data_grad = image.grad.data.sign()
# adversarial_image = image + epsilon * sign_data_grad
#
# # Forward pass through the model using the adversarial image
# output_2 = model_2(adversarial_image)
#
#
#
#
# # Print the top 5 classes predicted by the model
# _, indices_2 = torch.topk(output_2, 5)
# print(indices_2)
#
# with open('imagenet_class_index.json') as f:
#     idx_to_label = json.load(f)
#
# # Convert the indices to labels
# labels = [idx_to_label[str(int(idx))] for idx in indices_2[0]]
#
# print(labels)
#
#
# # Yes, you can transform and tensorize your own target image
# # in the same way you did for the input image. However,
# # the target image will not directly act as the "target"
# # in the same way as the target class does in a classification problem.
# # In the context of adversarial attacks, the target typically refers
# # to the desired class that you want the adversarial example to be
# # classified as. For example, if you're trying to get a model to
# # misclassify an image of a cat as a dog, the target would be the
# # class index corresponding to "dog".
# # If you're trying to generate an image that is visually similar to
# # a target image but is classified differently by the model,
# # that's a slightly different problem. It could be framed as an
# # optimization problem where the goal is to find an image that
# # both maximizes the loss of the model (so it's misclassified)
# # and minimizes the difference between the image and the target image
# # (so it's visually similar to the target image).
# # This would involve defining a loss function that incorporates
# # both of these objectives and using a method like gradient descent
# # to find an image that minimizes this loss function.
#
#
#
