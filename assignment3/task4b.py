import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)
import pathlib

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]

# Task 4b
plt.figure(figsize=(15, 6))
for i in range(len(indices)):
    #Plotting filter
    plt.subplot(2,len(indices), i + 1)
    kernel = torch_image_to_numpy(first_conv_layer.weight[indices[i],:,:,:])
    plt.imshow(kernel)
    plt.title(f"Filter {indices[i]}")

    #Plotting activation from corresponding filter
    plt.subplot(2,len(indices),i+len(indices) + 1)
    act_im = torch_image_to_numpy(activation[0,indices[i],:,:])
    plt.imshow(act_im)
    plt.title(f"Activation {indices[i]}")
plot_path = pathlib.Path("plots")    
plt.savefig(plot_path.joinpath(f"task4b.png"))
plt.show()

#Task 4c
#Dont know how to this one ...
