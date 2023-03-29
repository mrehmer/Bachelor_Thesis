import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

data_dir = str(input("Enter path of generated images: "))

batch_size = 50

# how the classes are saved in the representation depends on the order
classes = os.listdir(data_dir)
# k is the count, v is the value of classes (so v is the "true" class) and k is the class that the model assigns
classes = {k: v for k,v in enumerate(sorted(classes))}

# defining the image transformation
"""The images are resized to resize_size=[342] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[299]. 
Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]."""
resize_size = 342
crop_size = 299

preprocess = transforms.Compose([
    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

print("Loading images ...")
data = datasets.ImageFolder(root=data_dir, transform=preprocess)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

print("Loading pretrained Inception V3 model...")
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.eval() # evaluation mode

# Move the model to GPU for speed if available
if torch.cuda.is_available():
    model.to('cuda')

# instantiate dictionary with each category and default list to store probabilities for each sample
probs = {}
for key in range(1000):
    probs[str(key)] = []

accuracy = {}
for key in range(1000):
    accuracy[str(key)] = []

class_counter = {}
for key in range(1000):
    class_counter[str(key)] = 0

# batch counter
b = 1

# Iterate over the DataLoader to get batched data
with torch.no_grad():
    for batch in dataloader:
        print(f"Predicting batch {b} / {len(dataloader)}")
        b += 1
        # Get the input tensor from the batch
        x = batch[0]
        # labels in string format
        y = batch[1]
        # Move the input tensor to GPU if available
        if torch.cuda.is_available():
            x = x.to('cuda')
        # Feed the input tensor to the model
        outputs = model(x)

        # Print the output of the model
        for output, i in zip(outputs, y):
            probabilities = torch.nn.functional.softmax(output, dim=0)
            # Check whether the actual label is the top category
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            if top_catid == int(classes[i.item()]):
                accuracy[classes[i.item()]].append(1)
                probs[classes[i.item()]].append(top_prob.item())
            else:
                accuracy[classes[i.item()]].append(0)

# save dict of probs and accuracies to csv
import csv
from itertools import zip_longest

with open('probs.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(probs.keys())
    # replace empty values with zero
    for row in zip_longest(*probs.values(), fillvalue=0):
        writer.writerow(row)

with open('accuracies.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(accuracy.keys())
    for row in zip_longest(*accuracy.values(), fillvalue=0):
        writer.writerow(row)

with open('class_count.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['key', 'value'])
    for key, value in class_counter.items():
        writer.writerow([key, value])
