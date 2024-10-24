import requests

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import os
import sys

filedir_input = "weapon-tests-002"
filedir_output = "weapon-tests-002/outputs-002"

model_id = "IDEA-Research/grounding-dino-tiny"
# device = "cuda"
device = "cpu"

# pull the file list from the designated input directory
files = sorted(os.listdir(path = filedir_input))
for index, file in enumerate(files):
    print(f"{index}: {file}")

# user selection of file from directory
# FIXME: make sure not to show recursvice up or subdirectories (FILES ONLY)
# FIXME: do not show AVIFs, they crash the image-interp engine
# TODO: hit <enter> to batch process entire directory

selection = int(input("\nEnter the file number, then <enter>: "))
img_name = files[selection]
img_url = filedir_input + "/" + img_name

# alternate (original) method: get image from web
# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)

# user input of objects to detect
searchtext = input("\nEnter objects to detect, \nseparated by periods (hit \n<enter> for default \nof 'person. face. gun.'): ")
if searchtext == "":
    searchtext = "gun. person. face."

labels = [value.strip() for value in searchtext.split(".")]

# Check for cats and remote controls
# text = "gun. person. face."
image = Image.open(img_url).convert("RGB")

# cue up the monster
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

inputs = processor(images=image, text=searchtext, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
print(results)

my_object = results[0]

# List of tensors
tensors = my_object["boxes"]
labels = my_object["labels"]

# Convert the image size to get the dimensions
image_width, image_height = image.size

# Create a drawing context
draw = ImageDraw.Draw(image)

# TODO: this is explicit for the default search.
# TODO: make this modular for any arbitrary search objects
# TODO: also, use RGB tuples instead of word values

label_object = {
    "person" : "blue",
    "gun" : "red",
    "face" : "green",
}

# Define a font
# TODO: a better, bolder font
font = ImageFont.load_default()

# set the padding for the textbox
x_textpad = 3
y_textpad = 5

# Superimpose each tensor as a rectangle on the image
for tensor, label in zip(tensors, labels):

    # Denormalize the tensor coordinates to image dimensions
    x1, y1, x2, y2 = tensor # * torch.tensor([image_width, image_height, image_width, image_height])
    # Draw the rectangle on the image
    draw.rectangle([x1, y1, x2, y2], outline=label_object[label], width=3)

    # Get the text size using textbbox (returns the bounding box of the text)
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    label_x = x1  # Align label's top-left corner with bounding box's x1
    label_y = y1 - text_height - y_textpad  # 5 pixels above the bounding box
    
    # TODO: get this shit working around line 78 so we don't need this IF/THEN 

    # map text colors to RGB values
    if label_object[label] == "red" :
        supercolor = (255, 0, 0, 128)
    if label_object[label] == "green" :
        supercolor = (0, 166, 0, 128)
    if label_object[label] == "blue" :
        supercolor = (0, 0, 255, 128)

    # Draw the semi-transparent background for the label
    # TODO: make it actually semi-trans. currently it renders opaque

    draw.rectangle(
        [label_x, label_y, label_x + text_width + 6, label_y + text_height + 5],
        # fill=(255, 0, 0, 128)  # 50% transparent red (same as the outline)
        fill=supercolor  # 50% transparent red (same as the outline)
    )
    
    # Draw the text label on top of the semi-transparent background
    draw.text((label_x + 3, label_y), label, fill="white", font=font, font_size=24)

    # Position for the label text (top-left corner of the bounding box)
    # label_position = (x1, y1)
    # Draw the label text
    # draw.text(label_position, label, fill="white", font_size=24) # stroke_width=1


# Save the image with tensors superimposed
# into designated output directory
image.save(filedir_output + "/" + img_name)

# Display the image in an on-screen window
image.show()
