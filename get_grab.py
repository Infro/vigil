import requests
import cv2

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np

import os
import sys

filedir_input = "camera-tests-012"
filedir_output = "camera-tests-012/outputs-012"

model_id = "IDEA-Research/grounding-dino-tiny"
# device = "cuda"
device = "cpu"

print ("""
       

       +-----------------------------------+
        VIGIL 
        real time weapons detection system

        v012.20241023
        Gregory Roberts + John Heilman
       +-----------------------------------+

       """)

# user input of objects to detect
searchtext = input("\nType object classes to detect, separated by periods. \n(hit <enter> for default of 'person. face. gun.')\n>>> ")
if searchtext == "":
    searchtext = "gun. person. face."

labels = [value.strip() for value in searchtext.split(".")]

# cue up the AI monster
print("\nPlease wait,\n>>> LOADING NEURAL NET...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

print("\n>>> ACCESSING RTSP VIDEO STREAM...\n")

def rtsp_framegrab(frame):
    # Convert the OpenCV BGR image to RGB format
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array (OpenCV image) to a PIL image
    image = Image.fromarray(rgb_image)

    # Display or save the PIL image
    # image.show()  # To display the image
    # pil_image.save('output_image.png')  # To save it as a file

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
    
    # print tensors array to the screen
    # print(results)

    my_object = results[0]

    # List of tensors
    tensors = my_object["boxes"]
    labels = my_object["labels"]

    annotated_image = annotate_grab(image, tensors, labels)

    annotated_image.show()  # To display the annoated image on screen
    image_save(annotated_image)

    return annotated_image



def annotate_grab(image, tensors, labels):

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

    return image



def image_save(image):

    # Save a serialized image with tensors superimposed
    # into designated output directory

    # Generate 8 random digits as an integer array
    random_digits = np.random.randint(0, 10, size=8)

    # Convert the array of digits to a string
    random_string = ''.join(map(str, random_digits))

    image.save(filedir_output + "/frame_" + random_string + ".png")



# Display the image in an on-screen window
# image.show()
