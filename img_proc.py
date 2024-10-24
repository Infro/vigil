import requests
from datetime import datetime, timedelta
import sms_email
from _gsecrets import *

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np

import os
import sys

filedir_input = "camera-tests-012"
filedir_output = "camera-tests-012/outputs-012"
gun_trigger = 0

model_id = "IDEA-Research/grounding-dino-tiny"
# device = "cuda"
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# ANSI escape sequences for color and blink i nterminal output
# used for weapons notification message
RED = "\033[31m"
GREEN = "\033[32m"
BLINK = "\033[5m"
RESET = "\033[0m"  # Resets all formatting




def rtsp_framegrab(processor, model, frame, searchtext):
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

    if "gun" in labels:
        print(f"\n    {BLINK}{RED}!!! ALERT : WEAPON DETECTED !!!{RESET}")
        
        global gun_trigger
        if gun_trigger == 0:
            print("\n=== STUB : SEND ALERT EMAIL")
            # sms_email.send_alert(smtp_phonealias, sms_email.basic_alert_subject, sms_email.sample_alert_message)
            gun_trigger = 1


        

    annotated_image = annotate_grab(image, tensors, labels)

    # annotated_image.show()  # To display the annoated image on screen
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
    # font = ImageFont.load_default()
    g_labelfontsize = 30
    font = ImageFont.truetype("pricedown.otf", g_labelfontsize)

    # set the padding for the textbox
    x_textpad = 8
    y_textpad = 5
    y_offset = 4
    line_borderthick = 6

    # Superimpose each tensor as a rectangle on the image
    for tensor, label in zip(tensors, labels):

        # Denormalize the tensor coordinates to image dimensions
        x1, y1, x2, y2 = tensor # * torch.tensor([image_width, image_height, image_width, image_height])
        # Draw the rectangle on the image
        draw.rectangle([x1, y1, x2, y2], outline=label_object[label], width=line_borderthick)

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
            [label_x, label_y, label_x + text_width + (x_textpad*2), label_y + text_height + y_textpad],
            # fill=(255, 0, 0, 128)  # 50% transparent red (same as the outline)
            fill=supercolor  # 50% transparent red (same as the outline)
        )
        
        # Draw the text label on top of the semi-transparent background
        draw.text((label_x + x_textpad, label_y - y_offset), label, fill="white", font=font)

    global gun_trigger
    global gtimestamp
    global local_serial_alert_filename

    if gun_trigger == 2:
        print("\n>>> UPLOADING SCREENGRAB TO SECURE CLOUD...")
        # sms_email.send_email_with_image(smtp_phonealias, "VIGIL WEAPONS DETECT", "camera 001 LIVE", image)
        ftp_serial_alert_filename = ftp_remote_path + "_" + gtimestamp + ".jpg"
        sms_email.upload_to_sftp(ftp_server, ftp_username, ftp_password, local_serial_alert_filename, ftp_serial_alert_filename)

        gweb_imagelink = public_web_path +"wd_" + gtimestamp + ".jpg"
        g_fullalert = sample_alert_message_prefix + gweb_imagelink + sample_alert_message_suffix + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sms_email.send_alert(smtp_phonealias, basic_alert_subject, g_fullalert)

        # print("=== STUB : SEND IMAGE LINK")
        # DEBUG: SMS with image link ONLY
        # sms_email.send_alert(smtp_phonealias, "VIGIL Image Link", "click to view:\n" + gweb_imagelink)
        
        print(f"    LINK : {gweb_imagelink}")
        gun_trigger = 99 # no longer used

    if gun_trigger == 1:
        print("\n>>> SAVING SCREENGRAB TO LOCAL ARCHIVE...")
        # sms_email.send_email_with_image(smtp_phonealias, "VIGIL WEAPONS DETECT", "camera 001 LIVE", image)
        gtimestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        local_serial_alert_filename = ftp_local_path + "_" + gtimestamp + ".jpg"
        sms_email.save_image_locally(image, local_serial_alert_filename, quality=70)
        gun_trigger = 2

    return image



def image_save(image):

    # Save a serialized image with tensors superimposed
    # into designated output directory

    # Get the current time
    now = datetime.now()

    # Format the datetime object as a string in the desired format
    formatted_time = now.strftime("%Y%m%d-%H%M%S")

    # Generate 8 random digits as an integer array
    random_digits = np.random.randint(0, 10, size=3)

    # Convert the array of digits to a string
    frame_id_str = formatted_time + "-" + ''.join(map(str, random_digits))

    image.save(filedir_output + "/frame_" + frame_id_str + ".png")



# Display the image in an on-screen window
# image.show()
