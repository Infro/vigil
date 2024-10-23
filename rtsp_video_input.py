import cv2
import get_grab
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load the variables from .env
load_dotenv()

# Replace with your RTSP URL
rtsp_url = 'rtsp://admin:vanguard2024@192.168.1.68:554'

now = datetime.now()

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)
# cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)


if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        # Capture frame-by-frame
        # ret, frame = cap.read()
        # Capture the latest frame
        print("\n>>> DSKY.AI VISION ALGORITHM RUNNING...") 
        prevtime = now
        now = datetime.now()
        protime = now - prevtime
        print("    FRAME : " + now.strftime("%Y-%m-%d %H:%M:%S"))
        print("    tDELTA : " + str(protime.microseconds) + " ms")

        # Skip frames until only the most recent one is available
        # while cap.grab():
        #     prevtime = now
        #     now = datetime.now()
        #     if prevtime - now > timedelta(seconds=1):
        #         date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        #         print(date_time_str)
        #     else:
        #         print("x",end="", flush=True)
        #         break
        #     pass

        fc=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fc)

        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the resulting frame
        # cv2.imshow('RTSP Stream', frame)
        image = get_grab.rtsp_framegrab(frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()