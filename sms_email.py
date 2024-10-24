import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

from ftplib import FTP
import paramiko

from PIL import Image
import io

from datetime import datetime
from _gsecrets import *


# Get the current date and time
now = datetime.now()
# Format the date and time as a string
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

tYELLOW = "\033[33m"
tRESET = "\033[0m"  # Resets all formatting


# a quick and dirty SMTP mail server login
def loginSendQuit(server, to_email, msg):
    # server.esmtp_features['auth'] = 'LOGIN PLAIN'
    server.login(smtp_username, smtp_password)
    server.sendmail(smtp_username, to_email, msg.as_string())
    server.quit()



def send_alert(to_email, subject, message, override=False):
    try:
        msg = EmailMessage()
        msg["From"] = smtp_username
        msg["To"] = to_email
        msg["Subject"] = subject

        body = message

        msg.add_header("Content-Type", "text/plain")
        msg.set_payload(body)
        if smtp_ssl ^ override:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                # server.set_debuglevel(True)
                server.starttls()
                loginSendQuit(server, to_email, msg)
        else:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                #server.set_debuglevel(True)
                loginSendQuit(server, to_email, msg)

        print(f"\n+++ SMS successfully transmitted:\n{tYELLOW}    {alert_user} @ {alert_phone}{tRESET}")

    except Exception as e:
        print(f"Failed to send : {e}")



# Compress as JPG and prepare the image for FTP
def compress_image(pil_image, quality=70):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=quality)
    img_byte_arr.seek(0)
    return img_byte_arr


# 1. Save an image to the local file system as JPG
def save_image_locally(pil_image, save_path, quality=70):
    pil_image.save(save_path, format="JPEG", quality=quality)
    print(f"    Image saved locally as\n    {save_path}")
    return save_path


# 2. Upload the image to the FTP server
def upload_to_ftp(ftp_host, ftp_user, ftp_pass, local_file_path, remote_file_path):
    with open(local_file_path, "rb") as file:
        # Connect to FTP server
        ftp = FTP(ftp_host)
        ftp.login(user=ftp_user, passwd=ftp_pass)
        # Upload the file
        ftp.storbinary(f'STOR {remote_file_path}', file)
        ftp.quit()
        print(f"Uploaded {local_file_path} to FTP server at {ftp_host}/{remote_file_path}")

# 3. upload port 22 using SFTP
# Create SFTP client and connect to server
def upload_to_sftp(ftp_host, ftp_user, ftp_pass, local_file_path, remote_file_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the SFTP server
        client.connect(ftp_host, port=ftp_port, username=ftp_user, password=ftp_pass)

        # Open the SFTP session
        sftp = client.open_sftp()

        # DEBUG: list files in the current directory
        # print(sftp.listdir('.'))

        # Upload a file
        sftp.put(local_file_path, remote_file_path)

        # Close the SFTP session
        sftp.close()

    finally:
        # Close the SSH client
        client.close()


# Send email with image attachment
def send_email_with_image(to_email, subject, message_text, pil_image, override=False):
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the text message
        msg.attach(MIMEText(message_text, 'plain'))

        # Compress the image and get the byte stream
        img_byte_arr = compress_image(pil_image)

        # Attach the image as a MIMEImage object
        image_mime = MIMEImage(img_byte_arr.getvalue(), _subtype="jpeg", name="compressed_image.jpg")
        msg.attach(image_mime)

        try:
            # Use smtplib to send the email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(smtp_username, smtp_password)
                server.sendmail(smtp_username, to_email, msg.as_string())  # Correct: msg.as_string()
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")

        # msg.add_header("Content-Type", "multipart/mixed")
        # msg.set_payload("test")

        # if smtp_ssl ^ override:
        #     with smtplib.SMTP(smtp_server, smtp_port) as server:
        #         server.set_debuglevel(True)
        #         server.starttls()
        #         loginSendQuit(server, to_email, msg.as_string())
        # else:
        #     with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        #         #server.set_debuglevel(True)
        #         loginSendQuit(server, to_email, msg.as_string())

        print(f"\n+++ SCREENGRAB transmitted:\n{tYELLOW}    {alert_user} @ {alert_phone}{tRESET}")

    except Exception as e:
        print(f"Failed to send IMAGE : {e}")


