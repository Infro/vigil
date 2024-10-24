from twilio.rest import Client
from datetime import datetime
from _gsecrets import *

# Twilio configuration (replace with your actual credentials)
account_sid = twilio_account_sid
auth_token = twilio_auth_token
client = Client(account_sid, auth_token)

# Get the current date and time
now = datetime.now()
# Format the date and time as a string
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

sample_alert_message = """
    !!ALERT!! WEAPON DETECTED\n
    =======================\n
    ADDRESS: 1521 North Highland Ave.,\n
        Los Angeles, CA 90028\n
    LOCATION: SE entrance\n 
        (school auditorium)\n
    \n
    DESCRIP: black hoodie, backpack\n
    WEAPON-TYPE: assault rifle\n
    \n
    MAPLINK: https://www.google.com/maps/search/?api=1&query=34.098248,-118.340965\n
    \n
    VIGIL: Hollywood_HS_LA.8693\n
    GEOSTAMP: 34.098248, -118.340965\n
    TIMESTAMP: """ + date_time_str + "\n[PST:UTC-08:00]"

# TODO: get this to send to three number simultaneously: me, allen, john

def send_sms_alert():
    message = client.messages.create(
        body =sample_alert_message,
        from_='+18447917232',  # Twilio phone number
        to='+13104879662'  # Your phone number
    )
    print(f"SMS sent: {message.sid}")

# now, actually send it:
send_sms_alert()

