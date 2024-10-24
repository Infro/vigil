import vonage
from datetime import datetime
import _gsecrets

# Twilio configuration (replace with your actual credentials)
client = vonage.Client(key = _gsecrets.vonage_key, secret = _gsecrets.vonage_secret)
sms = vonage.Sms(client)

# Get the current date and time
now = datetime.now()
# Format the date and time as a string
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

basic_alert_message = "ALERT-GUN03"

alert_message = "ALERT-WEAPON DETECTED"

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

responseData = sms.send_message(
    {
        "from": "12137581843",
        "to": "13104879662",
        "text": basic_alert_message,
    }
)

if responseData["messages"][0]["status"] == "0":
    print("Message sent successfully.")
else:
    print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
