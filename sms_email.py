import smtplib
from email.message import EmailMessage
from gsecrets import *


# smtp_server=""
# smtp_ssl=True
# smtp_port=""
# smtp_username=""
# smtp_password=""

def loginSendQuit(server, to_email, msg):
    # server.esmtp_features['auth'] = 'LOGIN PLAIN'
    server.login(smtp_username, smtp_password)
    server.sendmail(smtp_username, to_email, msg.as_string())
    server.quit()

def send_email(to_email, subject, message, override=False):
    try:
        msg = EmailMessage()
        msg["From"] = smtp_username
        msg["To"] = to_email
        msg["Subject"] = subject

        body = message
        # (
        #     f" {message.author.global_name}:.\n"
        #     f"Jump to message {message.jump_url}\n"
        #     f"{message.content}"
        # )
        msg.add_header("Content-Type", "text/plain")
        msg.set_payload(body)
        if smtp_ssl ^ override:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                #server.set_debuglevel(True)
                server.starttls()
                self.loginSendQuit(server, to_email, msg)
        else:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                #server.set_debuglevel(True)
                loginSendQuit(server, to_email, msg)

        print(f"Email sent successfully to {to_email}")

    except Exception as e:
        print(f"Failed to send email: {e}")


send_email("3104879662@mypixmessages.com", "WEAPON ALERT", "ALERT: Weapon detected")