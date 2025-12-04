from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QRadioButton,
    QMessageBox, QDateEdit,QFrame,QSizePolicy,QListWidget,
    QListWidgetItem,QDialog,QCheckBox,QScrollArea,QStackedLayout,
    QComboBox, QTextEdit, QDateEdit,QGridLayout,QTimeEdit,
    QTabWidget,QMenu,QMessageBox,QSpinBox,QDialogButtonBox,
    QButtonGroup,QGroupBox,QTextBrowser, QStyle,QGraphicsOpacityEffect,
    QFileDialog,QGraphicsDropShadowEffect,QInputDialog)


from PyQt6.QtCore import(
    Qt, QDate, QSize,QPropertyAnimation,
    QRect,QPoint,QEasingCurve,QPropertyAnimation,
    QTimer,QTime,pyqtSignal,pyqtProperty,
    QEvent,QUrl,QRectF, pyqtSignal,
    QPointF,QObject)


from PyQt6.QtGui import (
    QFont, QPalette, QColor, QLinearGradient, QBrush,
    QIcon,QPixmap, QPainter, QPainterPath,QGuiApplication,
    QPainter,QAction,QPainter, QPolygon,QIntValidator,
    QDesktopServices,QPen, QFontMetrics,QMovie,
    QRegion,QImage, QPolygonF,QKeyEvent)


from PyQt6.QtSvgWidgets import QSvgWidget

import sys
import os
import json


import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

import pycountry
import phonenumbers

from pathlib import Path
from datetime import datetime, timedelta,timezone
from collections import defaultdict

import re

import winsound
import requests
from fonts import load_sf_pro_font


from math import cos, sin, pi


import numpy as np
import cv2
import sys
import time
import cvzone
from cvzone.PlotModule import LivePlot
import ctypes

import subprocess, threading
from flask import Flask, request, render_template_string
import resend
import uuid
import socket
from dotenv import load_dotenv

import urllib.parse
import webbrowser
import traceback

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import random


import firebase_admin
from google.cloud.firestore import SERVER_TIMESTAMP, FieldFilter
from firebase_admin import credentials, firestore,initialize_app,storage

from cryptography.fernet import Fernet
import base64
import tempfile

import hashlib
import logging

from typing import Any, Dict
import hashlib
app = QApplication(sys.argv)


def resource_path(relative_path):
    """Returns the relative path for non-bundled (normal) setup or bundled exe."""
    try:

        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)

        return os.path.join(os.path.abspath("."), relative_path)
    except Exception as e:
        print(f"Error with resource path: {e}")
        return relative_path
def get_image_path(path):
    if path.startswith("assets/") or path.startswith("assets\\"):
        return resource_path(path)
    else:
        base_path = os.path.abspath(".")
        return os.path.join(base_path, path)
load_dotenv(resource_path("assets/api.env"))

cred = credentials.Certificate(resource_path("assets/fitnesstrackerchat-firebase-adminsdk-fbsvc-5fd88409bd.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

REMEMBER_ME_FILE = os.path.join(tempfile.gettempdir(), ".remember_me")

ENCRYPTION_KEY = base64.urlsafe_b64encode(
    b"\x8f\x0c\xb3\xd9\x14\xc6\x8a\xc5\xd2\x9b\xaf\xf6\xee\xe4\x1d\xaa"
    b"\x3a\x98\x17\xf4\x06\xb9\xdb\x88\x7e\xc1\xae\x33\x11\xdd\x99\xfe"
)


def get_chat_folder_path(user1: str, user2: str) -> Path:

    sorted_users = sorted([user1, user2])
    base_dir = Path("chats") / f"{sorted_users[0]}_{sorted_users[1]}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def get_chat_file_path(user1: str, user2: str) -> Path:
    folder = get_chat_folder_path(user1, user2)
    return folder / "messages.json"

def load_local_chat(user1: str, user2: str) -> list:
    path = get_chat_file_path(user1, user2)
    if path.exists():
        try:
            messages = json.loads(path.read_text())
            return messages if isinstance(messages, list) else []
        except Exception as e:
            print(f"Failed to load local chat: {e}")
            return []
    return []

def save_local_chat(user1: str, user2: str, messages: list) -> None:
    path = get_chat_file_path(user1, user2)
    try:
        path.write_text(json.dumps(messages, indent=2))
    except Exception as e:
        print(f"Failed to save local chat: {e}")



def show_non_blocking_dialog(dlg: QDialog):
    dlg.exec()


with open(resource_path("assets/phone_formats.json"), "r") as f:
    PHONE_FORMATS = json.load(f)


sf_families = load_sf_pro_font()
if not sf_families:
    print("❌ Failed to load SF Pro fonts.")
    sf_family = "Sans Serif"
else:
    sf_family = sf_families[0]


if sys.platform.startswith('win'):


    def set_window_always_on_top(window_name="Heart Rate Monitor"):
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0,
                                              0x0001 | 0x0002)
        else:
            print(f"Window '{window_name}' not found.")
else:
    def set_window_always_on_top(window_name="Heart Rate Monitor"):
        pass


MAX_INPUT_WIDTH = 600


def format_date_label(dt: datetime):
    today = datetime.now().date()
    msg_date = dt.date()

    if msg_date == today:
        return "Today"
    elif msg_date == today - timedelta(days=1):
        return "Yesterday"
    else:
        return dt.strftime("%d %b %Y").lstrip("0")


def get_local_ip():
    """Get the LAN IP address of the current machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:

        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

USER_DATA_PATH = Path("user_data.json")
SERVER_MIRROR_PATH = Path("server_data.json")
FIREBASE_BUCKET_NAME = "bustrackerapp-gg3esw.appspot.com"


logging.basicConfig(filename="sync.log", level=logging.INFO, format="%(asctime)s - %(message)s")


cred2 = credentials.Certificate(resource_path("assets/bustrackerapp-gg3esw-firebase-adminsdk-qs0hb-b02ac93352.json"))
firebase_app2 = initialize_app(cred2, {
    "storageBucket": FIREBASE_BUCKET_NAME
}, name="app2")

bucket = storage.bucket(app=firebase_app2)


def parse_timestamp(ts: Any) -> float:
    """Parse a timestamp field to Unix epoch float. Supports int/float or ISO8601 string."""
    if ts is None:
        return 0
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except Exception:
        return 0

def deep_merge(local: Dict[str, Any], remote: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts (remote into local), preferring remote values if they are newer.
    Assumes dicts may contain nested dicts.
    """
    merged = dict(local)

    for k, remote_val in remote.items():
        local_val = merged.get(k)

        if isinstance(remote_val, dict) and isinstance(local_val, dict):
            merged[k] = deep_merge(local_val, remote_val)
        else:

            if k == "last_updated":
                merged[k] = max(parse_timestamp(local_val), parse_timestamp(remote_val))
            else:
                remote_ts = parse_timestamp(remote.get("last_updated"))
                local_ts = parse_timestamp(local.get("last_updated"))
                if remote_ts >= local_ts:
                    merged[k] = remote_val
                else:
                    merged[k] = local_val

    return merged

def merge_user_data(local_path: Path, downloaded_path: Path):
    if not local_path.exists() and not downloaded_path.exists():
        return

    local_users = []
    downloaded_users = []

    if local_path.exists():
        with open(local_path, "r") as f:
            local_users = json.load(f)

    if downloaded_path.exists():
        with open(downloaded_path, "r") as f:
            downloaded_users = json.load(f)

    local_dict = {user["username"]: user for user in local_users}
    downloaded_dict = {user["username"]: user for user in downloaded_users}


    for username, remote_user in downloaded_dict.items():
        if username in local_dict:
            local_user = local_dict[username]
            merged_user = deep_merge(local_user, remote_user)
            local_dict[username] = merged_user
        else:
            local_dict[username] = remote_user


    merged_users = list(local_dict.values())

    temp_path = local_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(merged_users, f, indent=2)
    temp_path.replace(local_path)

def file_md5(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def upload_if_changed():
    last_md5 = None
    while True:
        try:
            current_md5 = file_md5(USER_DATA_PATH)
            if current_md5 and current_md5 != last_md5:
                blob = bucket.blob("user_data.json")
                blob.upload_from_filename(USER_DATA_PATH)
                logging.info("Uploaded user_data.json to Firebase.")
                last_md5 = current_md5
        except Exception as e:
            logging.error(f"Upload error: {e}")
        time.sleep(5)


def download_if_updated():
    last_md5 = None
    while True:
        try:
            blob = bucket.blob("user_data.json")
            blob.reload()
            remote_md5 = blob.md5_hash
            if remote_md5 != last_md5:
                blob.download_to_filename(SERVER_MIRROR_PATH)
                logging.info("Downloaded updated user_data.json → server_data.json.")


                merge_user_data(USER_DATA_PATH, SERVER_MIRROR_PATH)
                logging.info("Merged downloaded data into local user_data.json.")

                last_md5 = remote_md5
        except Exception as e:
            logging.error(f"Download error: {e}")
        time.sleep(5)


def run_verification_server():
    app = Flask(__name__)

    VERIFY_SUCCESS_HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Email Verification Success</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap" rel="stylesheet" />
      <style>
        body {
          margin: 0; padding: 0;
          font-family: 'Inter', sans-serif;
          background-color: #f9f9f9;
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100vh;
          text-align: center;
          color: #222;
        }
        .container {
          background: white;
          padding: 40px 50px;
          border-radius: 16px;
          box-shadow: 0 5px 25px rgba(0,0,0,0.1);
          max-width: 460px;
          width: 90%;
        }
        img.logo {
          width: 120px;
          margin-bottom: 24px;
        }
        h1 {
          font-weight: 700;
          font-size: 28px;
          margin-bottom: 10px;
          color: #0057ff;
        }
        p {
          font-weight: 500;
          font-size: 16px;
          margin: 8px 0 24px;
          color: #444;
        }
        .footer {
          font-size: 13px;
          color: #888;
          margin-top: 30px;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <img class="logo" src="https://cea02b5d-5e80-453d-a5af-9e1ff603f2f8.b-cdn.net/e/c32c15ed-b667-4f03-94c7-214e30ee4315/c08619cc-1b8d-40da-836c-cff8c6a88900.png" alt="FitnessTracker Logo" />
        <h1>Congratulations, {{first_name}} {{last_name}}!</h1>
        <p>Your email has been successfully verified.</p>
        <p>You can now safely close this window and continue using the FitnessTracker app.</p>
        <div class="footer">
          If you did not request this verification, please ignore this message.
        </div>
      </div>
    </body>
    </html>
    """

    USER_DATA_PATH = Path("user_data.json")


    @app.route("/verify")
    def verify():
        try:
            username = request.args.get("username")
            token = request.args.get("token")
            print(f"Verify called with username={username}, token={token}")

            with open(USER_DATA_PATH, "r") as f:
                users = json.load(f)


            for user in users:
                if user["username"] == username and user.get("verification_token") == token:
                    user["verified"] = True
                    user.pop("verification_token", None)
                    USER_DATA_PATH.write_text(json.dumps(users, indent=2))
                    return render_template_string(
                        VERIFY_SUCCESS_HTML,
                        first_name=user.get("first_name", ""),
                        last_name=user.get("last_name", "")
                    )

            return "<h1>Invalid or expired verification link.</h1>"
        except Exception as e:

            traceback.print_exc()
            return f"<h1>Server error: {e}</h1>"


    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False),
        daemon=True
    ).start()

    threading.Thread(target=upload_if_changed, daemon=True).start()
    threading.Thread(target=download_if_updated, daemon=True).start()

def launch_verification_server():
    script_path = os.path.abspath(__file__)
    subprocess.Popen(
        [sys.executable, script_path, "--verify-server"]

    )


def send_verification_email(user_email, username):
    token = str(uuid.uuid4())

    with open("user_data.json", "r") as f:
        users = json.load(f)

    user = next((u for u in users if u["username"] == username), None)
    if not user:
        return False

    user["verification_token"] = token
    Path("user_data.json").write_text(json.dumps(users, indent=2))

    first_name = user.get("first_name", "")
    last_name = user.get("last_name", "")

    try:
        with open(resource_path("assets/verification/verificationemail.html"), "r", encoding="utf-8") as f:
            html_template = f.read()
    except FileNotFoundError:
        print("Email template not found.")
        return False

    ip_address = get_local_ip()
    verification_link = f"http://{ip_address}:8000/verify?username={username}&token={token}"

    html_content = html_template.replace("{firstname}", first_name)\
                                .replace("{lastname}", last_name)\
                                .replace('href="#"', f'href="{verification_link}"')


    sender_email = os.getenv("GMAIL_EMAIL")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Verify your email for FitnessTracker"
        msg["From"] = f"FitnessTracker <{sender_email}>"
        msg["To"] = user_email

        part = MIMEText(html_content, "html")
        msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, msg.as_string())

        return True

    except Exception as e:
        traceback.print_exc()
        print("Failed to send email via Gmail:", e)
        return False


def send_2fa_email(user_email, username, code, first_name, last_name):
    try:
        with open(resource_path("assets/verification/twofactoremail.html"), "r", encoding="utf-8") as f:
            html_template = f.read()
    except FileNotFoundError:
        print("2FA email template not found.")
        return False


    html_content = html_template \
        .replace("{{firstname}}", first_name) \
        .replace("{{lastname}}", last_name)

    for i in range(len(code)):
        html_content = html_content.replace(f"{{{{code[{i}]}}}}", code[i])

    html_content = html_content.replace("{{code}}", code)


    sender_email = os.getenv("GMAIL_EMAIL")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your 2FA Login Code"
        msg["From"] = f"FitnessTracker <{sender_email}>"
        msg["To"] = user_email

        part = MIMEText(html_content, "html")
        msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, msg.as_string())

        return True

    except Exception as e:

        traceback.print_exc()
        print("Failed to send 2FA email via Gmail:", e)
        return False

def check_password_strength(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least one number."
    if not re.search(r"[!@#$%^&*()_+=\[{\]};:<>|./?,-]", password):
        return False, "Password must include at least one special symbol."
    return True, "Strong password."

def set_opencv_window_icon(window_title, icon_path):

    user32 = ctypes.windll.user32


    hwnd = user32.FindWindowW(None, window_title)
    if hwnd == 0:
        print(f"Window '{window_title}' not found!")
        return

    hicon = user32.LoadImageW(
        0,
        icon_path,
        1,
        0, 0,
        0x00000010
    )
    if hicon == 0:
        print("Failed to load icon")
        return
    WM_SETICON = 0x80
    ICON_SMALL = 0
    ICON_BIG = 1


    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)

    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)



NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_APP_KEY = os.getenv("NUTRITIONIX_APP_KEY")

nutritionix_headers = {
    "x-app-id": NUTRITIONIX_APP_ID,
    "x-app-key": NUTRITIONIX_APP_KEY,
    "Content-Type": "application/json"
}



modelFile = resource_path("assets/dnn/opencv_face_detector_uint8.pb")
configFile = resource_path("assets/dnn/opencv_face_detector.pbtxt")

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def detect_faces_dnn(frame, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            w1, h1 = x2 - x1, y2 - y1
            faces.append((x1, y1, w1, h1))
    return faces

def webcam_heartbeat_monitor():
    realWidth = 640
    realHeight = 480
    videoWidth = 160
    videoHeight = 120
    videoChannels = 3
    videoFrameRate = 15
    set_top_called = False


    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
    webcam.set(cv2.CAP_PROP_FPS, videoFrameRate)

    levels = 3
    alpha = 170
    minFrequency = 1.0
    maxFrequency = 2.0
    bufferSize = 150
    bufferIndex = 0

    plotY = LivePlot(realWidth, realHeight, [60, 120], invert=True)

    def buildGauss(frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

    font = cv2.FONT_HERSHEY_SIMPLEX
    loadingTextLocation = (30, 40)
    bpmTextLocation = (videoWidth // 2, 40)

    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))

    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    bpmCalculationFrequency = 10
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))

    i = 0
    ptime = 0
    ftime = 0

    start_time = time.time()
    bpm_values = []
    bpm_smoothed = None
    smoothing_alpha = 0.15

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frameDraw = frame.copy()
        ftime = time.time()
        fps = 1 / (ftime - ptime) if (ftime - ptime) > 0 else 0
        ptime = ftime

        elapsed = time.time() - start_time
        time_left = max(0, int(60 - elapsed))

        cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        faces = detect_faces_dnn(frameDraw)

        if len(faces) > 0:
            x1, y1, w1, h1 = faces[0]
            cv2.rectangle(frameDraw, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

            videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)
            fourierTransform[~mask] = 0

            if bufferIndex % bpmCalculationFrequency == 0:
                i += 1
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm = 60.0 * hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            filtered = np.real(np.fft.ifft(fourierTransform, axis=0)) * alpha
            filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
            outputFrame = cv2.convertScaleAbs(detectionFrame + filteredFrame)

            bufferIndex = (bufferIndex + 1) % bufferSize
            outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
            frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

            bpm_value = bpmBuffer.mean()

            if i == bpmBufferSize + 1:
                bpm_values = []

            if i > bpmBufferSize:
                bpm_values.append(bpm_value)



            if bpm_smoothed is None:
                bpm_smoothed = bpm_value
            else:
                bpm_smoothed = smoothing_alpha * bpm_value + (1 - smoothing_alpha) * bpm_smoothed

            imgPlot = plotY.update(float(bpm_smoothed))

            if i > bpmBufferSize:
                cvzone.putTextRect(frameDraw, f'BPM: {int(bpm_smoothed)}', bpmTextLocation, scale=2)
            else:
                cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)

            imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)
        else:
            imgPlot = plotY.update(0)
            imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)

        cv2.putText(
            imgStack,
            f'Time Left: {time_left}s',
            (realWidth - 180, realHeight - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Heart Rate Monitor", imgStack)

        if not set_top_called:
            set_window_always_on_top("Heart Rate Monitor")
            set_opencv_window_icon("Heart Rate Monitor", resource_path("assets/heart.ico"))
            set_top_called = True

        key = cv2.waitKey(1)
        if key == ord('q') or time_left <= 0:
            break

    webcam.release()
    cv2.destroyAllWindows()

    return int(np.mean(bpm_values)) if bpm_values else -1

class ProfilePictureDialog(QDialog):
    def __init__(self, username, parent=None):
        super().__init__(parent)
        print("DEBUG: ProfilePictureDialog username =", username)
        self.setWindowTitle("Edit Profile Picture")
        self.setWindowIcon(QIcon(resource_path("assets/edit3.png")))
        self.setFixedSize(400, 520)
        self.setStyleSheet("background-color: #121212; color: white;")

        self.crop_size = 190
        self.username = username
        self.user_dir = os.path.join("user_images", self.username)
        os.makedirs(self.user_dir, exist_ok=True)


        self.original_path = os.path.join(self.user_dir, "profile_original.png")
        self.cropped_path = os.path.join(self.user_dir, "profile_cropped.png")
        self.state_path = os.path.join(self.user_dir, "state.json")


        if os.path.exists(self.original_path):
            self.original_pixmap = QPixmap(self.original_path)
        else:
            self.original_pixmap = QPixmap(400, 400)
            self.original_pixmap.fill(Qt.GlobalColor.gray)

        layout = QVBoxLayout(self)


        self.saved_state = {"scale_factor": None, "offset": None}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    self.saved_state = json.load(f)
            except Exception as e:
                print("Failed to load crop state:", e)


        self.crop_widget = CropWidget(self.crop_size, self.original_pixmap)
        layout.addWidget(self.crop_widget, alignment=Qt.AlignmentFlag.AlignHCenter)


        if self.saved_state.get("scale_factor") is not None and self.saved_state.get("offset") is not None:
            self.crop_widget.scale_factor = self.saved_state["scale_factor"]
            offset = self.saved_state["offset"]
            self.crop_widget.offset = QPoint(offset[0], offset[1])
            self.crop_widget.update_scaled_pixmap()
            self.crop_widget.update()


        btn_layout = QGridLayout()

        self.change_btn = QPushButton(" Change")
        self.change_btn.setIcon(QIcon("assets/edit4.png"))
        self.change_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.change_btn.setStyleSheet("""
            QPushButton {
                background-color: #005792;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 11pt;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #004060;
            }
        """)

        self.remove_btn = QPushButton(" Remove")
        self.remove_btn.setIcon(QIcon("assets/delete.png"))
        self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 11pt;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
        """)

        self.webcam_btn = QPushButton(" Use Webcam")
        self.webcam_btn.setIcon(QIcon("assets/camera.png"))
        self.webcam_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.webcam_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 11pt;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #115293;
            }
        """)



        self.save_btn = QPushButton(" Save")
        self.save_btn.setIcon(QIcon("assets/save.png"))
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #388e3c;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 11pt;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2e7d32;
            }
        """)

        btn_layout.addWidget(self.change_btn, 0, 0)
        btn_layout.addWidget(self.remove_btn, 0, 1)
        btn_layout.addWidget(self.webcam_btn, 1, 0)
        btn_layout.addWidget(self.save_btn, 1, 1)

        layout.addLayout(btn_layout)


        self.change_btn.clicked.connect(self.load_new_image)
        self.remove_btn.clicked.connect(self.remove_picture)
        self.webcam_btn.clicked.connect(self.open_webcam_dialog)
        self.save_btn.clicked.connect(self.save_cropped_image)



        self.pictureChanged = None
        self.pictureRemoved = None

    def load_new_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select New Profile Picture", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            self.original_pixmap = QPixmap(filename)
            self.crop_widget.set_pixmap(self.original_pixmap)


            self.crop_widget.reset_offset_and_scale()

    def remove_picture(self):
        default_path = resource_path("assets/defaultprofile.png")
        default_pixmap = QPixmap(default_path)

        self.original_pixmap = default_pixmap
        self.crop_widget.set_pixmap(self.original_pixmap)

        self.original_pixmap.save(self.original_path, "PNG")
        default_pixmap.save(self.cropped_path, "PNG")

        if os.path.exists(self.state_path):
            try:
                os.remove(self.state_path)
            except Exception as e:
                print("Failed to remove crop state file:", e)

        if self.pictureRemoved:
            self.pictureRemoved()

        if self.pictureChanged:
            self.pictureChanged(self.cropped_path, None)

        self.accept()


    def save_cropped_image(self):

        self.original_pixmap.save(self.original_path, "PNG")


        cropped = self.crop_widget.get_cropped_pixmap()
        cropped.save(self.cropped_path, "PNG")


        state = {
            "scale_factor": self.crop_widget.scale_factor,
            "offset": [self.crop_widget.offset.x(), self.crop_widget.offset.y()]
        }
        try:
            with open(self.state_path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print("Failed to save crop state:", e)


        if self.pictureChanged:
            self.pictureChanged(self.cropped_path, state)

        self.accept()

    def open_webcam_dialog(self):
        try:
            dialog = WebcamCaptureDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                pixmap = dialog.get_captured_pixmap()
                if pixmap:
                    self.original_pixmap = pixmap
                    self.crop_widget.set_pixmap(self.original_pixmap)
                    self.crop_widget.reset_offset_and_scale()
        except Exception as e:
            print("Error accessing webcam:", e)

class CropWidget(QLabel):
    def __init__(self, size, pixmap, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.crop_size = size
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.dragging = False
        self.last_pos = None
        self.setMouseTracking(True)
        self.update_scaled_pixmap()
        self.reset_offset_and_scale()

    def reset_offset_and_scale(self):
        self.scale_factor = self.crop_size / max(self.original_pixmap.width(), self.original_pixmap.height())
        self.update_scaled_pixmap()
        x = (self.width() - self.scaled_pixmap.width()) // 2
        y = (self.height() - self.scaled_pixmap.height()) // 2
        self.offset = QPoint(x, y)
        self.update()

    def update_scaled_pixmap(self):
        self.scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.scale_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    def set_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        self.update_scaled_pixmap()
        self.reset_offset_and_scale()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(42, 42, 42))
        painter.drawPixmap(self.offset, self.scaled_pixmap)

        path = QPainterPath()
        path.addEllipse(0, 0, self.width(), self.height())
        painter.setClipRegion(QRegion(self.rect()).subtracted(QRegion(path.toFillPolygon().toPolygon())))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRect(self.rect())
        painter.setClipping(False)

        painter.setPen(QColor(255, 255, 255, 180))
        painter.drawEllipse(0, 0, self.width() - 1, self.height() - 1)


        grid_pen = QPen(QColor(255, 255, 255, 50), 1)
        painter.setPen(grid_pen)
        step = self.width() // 3
        for i in range(1, 3):
            painter.drawLine(i * step, 0, i * step, self.height())
            painter.drawLine(0, i * step, self.width(), i * step)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_pos = event.position().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            pos = event.position().toPoint()
            delta = pos - self.last_pos
            self.last_pos = pos
            self.offset += delta
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        new_scale = self.scale_factor * factor
        if new_scale < 0.1 or new_scale > 5.0:
            return

        old_center = QPoint(self.offset.x() + self.scaled_pixmap.width() // 2,
                            self.offset.y() + self.scaled_pixmap.height() // 2)

        self.scale_factor = new_scale
        self.update_scaled_pixmap()
        new_size = self.scaled_pixmap.size()
        self.offset = QPoint(
            old_center.x() - new_size.width() // 2,
            old_center.y() - new_size.height() // 2
        )
        self.update()

    def get_cropped_pixmap(self) -> QPixmap:
        crop_pixmap = QPixmap(self.width(), self.height())
        crop_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(crop_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0, 0, self.width(), self.height())
        painter.setClipPath(path)
        painter.drawPixmap(self.offset, self.scaled_pixmap)
        painter.end()
        return crop_pixmap

class WebcamCaptureDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture from Webcam")
        self.setFixedSize(640, 520)
        self.setWindowIcon(QIcon(resource_path("assets/camera.png")))
        self.setStyleSheet("background-color: #121212; color: white;")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        self.capture_btn = QPushButton("Capture", self)
        self.capture_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #388e3c;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 11pt;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2e7d32;
            }
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.capture_btn)

        self.capture_btn.clicked.connect(self.capture_image)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_frame)
        self.timer.start(30)

        self.captured_frame = None

    def display_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_frame = frame
            self.timer.stop()
            self.cap.release()
            self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()

    def get_captured_pixmap(self):
        if self.captured_frame is None:
            return None
        rgb_frame = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

class SemiCircleButton(QPushButton):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        radius = 85


        circle_x = (width - 2 * radius) / 2

        circle_y = height - radius * 2

        circle_rect = QRectF(circle_x, circle_y, 2 * radius, 2 * radius)

        path = QPainterPath()
        path.moveTo(0, 0)
        path.arcTo(circle_rect, 180, 180)
        path.lineTo(width, 0)
        path.lineTo(0, 0)

        brush_color = QColor(255, 255, 255, int(0.3 * 255))
        painter.setBrush(brush_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        super().paintEvent(event)

class ProfilePicWithEdit(QWidget):
    def __init__(self, username, user_data, parent=None):
        super().__init__(parent)
        self.user_data = user_data
        self.username = username
        self.setFixedSize(170, 170)

        self.user_dir = os.path.join("user_images", self.username)
        os.makedirs(self.user_dir, exist_ok=True)
        self.cropped_path = os.path.join(self.user_dir, "profile_cropped.png")

        self.pic_label = QLabel(self)
        self.pic_label.setFixedSize(170, 170)
        self.pic_label.setStyleSheet("""
            QLabel {
                border-radius: 85px;
                background-color: #2a2a2a;
            }
        """)

        self.update_profile_pixmap(self.cropped_path if os.path.exists(self.cropped_path) else "assets/defaultprofile.png")

        self.edit_btn = SemiCircleButton(self)
        self.edit_btn.setIcon(QIcon(resource_path("assets/edit2.png")))
        self.edit_btn.setIconSize(QSize(24, 24))
        self.edit_btn.setFixedSize(170, 40)
        self.edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_btn.setStyleSheet("""
            QPushButton {
                border: none;
                color: black;
                background-color: transparent;
            }
            QPushButton:hover {
                color: black;
                background-color: transparent;
            }
        """)

        center_x = self.width() / 2
        self.edit_btn.move(
            int(center_x - self.edit_btn.width() / 2),
            int(self.height() - self.edit_btn.height())
        )

        self.edit_btn.hide()
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self.edit_btn.clicked.connect(self.open_edit_dialog)

    def update_profile_pixmap(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            pixmap = QPixmap(170, 170)
            pixmap.fill(Qt.GlobalColor.gray)

        scaled = pixmap.scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        masked_pixmap = QPixmap(170, 170)
        masked_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(masked_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addEllipse(5, 5, 160, 160)

        painter.setClipPath(path)
        x = (170 - scaled.width()) // 2
        y = (170 - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()

        self.pic_label.setPixmap(masked_pixmap)

    def enterEvent(self, event):
        self.edit_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.edit_btn.hide()
        super().leaveEvent(event)

    def open_edit_dialog(self):
        dialog = ProfilePictureDialog(self.username, self)
        dialog.pictureChanged = self.on_picture_changed
        dialog.pictureRemoved = self.on_picture_removed
        dialog.exec()

    def on_picture_changed(self, cropped_path, state=None):
        self.update_profile_pixmap(cropped_path)
        self.save_profile_picture(cropped_path, state)

    def on_picture_removed(self):
        default_pic = "assets/defaultprofile.png"
        self.update_profile_pixmap(default_pic)
        self.save_profile_picture(default_pic, None)

    def save_profile_picture(self, new_path, state=None):

        try:
            users = json.loads(Path("user_data.json").read_text())
        except Exception as e:
            print("user_data.json:", e)
            users = []

        username = self.username
        user_found = False
        for user in users:
            if user.get("username") == username:
                user["profile_picture"] = new_path
                if state:
                    user["crop_state"] = state
                user_found = True
                break

        if not user_found:
            new_user = {
                "username": username,
                "profile_picture": new_path,
            }
            if state:
                new_user["crop_state"] = state
            users.append(new_user)

        try:
            Path("user_data.json").write_text(json.dumps(users, indent=4))
            print(f"Profile picture updated and saved for user {username}")
        except Exception as e:
            print("Failed to save user_data.json:", e)

class EditableLabel(QWidget):
    def __init__(self, full_name, user_data: dict, parent=None):
        super().__init__(parent)
        self.user_data = user_data
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(6)

        self.label = QLabel(full_name)
        self.label.setStyleSheet("""
            color: white;
            font-size: 24pt;
            font-weight: bold;
        """)
        self.layout.addWidget(self.label)

        self.edit_btn = QPushButton()
        self.edit_btn.setIcon(QIcon(resource_path("assets/edit3.png")))
        self.edit_btn.setIconSize(QSize(20, 20))
        self.edit_btn.setFixedSize(28, 28)
        self.edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e2e2e;
                border-radius: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        self.edit_btn.hide()
        self.layout.addWidget(self.edit_btn)

        self.edit_btn.clicked.connect(self.open_edit_dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def enterEvent(self, event):
        self.edit_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.edit_btn.hide()
        super().leaveEvent(event)

    def open_edit_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Name")
        dialog.setFixedWidth(400)
        dialog.setWindowIcon(QIcon(resource_path("assets/editname.png")))
        layout = QVBoxLayout(dialog)
        input_style = """
            QLineEdit {
                border: 2px solid #00aaff;
                border-radius: 15px;
                padding-left: 15px;
                color: white;
                background-color: rgba(255, 255, 255, 0.15);
            }
            QLineEdit:focus {
                border-color: #33ccff;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """
        first_input = QLineEdit(self.user_data.get("first_name", ""))
        first_input.setFixedHeight(60)
        first_input.setFont(QFont(sf_family, 20))
        first_input.setPlaceholderText("First Name")
        first_input.setStyleSheet(input_style)

        last_input = QLineEdit(self.user_data.get("last_name", ""))
        last_input.setFixedHeight(60)
        last_input.setFont(QFont(sf_family, 20))
        last_input.setPlaceholderText("Last Name")
        last_input.setStyleSheet(input_style)
        save_btn = QPushButton("Save")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aaff;
                color: white;
                padding: 8px 20px;
                border-radius: 8px;
                font-size: 14pt;
            }
            QPushButton:hover {
                background-color: #0099dd;
            }
        """)



        label_style = "color: white; font-size: 20pt;background:transparent;border:none;"
        label1 = QLabel("First Name:")
        label1.setStyleSheet(label_style)
        layout.addWidget(label1)
        layout.addWidget(first_input)

        label2 = QLabel("Last Name:")
        label2.setStyleSheet(label_style)
        layout.addWidget(label2)
        layout.addWidget(last_input)

        layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)

        def save():
            new_first = first_input.text().strip()
            new_last = last_input.text().strip()
            self.user_data["first_name"] = new_first
            self.user_data["last_name"] = new_last
            self.label.setText(f"{new_first} {new_last}".strip())


            try:
                users = json.loads(Path("user_data.json").read_text())
                for user in users:
                    if user.get("username") == self.user_data.get("username"):
                        user["first_name"] = new_first
                        user["last_name"] = new_last
                        break
                Path("user_data.json").write_text(json.dumps(users, indent=4))
            except Exception as e:
                print("Failed to save name:", e)

            dialog.accept()

        save_btn.clicked.connect(save)
        dialog.exec()

class EditableContactRow(QWidget):
    def __init__(self, icon_path, label_text, value_text, user_data: dict, key: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.user_data = user_data

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(12)

        self.icon_btn = QPushButton()
        self.icon_btn.setIcon(QIcon(resource_path(icon_path)))
        self.icon_btn.setIconSize(QSize(36, 36))
        self.icon_btn.setFixedSize(40, 40)
        self.icon_btn.setStyleSheet("background-color: transparent; border: none;")

        self.label = QLabel(label_text)
        self.label.setStyleSheet("color: #bbb; font-size: 20pt; background-color: transparent;")

        self.value = QLabel(value_text)
        self.value.setStyleSheet("color: white; font-size: 16pt; background-color: transparent;")
        self.value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.edit_btn = QPushButton()
        self.edit_btn.setIcon(QIcon(resource_path("assets/edit3.png")))
        self.edit_btn.setIconSize(QSize(20, 20))
        self.edit_btn.setFixedSize(28, 28)
        self.edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e2e2e;
                border-radius: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        self.edit_btn.hide()

        self.edit_btn.clicked.connect(self.open_edit_dialog)

        self.layout.addWidget(self.icon_btn)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.value)
        self.layout.addStretch()
        self.layout.addWidget(self.edit_btn)

        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def enterEvent(self, event):
        self.edit_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.edit_btn.hide()
        super().leaveEvent(event)

    def open_edit_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit {self.key.capitalize()}")
        dialog.setStyleSheet("background-color: #121212; color: white;")
        dialog.setFixedWidth(400)
        layout = QVBoxLayout(dialog)
        label_style = "color: white; font-size: 16pt; background: transparent; border: none;"
        input_style = """
            QLineEdit {
                border: 2px solid #00aaff;
                border-radius: 15px;
                padding-left: 15px;
                color: white;
                background-color: rgba(255, 255, 255, 0.15);
            }
            QLineEdit:focus {
                border-color: #33ccff;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """


        if self.key == "phone":
            dialog.setWindowIcon(QIcon(resource_path("assets/editphone.png")))
            self.input_widget = PhoneInput()
            layout.addWidget(self.input_widget)


            current_number = self.user_data.get("phone", "")
            if current_number:
                try:

                    code = current_number.split()[0]
                    digits = " ".join(current_number.split()[1:])
                    region = phonenumbers.region_code_for_country_code(int(code[1:]))
                    self.input_widget.select_country(region, QWidget())
                    self.input_widget.phone_edit.setText(digits)
                except Exception as e:
                    print("Failed to parse phone number:", e)

            def get_value():
                return self.input_widget.get_full_number().strip()

        else:
            dialog.setWindowIcon(QIcon(resource_path("assets/editemail.png")))
            label = QLabel(f"{self.key.capitalize()}:")
            label.setStyleSheet(label_style)
            layout.addWidget(label)
            self.input_widget = QLineEdit(self.user_data.get(self.key, ""))
            self.input_widget.setStyleSheet(input_style)
            self.input_widget.setFixedHeight(60)
            self.input_widget.setFont(QFont(sf_family, 20))
            self.input_widget.setPlaceholderText("Email")
            layout.addWidget(self.input_widget)

            def get_value():
                return self.input_widget.text().strip()

        save_btn = QPushButton("Save")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aaff;
                color: white;
                padding: 8px 20px;
                border-radius: 8px;
                font-size: 14pt;
            }
            QPushButton:hover {
                background-color: #0099dd;
            }
        """)
        layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.setFixedHeight(dialog.sizeHint().height())
        def save():
            new_value = get_value()
            self.user_data[self.key] = new_value
            self.value.setText(new_value)

            try:
                users = json.loads(Path("user_data.json").read_text())
                for user in users:
                    if user.get("username") == self.user_data.get("username"):
                        user[self.key] = new_value
                        break
                Path("user_data.json").write_text(json.dumps(users, indent=4))
            except Exception as e:
                print(f"Failed to save {self.key}:", e)

            dialog.accept()

        save_btn.clicked.connect(save)
        dialog.exec()

class CelebrationOverlay(QLabel):
    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setVisible(False)

        self.movie = QMovie(resource_path("assets/completion.gif"))
        if self.movie.isValid():
            self.setMovie(self.movie)
        else:
            print("Error loading celebration GIF")

        self.setScaledContents(True)

        self.update_geometry()
        app = QApplication.instance()
        app.focusWindowChanged.connect(self.update_geometry)
        self.resizeEvent = self.on_resize

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(800)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self._display_text = ""

    def update_geometry(self, *_args):
        active_window = QApplication.activeWindow()
        if active_window:
            geo = active_window.geometry()
        else:
            geo = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(geo)
        self.raise_()

    def on_resize(self, event):
        self.setScaledContents(True)
        super().resizeEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self._display_text:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)


            font = QFont(sf_family, 42, QFont.Weight.Black)
            font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            painter.setFont(font)


            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0.0, QColor("#FFD700"))
            gradient.setColorAt(1.0, QColor("#FFA500"))
            brush = QBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(brush)


            shadow_color = QColor(0, 0, 0, 160)
            painter.setPen(shadow_color)
            offset = 3
            painter.drawText(self.rect().adjusted(offset, offset, offset, offset),
                            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                            self._display_text)


            painter.setPen(QPen(brush, 1))
            painter.drawText(self.rect(),
                            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                            self._display_text)

            painter.end()



    def show_celebration(self, duration_ms=5000):
        self.update_geometry()
        self.setVisible(True)
        self.movie.start()
        QTimer.singleShot(duration_ms, self.hide_celebration)

    def show_celebration(self, duration_ms=5000):
        self.update_geometry()
        self.setVisible(True)
        self.movie.start()


        self.fade_anim.stop()
        self.fade_anim.setDuration(800)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.start()


        QTimer.singleShot(duration_ms, self.fade_out_and_hide)

    def fade_out_and_hide(self):
        self.fade_anim.stop()
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()
        self.fade_anim.finished.connect(self._final_hide)

    def _final_hide(self):
        self.movie.stop()
        self.setVisible(False)
        self._display_text = ""
    def show_celebration_for(self, category: str, duration_ms=5000):
        messages = {
            "daily": "You completed the daily targets!",
            "weekly": "You completed the weekly targets!",
            "monthly": "You completed the monthly targets!",
            "final": "Congratulations! You completed all your targets!",
        }
        self._display_text = messages.get(category.lower(), "You completed your targets!")
        self.show_celebration(duration_ms=duration_ms)

class ClearableLineEdit(QLineEdit):
    def focusInEvent(self, event):
        if self.text().strip().lower() == "couldn't calculate calorie":
            self.setText("")
            self.setStyleSheet("color:#fff;background:#444;padding:6px;border-radius:6px;")
        super().focusInEvent(event)

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.setWindowIcon(QIcon(resource_path("assets/help.png")))
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setModal(False)


        layout = QHBoxLayout(self)
        icon_label = QLabel()
        icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
        icon_label.setPixmap(icon.pixmap(48, 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        content_layout = QVBoxLayout()


        browser = QTextBrowser(self)
        browser.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        help_file = resource_path("assets/help.html").replace("\\", "/")
        url = QUrl.fromLocalFile(help_file)
        browser.setHtml(
            f"This is the Fitness Tracking App.<br>"
            f'For more info, <a href="{url.toString()}">click me</a>.'
        )
        browser.setOpenExternalLinks(False)
        browser.anchorClicked.connect(self.on_link_clicked)
        browser.setFixedSize(250, 50)
        browser.setFrameStyle(QFrame.Shape.NoFrame)
        browser.setStyleSheet("background-color: transparent; color: white;")


        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)

        content_layout.addWidget(browser)
        content_layout.addWidget(buttons)

        layout.addWidget(icon_label)
        layout.addLayout(content_layout)

        self.setLayout(layout)
        self.setFixedSize(320, 90)

    def on_link_clicked(self, url: QUrl):
        QDesktopServices.openUrl(url)
        self.accept()

class TriangleBadge(QFrame):
    def __init__(self, parent=None, box_color="#316143", triangle_color="#2d2d2d",
                 box_width=90, triangle_depth=20, triangle_base=40, triangle_facing_right=True,
                 radius_tl=0, radius_tr=6, radius_br=6, radius_bl=0):
        super().__init__(parent)
        self.box_color = QColor(box_color)
        self.triangle_color = QColor(triangle_color)
        self.box_width = box_width
        self.triangle_depth = triangle_depth
        self.triangle_base = triangle_base
        self.triangle_facing_right = triangle_facing_right

        self.radius_tl = radius_tl
        self.radius_tr = radius_tr
        self.radius_br = radius_br
        self.radius_bl = radius_bl

        self.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("border: none;")
        self.setMinimumHeight(triangle_base + 10)
        self.setMinimumWidth(self.box_width + self.triangle_depth)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        w = self.width()
        h = self.height()
        td = self.triangle_depth
        tb = self.triangle_base

        r_tl = self.radius_tl
        r_tr = self.radius_tr
        r_br = self.radius_br
        r_bl = self.radius_bl


        path = QPainterPath()
        path.moveTo(0 + r_tl, 0)


        path.lineTo(w - r_tr, 0)
        if r_tr > 0:
            path.quadTo(w, 0, w, r_tr)


        path.lineTo(w, h - r_br)
        if r_br > 0:
            path.quadTo(w, h, w - r_br, h)


        path.lineTo(r_bl, h)
        if r_bl > 0:
            path.quadTo(0, h, 0, h - r_bl)


        path.lineTo(0, r_tl)
        if r_tl > 0:
            path.quadTo(0, 0, r_tl, 0)

        p.setBrush(self.box_color)
        p.drawPath(path)


        mid_y = h // 2
        top_y = mid_y - tb // 2
        bot_y = mid_y + tb // 2

        if self.triangle_facing_right:
            triangle = QPolygon([
                QPoint(-1, top_y),
                QPoint(-1, bot_y),
                QPoint(td, mid_y),
            ])
        else:
            triangle = QPolygon([
                QPoint(1, mid_y),
                QPoint(1, top_y),
                QPoint(td, bot_y),
            ])

        p.setBrush(self.triangle_color)
        p.drawPolygon(triangle)
        p.end()

class TightVerticalLabel(QLabel):
    def __init__(self, text, spacing=0, parent=None):
        super().__init__(parent)
        self.letters = list(text)
        self.spacing = spacing
        self.setMinimumSize(20, len(self.letters) * (10 + spacing))
        self.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")

    def sizeHint(self):
        return QSize(20, len(self.letters) * (12 + self.spacing))
    def paintEvent(self, event):
        painter = QPainter(self)
        font = QFont(sf_family, 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(self.palette().color(self.foregroundRole()))

        x_offset = 4
        y = 0
        for letter in self.letters:
            painter.drawText(x_offset, y + 12, letter)
            y += 12 + self.spacing

class DraggableBar(QWidget):
    def __init__(self,main_window=None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.BypassWindowManagerHint
        )
        self.main_window = main_window
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.bar_thickness = 36
        self.last_position = "top"
        self.fullscreen = True

        self.wrapper = QWidget(self)
        self.modern_colors = [
            "#0D47A1", "#1A237E", "#4A148C", "#880E4F", "#B71C1C",
            "#F57F17", "#1B5E20", "#00796B", "#006064", "#0097A7", "#1976D2",
        ]

        self.current_color_index = 0
        self._bg_color = QColor(self.modern_colors[0])
        self.wrapper.setStyleSheet(f"background-color: {self._bg_color.name()};")
        self.start_color_cycle()

        self.tag1 = QLabel(self.wrapper)
        self.tag2 = QLabel(self.wrapper)
        for tag in (self.tag1, self.tag2):
            tag.setFixedSize(12, 12)
            tag.setStyleSheet("background-color: transparent; border-radius: 6px;")
            tag.hide()

        self.set_top_bar()
        mw = self.main_window
        if mw:
            mw.installEventFilter(self)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def eventFilter(self, obj, event):
        if obj == QApplication.activeWindow():
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.WindowStateChange):
                QTimer.singleShot(10, self.restore_position)
        return super().eventFilter(obj, event)

    def restore_position(self):

        {
            "top": self.set_top_bar,
            "bottom": self.set_bottom_bar,
            "left": self.set_left_bar,
            "right": self.set_right_bar,
        }.get(self.last_position, self.set_top_bar)()

    def toggle_fullscreen(self):
        mw = self.main_window
        if mw:
            if self.fullscreen:
                mw.showNormal()
                self.fullscreen = False
            else:
                mw.showFullScreen()
                self.fullscreen = True
            QTimer.singleShot(50, self.restore_position)

    def minimize_all(self):
        mw = self.main_window
        if mw:
            mw.showMinimized()
        self.hide()


    def start_color_cycle(self):
        self.color_anim = QPropertyAnimation(self, b"bgColor", self)
        self.color_anim.setDuration(3000)
        self.color_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.color_anim.setLoopCount(1)
        self.color_anim.finished.connect(self.animate_to_next_color)
        self.animate_to_next_color()

    def animate_to_next_color(self):
        current = self._bg_color
        self.current_color_index = (self.current_color_index + 1) % len(self.modern_colors)
        next_color = QColor(self.modern_colors[self.current_color_index])

        self.color_anim.stop()
        self.color_anim.setStartValue(current)
        self.color_anim.setEndValue(next_color)
        self.color_anim.start()

        QTimer.singleShot(3000, self.animate_to_next_color)

    @pyqtProperty(QColor)
    def bgColor(self):
        return self._bg_color

    @bgColor.setter
    def bgColor(self, color):
        self._bg_color = color
        self.wrapper.setStyleSheet(f"background-color: {color.name()};")



    def clear_layout(self):
        old = self.wrapper.layout()
        if old:
            while old.count():
                item = old.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            QWidget().setLayout(old)

    def position_tags(self, horizontal: bool):
        w, h = self.width(), self.height()
        offset = int(w / 10) if horizontal else int(h / 10)
        if horizontal:
            self.tag1.move(offset - 6, h // 2 - 6)
            self.tag2.move(w - offset - 6, h // 2 - 6)
        else:
            self.tag1.move(w // 2 - 6, offset - 6)
            self.tag2.move(w // 2 - 6, h - offset - 6)
        self.tag1.hide()
        self.tag2.hide()

    def apply_layout(self, horizontal=True):
        self.clear_layout()
        layout = QHBoxLayout() if horizontal else QVBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0) if horizontal else layout.setContentsMargins(0, 10, 0, 10)

        icon_label = QLabel()
        icon_pixmap = QIcon(resource_path("assets/fitnessicon.ico")).pixmap(32, 32)
        icon_label.setPixmap(icon_pixmap)
        icon_label.setFixedSize(32, 32)

        title = QLabel("Fitness Tracker") if horizontal else TightVerticalLabel("Fitness Tracker", spacing=1)
        title.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")

        minimize_btn = QPushButton()
        minimize_btn.setIcon(QIcon(resource_path("assets/minimize.png")))
        minimize_btn.setIconSize(QSize(24, 24))
        minimize_btn.setFixedSize(32, 32)
        minimize_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        minimize_btn.setStyleSheet("QPushButton { border: none; background: transparent; } QPushButton:hover { background: rgba(37, 121, 238, 0.5);  border-radius: 16px; }")
        minimize_btn.clicked.connect(self.minimize_all)

        fullscreen_btn = QPushButton()
        icon_path = "assets/windowed.png" if self.fullscreen else "assets/fullscreen.png"
        fullscreen_btn.setIcon(QIcon(resource_path(icon_path)))
        fullscreen_btn.setIconSize(QSize(24, 24))
        fullscreen_btn.setFixedSize(32, 32)
        fullscreen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        fullscreen_btn.setStyleSheet("QPushButton { border: none; background: transparent; } QPushButton:hover { background: rgba(244, 67, 54, 0.5); border-radius: 16px; }")
        fullscreen_btn.clicked.connect(self.toggle_fullscreen)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon(resource_path("assets/close.png")))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("QPushButton { border: none; background: transparent; } QPushButton:hover { background: #aa0000; border-radius: 16px; }")
        close_btn.clicked.connect(QApplication.quit)

        if horizontal:
            layout.addWidget(icon_label)
            layout.addWidget(title)
            layout.addStretch()
            layout.addWidget(minimize_btn)
            layout.addWidget(fullscreen_btn)
            layout.addWidget(close_btn)
        else:
            layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addStretch()
            layout.addWidget(minimize_btn, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fullscreen_btn, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.wrapper.setLayout(layout)
        self.position_tags(horizontal)


    def set_bar_geometry(self, width, height, x=0, y=0, horizontal=True):
        """Set geometry of the bar manually to simulate fullscreen."""
        thickness = self.bar_thickness
        if horizontal:
            self.setGeometry(x, y, width, thickness)
            self.wrapper.setGeometry(0, 0, width, thickness)
            self.apply_layout(horizontal=True)
        else:
            self.setGeometry(x, y, thickness, height)
            self.wrapper.setGeometry(0, 0, thickness, height)
            self.apply_layout(horizontal=False)

    def set_top_bar(self):
        mw = self.main_window

        if mw and self.fullscreen == False:

            geom = mw.geometry()
            self.set_bar_geometry(geom.width(), self.bar_thickness, geom.x(), geom.y(), horizontal=True)
        else:
            g = QGuiApplication.primaryScreen().geometry()
            self.set_bar_geometry(g.width(), self.bar_thickness, 0, 0, horizontal=True)
        self.current_mode = "top"
        self.last_position = "top"
    def set_bottom_bar(self):

        mw = self.main_window
        if mw and self.fullscreen == False:
            geom = mw.geometry()
            self.set_bar_geometry(geom.width(), self.bar_thickness, geom.x(), geom.y() + geom.height() - self.bar_thickness, horizontal=True)
        else:
            g = QGuiApplication.primaryScreen().geometry()
            self.set_bar_geometry(g.width(), self.bar_thickness, 0, g.height() - self.bar_thickness, horizontal=True)
        self.last_position = "bottom"
        self.current_mode = "bottom"

    def set_left_bar(self):
        mw = self.main_window

        if mw and self.fullscreen == False:
            geom = mw.geometry()
            self.set_bar_geometry(self.bar_thickness, geom.height(), geom.x(), geom.y(), horizontal=False)
        else:
            g = QGuiApplication.primaryScreen().geometry()
            self.set_bar_geometry(self.bar_thickness, g.height(), g.x(), 0, horizontal=False)
        self.current_mode = "left"
        self.last_position = "left"
    def set_right_bar(self):

        mw = self.main_window

        if mw and self.fullscreen == False:
            geom = mw.geometry()
            self.set_bar_geometry(self.bar_thickness, geom.height(), geom.x() + geom.width() - self.bar_thickness, geom.y(), horizontal=False)
        else:
            g = QGuiApplication.primaryScreen().geometry()
            self.set_bar_geometry(self.bar_thickness, g.height(), g.width() - self.bar_thickness, 0, horizontal=False)
        self.current_mode = "right"
        self.last_position = "right"

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.old_pos = e.globalPosition().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton and self.old_pos:
            delta = e.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = e.globalPosition().toPoint()
            self.check_tags()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            QTimer.singleShot(10, self.snap_centered)
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def check_tags(self):

        if self.fullscreen:
            g = QGuiApplication.primaryScreen().geometry()
        else:
            mw = self.main_window

            if mw:
                g = mw.geometry()
            else:
                g = QGuiApplication.primaryScreen().geometry()

        th = 0
        p1 = self.tag1.mapToGlobal(QPoint(0, 0))
        p2 = self.tag2.mapToGlobal(QPoint(0, 0))

        if self.current_mode in ("top", "bottom"):

            if p1.x() <= g.left() + th:
                self.set_left_bar()
                self.old_pos = None
                return
            if p2.x() + 12 >= g.right() - th:
                self.set_right_bar()
                self.old_pos = None
                return
        else:

            if p1.y() <= g.top() + th:
                self.set_top_bar()
                self.old_pos = None
                return
            if p2.y() + 12 >= g.bottom() - th:
                self.set_bottom_bar()
                self.old_pos = None
                return

    def snap_centered(self):
        if self.fullscreen:
            g = QGuiApplication.primaryScreen().geometry()
        else:
            mw = self.main_window

            if mw:
                g = mw.geometry()
            else:
                g = QGuiApplication.primaryScreen().geometry()

        c = self.geometry().center()
        dist = {
            "left": abs(c.x() - g.left()),
            "right": abs(c.x() - g.right()),
            "top": abs(c.y() - g.top()),
            "bottom": abs(c.y() - g.bottom()),
        }
        edge, d = min(dist.items(), key=lambda kv: kv[1])
        if d > 40:
            self.old_pos = None
            return

        {
            "left": self.set_left_bar,
            "right": self.set_right_bar,
            "top": self.set_top_bar,
            "bottom": self.set_bottom_bar,
        }[edge]()
        self.old_pos = None


    def contextMenuEvent(self, event):
        menu = QMenu(self)

        help_action = QAction(QIcon(resource_path("assets/help.png")), "Help", self)
        help_action.triggered.connect(self.show_help_dialog)

        close_action = QAction(QIcon(resource_path("assets/close7.png")), "Close", self)
        close_action.setShortcut("Alt+F4")
        close_action.triggered.connect(QApplication.quit)

        menu.addAction(help_action)
        menu.addSeparator()
        menu.addAction(close_action)

        menu.popup(event.globalPos())



    def show_help_dialog(self):
        dlg = HelpDialog(self)
        dlg.exec()

class PhoneInput(QWidget):
    def __init__(self):
        super().__init__()
        self.setMaximumWidth(MAX_INPUT_WIDTH)

        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)


        label = QLabel("Phone Number")
        label.setStyleSheet("color: white;")
        label.setFont(QFont(sf_family, 20))
        wrapper.addWidget(label)


        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)


        self.country_btn = QPushButton()
        self.country_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.country_btn.setFixedSize(120, 60)
        self.country_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.15);
                border: 2px solid #00aaff;
                border-right: none;
                border-top-left-radius: 15px;
                border-bottom-left-radius: 15px;
                padding: 0 10px;
                color: white;
                font-size: 20px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.25);
            }
        """)
        layout.addWidget(self.country_btn)


        self.phone_edit = QLineEdit()
        self.phone_edit.setFont(QFont(sf_family, 24))
        self.phone_edit.setPlaceholderText("Phone Number")
        self.phone_edit.setFixedHeight(60)
        self.phone_edit.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255,255,255,0.15);
                border: 2px solid #00aaff;
                border-left: none;
                border-top-right-radius: 15px;
                border-bottom-right-radius: 15px;
                padding-left: 15px;
                color: white;
            }
            QLineEdit:hover {
                background-color: rgba(255,255,255,0.3);
            }
        """)
        layout.addWidget(self.phone_edit)
        wrapper.addLayout(layout)

        self.current_country = "OM"
        self.update_country_display()
        self.country_btn.clicked.connect(self.show_country_selector)

    def update_country_display(self):
        code = f"+{phonenumbers.country_code_for_region(self.current_country)}"
        flag_path = resource_path(f"assets/flags/{self.current_country.lower()}.svg")
        self.country_btn.setText(code)
        self.country_btn.setIcon(QIcon(flag_path))
        self.country_btn.setIconSize(QSize(24, 24))
        mask = PHONE_FORMATS.get(self.current_country, "")
        self.phone_edit.setInputMask(mask)

    def show_country_selector(self):
        popup = QWidget(self, Qt.WindowType.Popup)
        popup.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        popup.setStyleSheet("""
            background-color: #2a2a2a;
            border: 2px solid #444;
        """)

        scroll = QScrollArea(popup)
        scroll.setWidgetResizable(True)
        scroll.setFixedSize(350, 240)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 4px 0 4px 0;
            }

            QScrollBar::handle:vertical {
                background-color: rgba(255, 255, 255, 60);
                border-radius: 5px;
                min-height: 30px;
                border: 1px solid rgba(255, 255, 255, 40);
            }

            QScrollBar::handle:vertical:hover {
                background-color: rgba(255, 255, 255, 100);
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
        """)


        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        for country in sorted(pycountry.countries, key=lambda c: c.name):
            code = phonenumbers.country_code_for_region(country.alpha_2)
            if not code:
                continue
            flag_path = resource_path(f"assets/flags/{country.alpha_2.lower()}.svg")
            button = QPushButton(f"{country.name}  (+{code})")
            button.setIcon(QIcon(flag_path))
            button.setIconSize(QSize(24, 24))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: white;
                    padding: 8px 12px;
                    text-align: left;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                }
            """)
            button.clicked.connect(lambda checked=False, code=country.alpha_2: self.select_country(code, popup))
            layout.addWidget(button)

        scroll.setWidget(container)
        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.addWidget(scroll)

        popup.move(self.country_btn.mapToGlobal(QPoint(0, self.country_btn.height())))
        popup.show()

    def select_country(self, alpha2, popup):
        self.current_country = alpha2
        self.update_country_display()
        popup.close()

    def get_full_number(self):
        code = f"+{phonenumbers.country_code_for_region(self.current_country)}"
        return f"{code} {self.phone_edit.text().strip()}"

class LoginPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.user_icon_widget = None  
        self.setup_ui()
        self.bar = None


    def toggle_password_visibility(self):
        is_hidden = self.password.echoMode() == QLineEdit.EchoMode.Password
        self.password.setEchoMode(QLineEdit.EchoMode.Normal if is_hidden else QLineEdit.EchoMode.Password)
        self.toggle_password_btn.setIcon(QIcon(resource_path("assets/openeye.png") if is_hidden else resource_path("assets/closedeye.png")))
        self.show_password_label.setText("Hide Password" if is_hidden else "Show Password")

        self.update_password_font()


    def toggle_password_visibility_event(self, event):
        self.toggle_password_visibility()


    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget()
        self.container_layout = QHBoxLayout()
        self.container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.user_icon_widget = QSvgWidget(resource_path("assets/defaultprofile.svg"))
        self.user_icon_widget.setFixedSize(200, 200)

        self.container_layout.addWidget(self.user_icon_widget)
        container.setLayout(self.container_layout)

        layout.addWidget(container)

        self.username = QLineEdit()
        self.username.setPlaceholderText("User Name")
        self.username.setFixedHeight(60)
        self.username.setFont(QFont(sf_family, 24))
        self.username.setMaximumWidth(MAX_INPUT_WIDTH)
        self.username.setStyleSheet(self.input_style())
        layout.addWidget(self.username)

        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setFixedHeight(60)
        self.password.setMaximumWidth(MAX_INPUT_WIDTH)
        self.password.setStyleSheet(self.input_style())
        self.password.textChanged.connect(self.update_password_font)
        self.update_password_font()

        layout.addWidget(self.password)
        self.update_password_font()


        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(10)


        self.toggle_password_btn = QPushButton()
        self.toggle_password_btn.setIcon(QIcon(resource_path("assets/closedeye.png")))
        self.toggle_password_btn.setFixedSize(32, 32)
        self.toggle_password_btn.setIconSize(self.toggle_password_btn.size())
        self.toggle_password_btn.setStyleSheet("border: none;")
        self.toggle_password_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_password_btn.clicked.connect(self.toggle_password_visibility)

        self.show_password_label = QLabel("Show Password")
        self.show_password_label.setFont(QFont(sf_family, 14))
        self.show_password_label.setStyleSheet("color: white;")
        self.show_password_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_password_label.mousePressEvent = self.toggle_password_visibility_event

        left_widget = QWidget()
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.toggle_password_btn)
        left_layout.addWidget(self.show_password_label)
        left_widget.setLayout(left_layout)


        self.remember_me_checkbox = QCheckBox("Remember Me")
        self.remember_me_checkbox.setFont(QFont(sf_family, 14))
        self.remember_me_checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        checkwhite = resource_path("assets/checkwhite.png").replace("\\", "/")
        self.remember_me_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: white;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 2px solid #aaa;
                background-color: transparent;
            }}
            QCheckBox::indicator:checked {{
                image: url({checkwhite});
                background-color: transparent;
                border: 2px solid #fff;
            }}
        """)


        right_widget = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.remember_me_checkbox)
        right_widget.setLayout(right_layout)


        container_layout.addWidget(left_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        container_layout.addStretch()
        container_layout.addWidget(right_widget, alignment=Qt.AlignmentFlag.AlignRight)

        container.setLayout(container_layout)
        container.setMaximumWidth(MAX_INPUT_WIDTH)

        layout.addWidget(container)


        self.login_btn = QPushButton("Login")
        self.login_btn.setFixedHeight(70)
        self.login_btn.setFixedWidth(400)
        self.login_btn.setFont(QFont(sf_family, 28))
        self.login_btn.setStyleSheet(self.button_style())
        self.login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout.addWidget(self.login_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        no_account_label = QLabel("No account?")
        no_account_label.setFont(QFont(sf_family, 14))
        no_account_label.setStyleSheet("color: #cccccc;")
        bottom_layout.addWidget(no_account_label)

        create_link = QLabel('<a href="#">Create one</a>')
        create_link.setFont(QFont(sf_family, 14))
        create_link.setTextFormat(Qt.TextFormat.RichText)
        create_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        create_link.setOpenExternalLinks(False)
        create_link.setStyleSheet("color: #00aaff;")
        bottom_layout.addWidget(create_link)

        layout.addLayout(bottom_layout)
        self.setLayout(layout)


        create_link.linkActivated.connect(self.goto_create_account)
        self.login_btn.clicked.connect(self.login_user)

        remember_me, saved_username, saved_password, profile_picture = self.load_remember_me()
        self.remember_me_checkbox.setChecked(remember_me)
        if remember_me:
            self.username.setText(saved_username)
            self.password.setText(saved_password)
            if profile_picture:
                self.set_user_profile_picture(profile_picture)


    def goto_create_account(self):
        self.parent().setCurrentIndex(1)

    def login_user(self):
        username = self.username.text().strip()
        password = self.password.text().strip()

        try:
            with open("user_data.json", "r") as file:
                users = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            users = []

        for user in users:
            if user["username"] == username and user["password"] == password:

                profile_picture = user.get("profile_picture", None)
                self.save_remember_me(self.remember_me_checkbox.isChecked(), username, password, profile_picture)

                remember_me, saved_username, saved_password, saved_profile_picture = self.load_remember_me()
                if remember_me and username == saved_username:
                    if saved_profile_picture:
                        self.set_user_profile_picture(saved_profile_picture)

                if user.get("2fa", False):
                    self.send_2fa_code(user)
                else:
                    self.finish_login(user)
                return

        # winsound.MessageBeep(winsound.MB_ICONHAND) # Uncomment if you have winsound
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Login Failed")
        msg.setText("Invalid username or password.")
        # msg.setWindowIcon(QIcon(resource_path("assets/loginfailed.png"))) # Uncomment if you have the asset
        msg.exec()

    def update_password_font(self):
        text = self.password.text()
        is_hidden = self.password.echoMode() == QLineEdit.EchoMode.Password

        if not text:
            input_font_size = 24
            placeholder_font_size = 24
        elif text and is_hidden:
            input_font_size = 12
            placeholder_font_size = 24
        else:
            input_font_size = 24
            placeholder_font_size = 24

        self.password.setStyleSheet(self.input_style(input_font_size=input_font_size, placeholder_font_size=placeholder_font_size))


    def input_style(self, input_font_size=24, placeholder_font_size=24):
        return f"""
            QLineEdit {{
                border: 2px solid #00aaff;
                border-radius: 15px;
                padding-left: 15px;
                color: white;
                background-color: rgba(255, 255, 255, 0.15);
                font-size: {input_font_size}pt;
                font-family: {sf_family};
            }}
            QLineEdit::placeholder {{
                font-size: {placeholder_font_size}pt;
                font-family: {sf_family};
                color: rgba(255, 255, 255, 0.6);
            }}
            QLineEdit:focus {{
                border-color: #33ccff;
                background-color: rgba(255, 255, 255, 0.3);
            }}
        """


    def button_style(self):
        return """
            QPushButton {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #00c6ff, stop:1 #0072ff);
                border-radius: 20px;
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #33ccff, stop:1 #0099ff);
            }
            QPushButton:pressed {
                background-color: #005bb5;
            }
        """
    def reset_fields(self):
        if not self.remember_me_checkbox.isChecked():
            self.username.clear()
            self.password.clear()
            self.set_user_profile_picture(None)

        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.toggle_password_btn.setIcon(QIcon(resource_path("assets/closedeye.png")))
        self.show_password_label.setText("Show Password")


    def encrypt_data(self, data: bytes) -> bytes:
        cipher = Fernet(ENCRYPTION_KEY)
        return cipher.encrypt(data)

    def decrypt_data(self, data: bytes) -> bytes:
        cipher = Fernet(ENCRYPTION_KEY)
        return cipher.decrypt(data)

    def save_remember_me(self, remember_me: bool, username: str, password: str, profile_picture: str = None):
        try:
            if os.path.exists(REMEMBER_ME_FILE):
                try:
                    os.remove(REMEMBER_ME_FILE)
                except PermissionError as e:
                    print(f"Permission error deleting remember me file: {e}")

            if remember_me:
                data = {
                    "remember_me": remember_me,
                    "username": username,
                    "password": password,
                    "profile_picture": profile_picture
                }
                json_bytes = json.dumps(data).encode('utf-8')
                encrypted = self.encrypt_data(json_bytes)
                with open(REMEMBER_ME_FILE, "wb") as f:
                    f.write(encrypted)

                if sys.platform.startswith("win"):
                    import ctypes
                    FILE_ATTRIBUTE_HIDDEN = 0x02
                    ctypes.windll.kernel32.SetFileAttributesW(REMEMBER_ME_FILE, FILE_ATTRIBUTE_HIDDEN)

        except Exception as e:
            print(f"Failed to save remember me file: {e}")


    def load_remember_me(self):
        try:
            if not os.path.exists(REMEMBER_ME_FILE):
                return False, "", "", None
            with open(REMEMBER_ME_FILE, "rb") as f:
                encrypted = f.read()
            decrypted = self.decrypt_data(encrypted)
            data = json.loads(decrypted.decode("utf-8"))
            remember_me = data.get("remember_me", False)
            username = data.get("username", "")
            password = data.get("password", "")
            profile_picture = data.get("profile_picture", None)
            return remember_me, username, password, profile_picture
        except Exception as e:
            print(f"Failed to load remember me file: {e}")
            return False, "", "", None

    def set_user_profile_picture(self, profile_picture_path):
        # Remove the old widget first
        if self.user_icon_widget:
            self.container_layout.removeWidget(self.user_icon_widget)
            self.user_icon_widget.deleteLater()

        if profile_picture_path and os.path.exists(profile_picture_path):
            pixmap = QPixmap(profile_picture_path)
            self.user_icon_widget = QLabel()



            self.user_icon_widget.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

            self.user_icon_widget.setFixedSize(200, 200)
        else:
            print("Loading default profile picture.")
            self.user_icon_widget = QSvgWidget(resource_path("assets/defaultprofile.svg"))
            self.user_icon_widget.setFixedSize(200, 200)


        self.container_layout.addWidget(self.user_icon_widget)


    def send_2fa_code(self, user):
        self.generated_code = str(random.randint(100000, 999999))
        user["2fa_code"] = self.generated_code

        try:
            with open("user_data.json", "r") as file:
                users = json.load(file)
            for u in users:
                if u["username"] == user["username"]:
                    u["2fa_code"] = self.generated_code
                    break
            Path("user_data.json").write_text(json.dumps(users, indent=2))
        except:
            QMessageBox.critical(self, "Error", "Failed to save 2FA code.")
            return

        first_name = user.get("first_name", user["username"])
        last_name = user.get("last_name", "")

        send_code_success = send_2fa_email(user["email"], user["username"], self.generated_code, first_name, last_name)
        if send_code_success:
            self.show_2fa_dialog(user)
        else:
            QMessageBox.critical(self, "Error", "Failed to send 2FA code. Check your email setup.")


    def show_2fa_dialog(self, user):
        dialog = QDialog(self)
        dialog.setWindowTitle("Enter 2FA Code")
        dialog.setStyleSheet("background-color: #1e1e1e; color: white;")

        layout = QVBoxLayout(dialog)

        label = QLabel("Enter the 6-digit code sent to your email:")
        layout.addWidget(label)

        code_container = QHBoxLayout()
        code_inputs = []


        for i in range(6):
            box = QLineEdit()
            box.setMaxLength(1)
            box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.setFont(QFont(sf_family, 18))
            box.setFixedSize(40, 50)
            box.setStyleSheet("background: #333; color: white; border-radius: 6px; font-size: 18pt;")
            box.setValidator(QIntValidator(0, 9))
            code_inputs.append(box)
            code_container.addWidget(box)

        layout.addLayout(code_container)


        for i in range(6):
            def make_handler(index):
                def on_text_changed():
                    if code_inputs[index].text() and index < 5:
                        code_inputs[index + 1].setFocus()
                return on_text_changed

            def make_key_handler(index):
                def on_key(event):
                    if event.key() in [Qt.Key.Key_Backspace, Qt.Key.Key_Left] and not code_inputs[index].text():
                        if index > 0:
                            code_inputs[index - 1].setFocus()
                    return QLineEdit.keyPressEvent(code_inputs[index], event)
                return on_key

            code_inputs[i].textChanged.connect(make_handler(i))
            code_inputs[i].keyPressEvent = make_key_handler(i)

        code_inputs[0].setFocus()


        btn_verify = QPushButton("Verify")
        btn_verify.setStyleSheet("background-color: #00aaff; color: white; padding: 10px; border-radius: 8px;")
        layout.addWidget(btn_verify)


        resend_label = QLabel("Send another (60)")
        resend_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        resend_label.setStyleSheet("color: #888888; font-size: 12pt;")
        layout.addWidget(resend_label)

        resend_timer = QTimer(self)
        resend_seconds = [60]

        def update_resend_label():
            if resend_seconds[0] > 0:
                resend_label.setText(f"Send another ({resend_seconds[0]})")
                resend_label.setStyleSheet("color: #888888; font-size: 12pt;")
                resend_label.setCursor(Qt.CursorShape.ArrowCursor)
                resend_label.setEnabled(False)
                resend_seconds[0] -= 1
            else:
                resend_label.setText("Send another")
                resend_label.setStyleSheet("color: #00aaff; font-size: 12pt;")
                resend_label.setCursor(Qt.CursorShape.PointingHandCursor)
                resend_label.setEnabled(True)
                resend_timer.stop()

        resend_timer.timeout.connect(update_resend_label)
        resend_timer.start(1000)

        def resend_clicked(event):
            if resend_seconds[0] == 0:
                self.send_2fa_code(user)
                resend_seconds[0] = 60
                resend_timer.start(1000)

        resend_label.mousePressEvent = resend_clicked

        def verify_code():
            entered = "".join([box.text() for box in code_inputs])
            if entered == self.generated_code:
                dialog.close()
                QTimer.singleShot(100, lambda: self.finish_login(user))
            else:
                QMessageBox.warning(dialog, "Invalid", "Incorrect code.")

        btn_verify.clicked.connect(verify_code)
        dialog.show()


    def finish_login(self, user):
        first_name = user.get("first_name", user["username"])
        # winsound.MessageBeep(winsound.MB_ICONASTERISK) # Uncomment if you have winsound
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Login Successful")
        msg.setText(f"Welcome {first_name}!")
        # msg.setWindowIcon(QIcon(resource_path("assets/loginsuccess.png"))) # Uncomment if you have the asset
        msg.exec()

        main_window = self.parent()
        # Ensure main_window.dashboard_page exists and has update_data method
        if hasattr(main_window, 'dashboard_page') and hasattr(main_window.dashboard_page, 'update_data'):
            main_window.dashboard_page.update_data(user)
        main_window.setCurrentIndex(5)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.login_user()
        else:
            super().keyPressEvent(event)

class CreateAccountPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.setup_ui()

    def clear_error_text(self):
        if self.email.placeholderText() == "Email already registered!":
            self.email.setPlaceholderText("Email")
            self.email.setStyleSheet(LoginPage.input_style(self))

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)



        back_btn_layout = QHBoxLayout()
        back_btn_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.back_btn = QPushButton("← Back")
        self.back_btn.setFont(QFont(sf_family, 18))
        self.back_btn.setFixedSize(120, 40)
        self.back_btn.setStyleSheet("""
            QPushButton {
                color: white;
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                color: #00aaff;
            }
        """)
        back_btn_layout.addWidget(self.back_btn)
        main_layout.addLayout(back_btn_layout)

        title = QLabel("Create Account")
        title.setFont(QFont(sf_family, 36))
        title.setStyleSheet("color: white;")
        main_layout.addWidget(title)


        self.first_name = QLineEdit()
        self.first_name.setPlaceholderText("First Name")
        self.first_name.setFixedHeight(60)
        self.first_name.setFont(QFont(sf_family, 24))
        self.first_name.setMaximumWidth(MAX_INPUT_WIDTH)
        self.first_name.setStyleSheet(LoginPage.input_style(self))
        main_layout.addWidget(self.first_name)

        self.last_name = QLineEdit()
        self.last_name.setPlaceholderText("Last Name")
        self.last_name.setFixedHeight(60)
        self.last_name.setFont(QFont(sf_family, 24))
        self.last_name.setMaximumWidth(MAX_INPUT_WIDTH)
        self.last_name.setStyleSheet(LoginPage.input_style(self))
        main_layout.addWidget(self.last_name)


        dob_label = QLabel("Date of Birth")
        dob_label.setFont(QFont(sf_family, 20))
        dob_label.setStyleSheet("color: white;")
        main_layout.addWidget(dob_label)
        self.dob = QDateEdit()
        self.dob.setCalendarPopup(True)
        self.dob.setDate(QDate.currentDate())
        self.dob.setDisplayFormat("dd MMM yyyy")
        self.dob.setFont(QFont(sf_family, 24))
        self.dob.setFixedHeight(60)
        self.dob.setMaximumWidth(MAX_INPUT_WIDTH)
        arrow_path = resource_path("assets/blue_arrow.png").replace("\\", "/")
        self.dob.setStyleSheet(f"""
            QDateEdit {{
                border: 2px solid #00aaff;
                border-radius: 15px;
                padding-left: 15px;
                padding-right: 15px;
                color: white;
                background-color: rgba(255, 255, 255, 0.15);
            }}
            QDateEdit:hover {{
                background-color: rgba(255, 255, 255, 0.3);
                border-color: #33ccff;
            }}
            QDateEdit::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 1px solid #00aaff;
            }}
            QDateEdit::down-arrow {{
                image: url("{arrow_path}");
                width: 20px;
                height: 20px;
            }}
        """)

        main_layout.addWidget(self.dob)

        self.email = QLineEdit()
        self.email.setPlaceholderText("Email")
        self.email.setFixedHeight(60)
        self.email.setFont(QFont(sf_family, 24))
        self.email.setMaximumWidth(MAX_INPUT_WIDTH)
        self.email.setStyleSheet(LoginPage.input_style(self))
        main_layout.addWidget(self.email)


        self.phone_input = PhoneInput()
        main_layout.addWidget(self.phone_input)


        gender_frame = QFrame()
        gender_frame.setFixedHeight(60)
        gender_frame.setMaximumWidth(MAX_INPUT_WIDTH)
        gender_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #00aaff;
                border-radius: 15px;
                background-color: rgba(255, 255, 255, 0.15);
            }
        """)


        gender_layout = QHBoxLayout()
        gender_layout.setContentsMargins(15, 0, 15, 0)
        gender_layout.setSpacing(40)
        gender_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)


        gender_label = QLabel("Select Gender")
        gender_label.setFont(QFont(sf_family, 24))
        gender_label.setStyleSheet("""
                color: white;
                background-color: transparent;
                border: none;
                """)
        gender_layout.addWidget(gender_label)


        self.male_radio = QRadioButton("Male")
        self.male_radio.setFont(QFont(sf_family, 20))
        self.male_radio.setStyleSheet("color: white;")
        self.male_radio.setIcon(QIcon(resource_path("assets/male_icon.png")))
        self.male_radio.setIconSize(QSize(32, 32))
        gender_layout.addWidget(self.male_radio)


        self.female_radio = QRadioButton("Female")
        self.female_radio.setFont(QFont(sf_family, 20))
        self.female_radio.setStyleSheet("color: white;")
        self.female_radio.setIcon(QIcon(resource_path("assets/female_icon.png")))
        self.female_radio.setIconSize(QSize(32, 32))
        gender_layout.addWidget(self.female_radio)

        gender_frame.setLayout(gender_layout)


        main_layout.addWidget(gender_frame)

        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedHeight(70)
        self.next_btn.setFont(QFont(sf_family, 28))
        self.next_btn.setMaximumWidth(MAX_INPUT_WIDTH)
        self.next_btn.setStyleSheet(LoginPage.button_style(self))
        main_layout.addWidget(self.next_btn)

        self.setLayout(main_layout)


        self.next_btn.clicked.connect(self.go_next)
        self.back_btn.clicked.connect(self.go_back)
        self.email.selectionChanged.connect(self.clear_error_text)
        self.email.textEdited.connect(self.clear_error_text)

    def go_next(self):
        email_input = self.email.text().strip()

        try:
            with open("user_data.json", "r") as f:
                users = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            users = []

        if any(user.get("email", "").lower() == email_input.lower() for user in users):
            self.email.clear()
            self.email.setPlaceholderText("Email already registered!")
            self.email.setStyleSheet("""
                QLineEdit {
                    border: 2px solid red;
                    border-radius: 15px;
                    padding-left: 15px;
                    color: red;
                    background-color: rgba(255, 255, 255, 0.15);
                }
            """)
            self.email.setFocus()
            return

        self.email.setStyleSheet(LoginPage.input_style(self))
        self.email.setPlaceholderText("Email")


        if self.male_radio.isChecked():
            gender = "Male"
        elif self.female_radio.isChecked():
            gender = "Female"
        else:
            QMessageBox.warning(self, "Missing Info", "Please select a gender.")
            return


        self.parent().data['first_name'] = self.first_name.text()
        self.parent().data['last_name'] = self.last_name.text()
        self.parent().data['dob'] = self.dob.date().toString("yyyy-MM-dd")
        self.parent().data['email'] = email_input

        self.parent().data['phone'] = self.phone_input.get_full_number()


        self.parent().data['gender'] = gender

        self.parent().setCurrentIndex(2)

    def go_back(self):
        self.parent().setCurrentIndex(0)

class RoleSelectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)





        back_btn_layout = QHBoxLayout()
        back_btn_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.back_btn = QPushButton("← Back")
        self.back_btn.setFont(QFont(sf_family, 18))
        self.back_btn.setFixedSize(120, 40)
        self.back_btn.setStyleSheet(
            """
            QPushButton {
                color: white;
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                color: #00aaff;
            }
        """
        )
        back_btn_layout.addWidget(self.back_btn)
        main_layout.addLayout(back_btn_layout)

        title = QLabel("Select Your Role")
        title.setFont(QFont(sf_family, 36))
        title.setStyleSheet("color: white;")
        main_layout.addWidget(title)


        trainer_layout = QHBoxLayout()
        trainer_icon = QLabel()
        trainer_pixmap = QPixmap(resource_path("assets/trainer.png")).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        trainer_icon.setPixmap(trainer_pixmap)
        trainer_layout.addWidget(trainer_icon)

        self.trainer_radio = QRadioButton("Trainer")
        self.trainer_radio.setFont(QFont(sf_family, 28))
        self.trainer_radio.setStyleSheet("color: white;")
        trainer_layout.addWidget(self.trainer_radio)
        trainer_layout.addStretch()
        main_layout.addLayout(trainer_layout)


        user_layout = QHBoxLayout()
        user_icon = QLabel()
        user_pixmap = QPixmap(resource_path("assets/client.png")).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        user_icon.setPixmap(user_pixmap)
        user_layout.addWidget(user_icon)

        self.user_radio = QRadioButton("User")
        self.user_radio.setFont(QFont(sf_family, 28))
        self.user_radio.setStyleSheet("color: white;")
        user_layout.addWidget(self.user_radio)
        user_layout.addStretch()
        main_layout.addLayout(user_layout)


        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.setFixedHeight(70)
        self.confirm_btn.setFont(QFont(sf_family, 28))
        self.confirm_btn.setMaximumWidth(MAX_INPUT_WIDTH)
        self.confirm_btn.setStyleSheet(LoginPage.button_style(self))
        main_layout.addWidget(self.confirm_btn)

        container = QWidget()
        container.setLayout(main_layout)

        outer_layout = QVBoxLayout(self)
        outer_layout.addStretch()
        outer_layout.addWidget(container, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer_layout.addStretch()


        self.confirm_btn.clicked.connect(self.confirm_role)
        self.back_btn.clicked.connect(self.go_back)

    def confirm_role(self):
        if self.trainer_radio.isChecked():
            self.parent().data['role'] = "Trainer"
            QMessageBox.information(self, "Role Selected", "Trainer role selected!")
        elif self.user_radio.isChecked():
            self.parent().data['role'] = "User"
            QMessageBox.information(self, "Role Selected", "User role selected!")
        else:
            QMessageBox.warning(self, "No Role Selected", "Please select a role!")
            return

        self.parent().setCurrentIndex(3)

    def go_back(self):
        self.parent().setCurrentIndex(1)

class AccountCredentialsPage(QWidget):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.data = data if data is not None else {}
        self.setup_ui()

    def clear_error_text(self):

        if self.username.placeholderText() == "Username already taken":
            self.username.setPlaceholderText("Username")
            self.username.setStyleSheet(LoginPage.input_style(self))

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)



        self.back_btn = QPushButton("← Back")
        self.back_btn.setFont(QFont(sf_family, 18))
        self.back_btn.setFixedSize(120, 40)
        self.back_btn.setStyleSheet("""
            QPushButton {
                color: white;
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                color: #00aaff;
            }
        """)
        layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignLeft)


        title = QLabel("Set Account Credentials")
        title.setFont(QFont(sf_family, 36))
        title.setStyleSheet("color: white;")
        layout.addWidget(title)

        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")
        self.username.setFixedHeight(60)
        self.username.setFont(QFont(sf_family, 24))
        self.username.setMaximumWidth(MAX_INPUT_WIDTH)
        self.username.setStyleSheet(LoginPage.input_style(self))
        layout.addWidget(self.username)

        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setFixedHeight(60)
        self.password.setFont(QFont(sf_family, 24))
        self.password.setMaximumWidth(MAX_INPUT_WIDTH)
        self.password.setStyleSheet(LoginPage.input_style(self))
        layout.addWidget(self.password)

        self.confirm_password = QLineEdit()
        self.confirm_password.setPlaceholderText("Confirm Password")
        self.confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_password.setFixedHeight(60)
        self.confirm_password.setFont(QFont(sf_family, 24))
        self.confirm_password.setMaximumWidth(MAX_INPUT_WIDTH)
        self.confirm_password.setStyleSheet(LoginPage.input_style(self))
        layout.addWidget(self.confirm_password)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setFixedHeight(70)
        self.submit_btn.setFont(QFont(sf_family, 28))
        self.submit_btn.setMaximumWidth(MAX_INPUT_WIDTH)
        self.submit_btn.setStyleSheet(LoginPage.button_style(self))
        layout.addWidget(self.submit_btn)

        self.setLayout(layout)

        self.submit_btn.clicked.connect(self.submit_data)
        self.back_btn.clicked.connect(self.go_back)


        self.username.selectionChanged.connect(self.clear_error_text)
        self.username.textEdited.connect(self.clear_error_text)


    def go_back(self):
        self.parent().setCurrentIndex(2)


    def submit_data(self):
        if self.password.text() != self.confirm_password.text():
            QMessageBox.warning(self, "Error", "Passwords do not match!")
            return
        if not self.username.text().strip() or not self.password.text():
            QMessageBox.warning(self, "Error", "Username and password cannot be empty!")
            return

        entered_username = self.username.text().strip()


        is_strong, message = check_password_strength(self.password.text())
        if not is_strong:
            QMessageBox.warning(self, "Password Strength", message)
            return


        try:
            with open("user_data.json", "r") as f:
                users = json.load(f)
                if not isinstance(users, list):
                    users = [users]
        except (FileNotFoundError, json.JSONDecodeError):
            users = []


        if any(user.get("username", "").lower() == entered_username.lower() for user in users):
            self.username.clear()
            self.username.setPlaceholderText("Username already taken")
            self.username.setStyleSheet("""
                QLineEdit {
                    border: 2px solid red;
                    border-radius: 15px;
                    padding-left: 15px;
                    color: red;
                    background-color: rgba(255, 255, 255, 0.15);
                }
            """)
            self.username.setFocus()
            return


        self.username.setStyleSheet(LoginPage.input_style(self))
        self.username.setPlaceholderText("Username")

        self.data['username'] = entered_username
        self.data['password'] = self.password.text()
        self.data['profile_picture'] = resource_path("assets/defaultprofile.png")

        role = self.data.get('role')
        if role == 'Trainer':
            self.data['tobechosenclients'] = []
            self.data['clients'] = []
        elif role == 'User':
            self.data['chosentrainers'] = []
            self.data['trainers'] = []

        users.append(self.data)

        try:
            with open("user_data.json", "w") as f:
                json.dump(users, f, indent=4)
            QMessageBox.information(self, "Success", "Account created and data saved!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data:\n{str(e)}")
            return

        self.parent().welcome_page.update_data()
        self.parent().setCurrentIndex(4)

class WelcomePage(QWidget):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.data = data if data is not None else {}
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(50, 50, 50, 50)
        main_layout.setSpacing(40)


        self.image_label = QLabel()
        icon = QIcon(resource_path("assets/welcome_image.png"))
        pixmap = icon.pixmap(600, 600)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label, stretch=1)


        right_container = QWidget()
        right_layout = QVBoxLayout()


        right_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        right_layout.setSpacing(24)


        self.welcome_label = QLabel("Welcome!")
        self.welcome_label.setFont(QFont(sf_family, 36, QFont.Weight.Bold))
        self.welcome_label.setStyleSheet("color: white;")
        right_layout.addWidget(self.welcome_label)


        self.role_label = QLabel("Role: Unknown")
        self.role_label.setFont(QFont(sf_family, 24))
        self.role_label.setStyleSheet("color: white;")
        right_layout.addWidget(self.role_label)


        intro_text = (
            "Welcome to FitnessTracker!\n\n"
            "Track your workouts, monitor your progress, and stay motivated "
            "with personalized fitness insights.\n\n"
            "Let's achieve your goals together!"
        )
        self.intro_label = QLabel(intro_text)
        self.intro_label.setWordWrap(True)
        self.intro_label.setFont(QFont(sf_family, 16))
        self.intro_label.setStyleSheet("color: #cccccc;")
        right_layout.addWidget(self.intro_label)





        self.continue_btn = QPushButton("Continue →")
        self.continue_btn.setFixedSize(200, 60)
        self.continue_btn.setFont(QFont(sf_family, 20))
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #00c6ff, stop:1 #0072ff);
                border-radius: 30px;
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #33ccff, stop:1 #0099ff);
            }
            QPushButton:pressed {
                background-color: #005bb5;
            }
        """)
        self.continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        right_layout.addWidget(self.continue_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        right_container.setLayout(right_layout)
        right_container.setMaximumWidth(500)

        main_layout.addWidget(right_container, stretch=2)
        self.setLayout(main_layout)


        self.continue_btn.clicked.connect(lambda: self.parent().setCurrentIndex(0))


    def update_data(self):
        first_name = self.data.get("first_name", "User")
        role = self.data.get("role", "Unknown")
        self.welcome_label.setText(f"Welcome, {first_name}!")
        self.role_label.setText(f"Role: {role}")

class CustomWorkoutCalendar(QWidget):
    dateSelected = pyqtSignal(QDate)

    def __init__(self, workout_data: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workout_data = workout_data
        self.setStyleSheet("background:transparent;")

        self.setFixedSize(700, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.top_bar = QHBoxLayout()
        self.top_bar.setContentsMargins(10, 0, 10, 0)

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.setStyleSheet("color:white; background:transparent; border:none; font-size:18px;")
        self.prev_btn.setFixedSize(32, 32)

        self.next_btn = QPushButton("▶")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setStyleSheet("color:white; background:transparent; border:none; font-size:18px;")
        self.next_btn.setFixedSize(32, 32)

        self.month_label = QLabel()
        self.month_label.setFont(QFont(sf_family, 20, QFont.Weight.Bold))
        self.month_label.setStyleSheet("color:white;")
        self.month_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.month_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.month_label.mousePressEvent = self.handle_month_year_click

        self.top_bar.addWidget(self.prev_btn)
        self.top_bar.addStretch()
        self.top_bar.addWidget(self.month_label)
        self.top_bar.addStretch()
        self.top_bar.addWidget(self.next_btn)
        layout.addLayout(self.top_bar)

        self.grid = QGridLayout()
        self.grid.setSpacing(0)
        layout.addLayout(self.grid)

        self.current_date = QDate.currentDate()
        self.selected_date = self.current_date

        self.prev_btn.clicked.connect(self.goto_prev_month)
        self.next_btn.clicked.connect(self.goto_next_month)

        self.draw_calendar()

    def goto_prev_month(self):
        self.current_date = self.current_date.addMonths(-1)
        self.draw_calendar()

    def goto_next_month(self):
        self.current_date = self.current_date.addMonths(1)
        self.draw_calendar()

    def draw_calendar(self):
        self.month_label.setText(self.current_date.toString("MMMM yyyy"))

        for i in reversed(range(self.grid.count())):
            item = self.grid.itemAt(i)
            if item.widget():
                item.widget().setParent(None)

        for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            label = QLabel(day)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color:white;font-weight:bold;padding:4px;")
            self.grid.addWidget(label, 0, i)

        first_day = QDate(self.current_date.year(), self.current_date.month(), 1)
        start_column = (first_day.dayOfWeek() + 6) % 7
        days_in_month = first_day.daysInMonth()

        row = 1
        col = 0

        prev_month = self.current_date.addMonths(-1)
        prev_days = QDate(prev_month.year(), prev_month.month(), 1).daysInMonth()
        for i in range(start_column):
            date = QDate(prev_month.year(), prev_month.month(), prev_days - start_column + i + 1)
            self.grid.addWidget(self.create_day_cell(date, is_other_month=True), row, col)
            col += 1

        for day in range(1, days_in_month + 1):
            if col > 6:
                col = 0
                row += 1
            date = QDate(self.current_date.year(), self.current_date.month(), day)
            self.grid.addWidget(self.create_day_cell(date), row, col)
            col += 1

        next_day = 1
        while col <= 6:
            date = QDate(self.current_date.addMonths(1).year(), self.current_date.addMonths(1).month(), next_day)
            self.grid.addWidget(self.create_day_cell(date, is_other_month=True), row, col)
            next_day += 1
            col += 1

    def create_day_cell(self, date: QDate, is_other_month=False):
        day_str = date.toString("yyyy-MM-dd")
        is_selected = date == self.selected_date
        workout_count = len(self.workout_data.get(day_str, []))

        bg_color = "#3e3e3e" if is_other_month else "#1e1e2e"
        text_color = "#aaa" if is_other_month else "white"

        w = QWidget()
        w.setFixedSize(700 // 7, 700 // 7)
        w.setStyleSheet(f"""
            background-color: {bg_color};
            border: 0.5px solid rgba(255, 255, 255, 0.1);
        """)

        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        day_label = QLabel(str(date.day()))
        day_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        day_label.setFixedSize(24, 24)

        if is_selected:
            day_label.setStyleSheet("""
                background-color: white;
                color: black;
                font-weight: bold;
                border-radius: 12px;
            """)
        else:
            day_label.setStyleSheet(f"""
                color: {text_color};
                font-weight: bold;
                background: transparent;
                border: none;
            """)

        layout.addWidget(day_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        if workout_count > 0:
            counts = {"Completed": 0, "Planned": 0, "Skipped": 0}
            for log in self.workout_data.get(day_str, []):
                status = log.get("status", "Planned")
                if status in counts:
                    counts[status] += 1


            row = QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)
            row.setAlignment(Qt.AlignmentFlag.AlignCenter)

            def create_circle(count, color):
                if count <= 0:
                    return None
                label = QLabel(str(count))
                label.setFixedSize(22, 22)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet(f"""
                    background-color: {color};
                    color: white;
                    font-weight: bold;
                    font-size: 10pt;
                    border-radius: 11px;
                """)
                return label

            completed = create_circle(counts["Completed"], "#4CAF50")
            planned   = create_circle(counts["Planned"], "#FFA000")
            skipped   = create_circle(counts["Skipped"], "#F44336")

            for item in [completed, planned, skipped]:
                if item:
                    row.addWidget(item)

            layout.addLayout(row)




        def handle_click(event):
            self.select_date(date)

        w.mousePressEvent = handle_click
        return w

    def select_date(self, date: QDate):
        self.selected_date = date
        self.draw_calendar()
        self.dateSelected.emit(date)

    def handle_month_year_click(self, event):

        font_metrics = self.month_label.fontMetrics()
        month_str = self.current_date.toString("MMMM")
        month_width = font_metrics.horizontalAdvance(month_str)

        if event.pos().x() < month_width + 10:
            self.open_month_dropdown()
        else:
            self.edit_year_input()

    def open_month_dropdown(self):
        menu = QMenu(self)
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        for i, name in enumerate(months, 1):
            action = QAction(name, self)
            action.triggered.connect(lambda _, m=i: self.set_month(m))
            menu.addAction(action)
        menu.popup(self.month_label.mapToGlobal(self.month_label.rect().bottomLeft()))

    def set_month(self, month: int):
        self.current_date = QDate(self.current_date.year(), month, 1)
        self.draw_calendar()

    def edit_year_input(self):
        current_year = self.current_date.year()

        dlg = QDialog(self)
        dlg.setWindowTitle("Change Year")
        dlg.setWindowIcon(QIcon(resource_path("assets/schedule2.png")))

        dlg.setStyleSheet("background-color:#1e1e2e; color:white;")

        layout = QVBoxLayout(dlg)

        label = QLabel("Enter Year:")
        label.setStyleSheet("font-size:12pt;")
        layout.addWidget(label)

        spin = QSpinBox()
        spin.setRange(1000, 9999)
        spin.setValue(current_year)
        spin.setStyleSheet("background:#333; color:white; font-size:11pt;")
        layout.addWidget(spin)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet("background:#444; color:white;")
        btn_box.button(QDialogButtonBox.StandardButton.Cancel).setStyleSheet("background:#444; color:white;")
        layout.addWidget(btn_box)

        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_year = spin.value()
            self.current_date = QDate(new_year, self.current_date.month(), 1)
            self.draw_calendar()

class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" ")
        self.setWindowIcon(QIcon(resource_path("assets/fitnessicon.ico")))

        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        self.showFullScreen()

        self.setStyleSheet("background-color: #121212;")

        self.data = {}

        self.login_page = LoginPage(self)
        self.create_account_page = CreateAccountPage(self)
        self.role_selection_page = RoleSelectionPage(self)
        self.account_credentials_page = AccountCredentialsPage(data=self.data, parent=self)
        self.welcome_page = WelcomePage(data=self.data, parent=self)
        self.dashboard_page = DashboardPage({})

        self.addWidget(self.login_page)
        self.addWidget(self.create_account_page)
        self.addWidget(self.role_selection_page)
        self.addWidget(self.account_credentials_page)
        self.addWidget(self.welcome_page)
        self.addWidget(self.dashboard_page)

        QTimer.singleShot(100, self.create_bar)


        self.installEventFilter(self)


        QApplication.instance().focusChanged.connect(self.on_focus_changed)

    def create_bar(self):
        self.bar = DraggableBar(main_window = self)
        self.bar.show()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMinimized:
                if hasattr(self, 'bar'):
                    self.bar.hide()
            else:
                if hasattr(self, 'bar'):
                    self.bar.show()

    def eventFilter(self, obj, event):

        if obj == self:
            if event.type() == QEvent.Type.FocusOut:
                if hasattr(self, 'bar'):
                    self.bar.hide()
            elif event.type() == QEvent.Type.FocusIn:
                if hasattr(self, 'bar') and not (self.windowState() & Qt.WindowState.WindowMinimized):
                    self.bar.show()
        return super().eventFilter(obj, event)


    def on_focus_changed(self, old, new):
        if not hasattr(self, 'bar'):
            return


        any_active = any(
            w.isActiveWindow() and w.isVisible()
            for w in QApplication.topLevelWidgets()
            if isinstance(w, QWidget)
        )

        if any_active:

            if not (self.windowState() & Qt.WindowState.WindowMinimized):
                self.bar.show()
        else:

            self.bar.hide()

class DashboardPage(QWidget):
    def __init__(self, user_data=None):
        super().__init__()
        self.setWindowTitle("Dashboard")
        self.setMinimumSize(1000, 600)
        self.user_data = user_data
        self.button_list = []
        self.tab_animation = None
        self.anim_out = None
        self.anim_in = None
        self.current_index = 0
        self.pages = []

        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)


        self.top_container = QWidget()
        self.top_container.setFixedHeight(110)
        self.top_container.setStyleSheet("background-color: #1e1e2f;")
        self.main_layout.addWidget(self.top_container)


        self.top_layout = QVBoxLayout(self.top_container)
        self.top_layout.setContentsMargins(10, 10, 10, 0)
        self.top_layout.setSpacing(0)


        self.button_row = QWidget()
        self.button_layout = QHBoxLayout(self.button_row)
        self.button_layout.setContentsMargins(0, 20, 0, 0)
        self.button_layout.setSpacing(10)
        self.top_layout.addWidget(self.button_row)


        self.tab_indicator = QWidget(self.top_container)
        self.tab_indicator.setFixedHeight(4)
        self.tab_indicator.setStyleSheet("background-color: #00BCD4; border-radius: 2px;")
        self.tab_indicator.move(0, self.top_container.height() - 4)
        self.tab_indicator.show()


        self.content_area = QStackedWidget()
        self.content_area.setStyleSheet("background-color: #2e2e3e; color: white;")
        self.main_layout.addWidget(self.content_area)

        if user_data:
            self.update_data(user_data)

    def update_data(self, user_data):
        self.user_data = user_data
        self.clear_buttons()
        role = user_data.get("role", "").lower()

        if role == "trainer":
            self.setup_trainer_dashboard()
        elif role == "user":
            self.setup_user_dashboard()
        else:
            self.add_content_page("Unknown role")

    def clear_buttons(self):
        for btn in self.button_list:
            self.button_layout.removeWidget(btn)
            btn.deleteLater()
        self.button_list.clear()
        self.clear_content_area()

    def clear_content_area(self):
        while self.content_area.count():
            widget = self.content_area.widget(0)
            self.content_area.removeWidget(widget)
            widget.deleteLater()
        self.pages.clear()
        self.current_index = 0

    def setup_user_dashboard(self) -> None:
        def circular(path: str, d: int = 44) -> QPixmap:
            pm = QPixmap(path).scaled(d, d, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                    Qt.TransformationMode.SmoothTransformation)
            out = QPixmap(d, d); out.fill(Qt.GlobalColor.transparent)
            p = QPainter(out); p.setRenderHint(QPainter.RenderHint.Antialiasing)
            circle = QPainterPath(); circle.addEllipse(0, 0, d, d)
            p.setClipPath(circle); p.drawPixmap(0, 0, pm); p.end()
            return out

        def flexible_block(cards: list[QWidget]) -> QWidget:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            for card in cards:
                layout.addWidget(card)
            layout.addStretch()
            return container

        def wrap_scroll(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(widget)
            scroll.setStyleSheet("""
                QScrollArea { background-color: transparent; }
                QScrollBar:vertical { background:#1e1e2f;width:8px;border-radius:4px; }
                QScrollBar::handle:vertical { background:#555;border-radius:4px;min-height:40px; }
                QScrollBar::add-line,QScrollBar::sub-line{height:0;}
            """)
            return scroll

        def create_trainer_tab() -> QWidget:
            class SignalEmitter(QObject):
                message_signal = pyqtSignal(str, str, object, str)
                typing_signal = pyqtSignal(bool)
                trainer_list_changed_signal = pyqtSignal()
            emitter = SignalEmitter()


            previous_trainers_hash = None
            previous_chosentrainers_hash = None
            def trainer_card(tr: dict, pending: bool, on_remove) -> QWidget:
                f = QFrame()
                f.setFixedHeight(60)


                shadow_effect = QGraphicsDropShadowEffect()
                shadow_effect.setOffset(0, 4)
                shadow_effect.setBlurRadius(10)
                shadow_effect.setColor(Qt.GlobalColor.black)
                f.setGraphicsEffect(shadow_effect)

                f.setStyleSheet("""
                    QFrame {
                        background: rgba(10, 117, 194, 0.4);  /* 40% opacity */
                        border-radius: 8px;
                    }
                """)

                h = QHBoxLayout(f)
                h.setContentsMargins(12, 0, 12, 0)
                h.setSpacing(8)


                avatar = QLabel()
                avatar_path = tr.get("profile_picture", "assets/defaultprofile.png")
                avatar_full_path = get_image_path(avatar_path)
                avatar.setPixmap(QIcon(avatar_full_path).pixmap(44, 44))
                avatar.setFixedSize(44, 44)
                avatar.setStyleSheet("background: transparent;")
                h.addWidget(avatar)


                name = QLabel(f"{tr['first_name']} {tr['last_name']}")
                name.setStyleSheet("color:#fff;font-weight:600;font-size:13pt;background: transparent;")
                h.addWidget(name, 1)

                if pending:

                    tag = QLabel("Pending")
                    tag.setStyleSheet("color:#ffc107;font-weight:600;background: transparent;")
                    h.addWidget(tag)


                remove_btn = QPushButton()
                remove_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                remove_btn.setIconSize(QSize(20, 20))
                remove_btn.setFixedSize(32, 32)
                remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                remove_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: rgba(255, 82, 82, 0.3);
                        border-radius: 4px;
                    }
                """)
                remove_btn.clicked.connect(lambda: on_remove(tr))
                h.addWidget(remove_btn)

                chat_btn = QPushButton("Chat")
                chat_btn.setIcon(QIcon(resource_path("assets/chat.png")))
                chat_btn.setIconSize(QSize(20, 20))
                chat_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        color: #fff;
                        padding: 10px 15px;  /* Add padding */
                        font-size: 14px;      /* Increase font size */
                        border-radius: 8px;   /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: rgba(0, 188, 212, 0.2);
                    }
                    QPushButton:pressed {
                        background-color: rgba(0, 188, 212, 0.4);
                    }
                """)
                chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                chat_btn.clicked.connect(lambda: open_chat_dialog(tr))
                h.addWidget(chat_btn)


                return f


            class MessageBubble(QWidget):

                long_pressed = pyqtSignal(str, str, datetime, str, QPoint)

                def __init__(self, message: str, is_my_message: bool, timestamp=None, status=None, message_id=None, edited=False):
                    super().__init__()
                    self.message = message
                    self.is_my_message = is_my_message
                    self.timestamp = timestamp
                    self.status = status
                    self.message_id = message_id
                    self.edited = edited

                    self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
                    self.opacity_effect = QGraphicsOpacityEffect(self)
                    self.opacity_effect.setOpacity(1.0)
                    self.setGraphicsEffect(self.opacity_effect)

                    self.label = QLabel(message, self)
                    self.label.setWordWrap(True)
                    self.label.setStyleSheet(f"""
                        QLabel {{
                            color: {'#FFFFFF' if is_my_message else '#D1D1D6'};
                            padding: 0px 4px;
                        }}
                    """)
                    self.label.setMaximumWidth(280)
                    self.label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


                    self.time_label = QLabel(self)
                    self.time_label.setStyleSheet("color: #ccc; font-size: 9pt;")
                    self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

                    layout = QVBoxLayout(self)
                    layout.setContentsMargins(14, 8, 14, 20)
                    layout.addWidget(self.label)
                    layout.addWidget(self.time_label, alignment=Qt.AlignmentFlag.AlignRight)


                    self.status_icon_label = None
                    self.time_text_label = None


                    if self.timestamp:
                        self.time_str = self.timestamp.strftime("%I:%M %p").lstrip("0")

                        time_container = QWidget()
                        time_layout = QHBoxLayout(time_container)
                        time_layout.setContentsMargins(0, 0, 0, 0)
                        time_layout.setSpacing(4)

                        self.time_text_label = QLabel(self.time_str)
                        self.time_text_label.setStyleSheet("color: #ccc; font-size: 9pt;")

                        time_layout.addWidget(self.time_text_label)

                        if self.is_my_message and status:
                            icon_path = "assets/seen.png" if status == "seen" else "assets/delivered.png"
                            icon = QIcon(resource_path(icon_path))
                            pixmap = icon.pixmap(16, 16)
                            self.status_icon_label = QLabel()
                            self.status_icon_label.setPixmap(pixmap)
                            time_layout.addWidget(self.status_icon_label)

                        layout.removeWidget(self.time_label)
                        self.time_label.deleteLater()
                        layout.addWidget(time_container, alignment=Qt.AlignmentFlag.AlignRight)


                    if self.edited:

                        self.edited_label = QLabel("Edited", self)
                        self.edited_label.setStyleSheet("""
                            color: #B0B0B0;
                            font-style: italic;
                            font-size: 8pt;
                        """)
                        self.edited_label.setAlignment(Qt.AlignmentFlag.AlignRight)


                        self.edited_label.setContentsMargins(0, 0, 10, 0)


                        layout.addWidget(self.edited_label)


                    self._long_press_timer = QTimer(self)
                    self._long_press_timer.setInterval(700)
                    self._long_press_timer.setSingleShot(True)
                    self._long_press_timer.timeout.connect(self._emit_long_press)
                    self._mouse_press_pos = QPoint()

                def paintEvent(self, event):
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                    rect = self.rect().adjusted(4, 0, -4, -12)

                    if self.is_my_message:
                        gradient = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomLeft()))
                        gradient.setColorAt(0, QColor("#3B82F6"))
                        gradient.setColorAt(1, QColor("#0A84FF"))
                        painter.setBrush(QBrush(gradient))
                    else:
                        gradient = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomLeft()))
                        gradient.setColorAt(0, QColor("#646464"))
                        gradient.setColorAt(1, QColor("#494949"))
                        painter.setBrush(QBrush(gradient))

                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(rect, 12, 12)

                    tail = QPolygonF()
                    if self.is_my_message:
                        base = rect.right() - 20
                        tail.append(QPointF(base, rect.bottom()))
                        tail.append(QPointF(base + 10, rect.bottom()))
                        tail.append(QPointF(base + 5, rect.bottom() + 10))
                    else:
                        base = rect.left() + 20
                        tail.append(QPointF(base, rect.bottom()))
                        tail.append(QPointF(base - 10, rect.bottom()))
                        tail.append(QPointF(base - 5, rect.bottom() + 10))

                    painter.drawPolygon(tail)

                def set_status(self, status):
                    self.status = status
                    if self.is_my_message and self.status_icon_label:
                        icon_path = "assets/seen.png" if status == "seen" else "assets/delivered.png"
                        icon = QIcon(resource_path(icon_path))
                        pixmap = icon.pixmap(16, 16)
                        self.status_icon_label.setPixmap(pixmap)

                def mousePressEvent(self, event):
                    if event.button() == Qt.MouseButton.LeftButton:

                        self.opacity_effect.setOpacity(0.85)
                        self._mouse_press_pos = event.globalPosition()
                        self._long_press_timer.start()
                    super().mousePressEvent(event)

                def mouseReleaseEvent(self, event):
                    if event.button() == Qt.MouseButton.LeftButton:

                        self.opacity_effect.setOpacity(1.0)
                        self._long_press_timer.stop()
                    super().mouseReleaseEvent(event)

                def mouseMoveEvent(self, event):

                    if self._long_press_timer.isActive() and \
                            (event.globalPosition() - self._mouse_press_pos).manhattanLength() > 10:
                        self._long_press_timer.stop()
                    super().mouseMoveEvent(event)

                def _emit_long_press(self):

                    sender = "me" if self.is_my_message else "other"
                    self.long_pressed.emit(self.message, sender, self.timestamp, self.status, self.mapToGlobal(self.rect().center()))
                    self.opacity_effect.setOpacity(1.0)

                def update_message_text(self, new_text):
                    self.message = new_text
                    self.label.setText(new_text)
                    self.label.adjustSize()


                    if self.edited:

                        if not hasattr(self, 'edited_label'):
                            self.edited_label = QLabel("Edited", self)
                            self.edited_label.setStyleSheet("""
                                color: #B0B0B0;
                                font-style: italic;
                                font-size: 8pt;
                            """)
                            self.edited_label.setAlignment(Qt.AlignmentFlag.AlignRight)


                            self.edited_label.setContentsMargins(0, 0, 10, 0)
                            self.layout().addWidget(self.edited_label)
                    else:

                        if hasattr(self, 'edited_label'):
                            self.edited_label.deleteLater()
                            del self.edited_label



                    self.parentWidget().layout().invalidate()
                    self.update()

            def open_chat_dialog(receiver):
                emitter = SignalEmitter()
                last_message_date = None

                message_widgets = {}
                message_data_map = {}

                me_username = self.user_data.get("username")
                receiver_username = receiver["username"]
                chat_id = "_".join(sorted([me_username, receiver_username]))
                messages_ref = db.collection("chats").document(chat_id).collection("messages")

                dlg = QDialog(self)
                dlg.setWindowTitle(f"Chat with {receiver['first_name']} {receiver['last_name']}")
                dlg.setFixedSize(400, 600)
                dlg.setStyleSheet("background:#2e2e3e; color:#fff; border:none;")
                dlg.setWindowIcon(QIcon(resource_path("assets/chat.png")))

                v = QVBoxLayout(dlg)
                v.setContentsMargins(12, 12, 12, 12)
                v.setSpacing(10)

                message_box = QScrollArea()
                message_box.setWidgetResizable(True)
                message_box.setStyleSheet("""
                    QScrollArea {
                        border: none;
                        background: transparent;
                    }
                    QScrollBar:vertical {
                        background: transparent;
                        width: 8px;
                        margin: 0px 0px 0px 0px;
                    }
                    QScrollBar::handle:vertical {
                        background: #888;
                        min-height: 20px;
                        border-radius: 4px;
                    }
                    QScrollBar::handle:vertical:hover {
                        background: #555;
                    }
                    QScrollBar::add-line, QScrollBar::sub-line {
                        height: 0px;
                    }
                    QScrollBar::add-page, QScrollBar::sub-page {
                        background: none;
                    }
                """)

                message_area = QWidget()
                message_layout = QVBoxLayout(message_area)
                message_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
                message_area.setStyleSheet("background: transparent; border: none;")
                message_box.setWidget(message_area)

                v.addWidget(message_box)
                typing_label = QLabel("")
                typing_label.setStyleSheet("color: #aaa; font-size: 12px; margin-left: 6px;")
                typing_label.setVisible(False)
                v.addWidget(typing_label,alignment=Qt.AlignmentFlag.AlignLeft)
                typing_status_ref = db.collection("chats").document(chat_id).collection("status").document(me_username)
                typing_timer = QTimer()
                typing_timer.setInterval(1500)
                typing_timer.setSingleShot(True)
                wave_timer = QTimer()
                wave_timer.setInterval(300)
                wave_frame = 0

                dot_wave_frames = [
                    ". . .",
                    "• . .",
                    ". • .",
                    ". . •",
                    ". • .",
                    "• . .",
                ]

                message_input = QLineEdit()
                message_input.setPlaceholderText("Type your message...")
                message_input.setStyleSheet("""
                    QLineEdit {
                        background: #1e1e2f;
                        border: none;
                        border-radius: 6px;
                        padding: 8px 12px;
                        color: #fff;
                        font-size: 14px;
                    }
                """)

                send_btn = QPushButton()
                send_btn.setIcon(QIcon(resource_path("assets/send4.png")))
                send_btn.setIconSize(QSize(32, 32))
                send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                send_btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        border: none;
                        padding: 3px;
                    }
                    QPushButton:hover {
                        background: rgba(255, 255, 255, 0.08);
                        border-radius: 12px;
                    }
                """)


                local_messages = []

                def hyphenate_text(text, font, max_width):
                    metrics = QFontMetrics(font)
                    words = text.split()
                    result = []
                    padding_correction = 12 * 2

                    for word in words:
                        if metrics.horizontalAdvance(word) + padding_correction <= max_width:
                            result.append(word)
                        else:
                            split_word = ""
                            current = ""
                            for char in word:
                                if metrics.horizontalAdvance(current + char + '-') + padding_correction > max_width:
                                    split_word += current + "-\n"
                                    current = char
                                else:
                                    current += char
                            split_word += current
                            result.append(split_word)
                    return " ".join(result)

                class DateLabel(QLabel):
                    def __init__(self, text, parent=None):
                        super().__init__(text, parent)
                        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.setFont(QFont(sf_family, 10, QFont.Weight.Medium))
                        self.setStyleSheet("color: #fff; padding: 4px 12px;")

                        fm = QFontMetrics(self.font())
                        text_width = fm.horizontalAdvance(text)
                        self.setFixedSize(text_width + 24, fm.height() + 12)

                    def paintEvent(self, event):
                        painter = QPainter(self)
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                        rect = self.rect().adjusted(0, 0, 0, 0)

                        gradient = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomLeft()))
                        gradient.setColorAt(0, QColor("#5c5c5c"))
                        gradient.setColorAt(1, QColor("#3a3a3a"))

                        painter.setBrush(QBrush(gradient))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawRoundedRect(rect, 10, 10)

                        super().paintEvent(event)


                def create_date_label(text):
                    label = DateLabel(text)
                    container = QWidget()
                    layout = QHBoxLayout(container)
                    layout.setContentsMargins(0, 10, 0, 10)
                    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(label)
                    return container

                def handle_message_long_press(message_text, is_my_message, message_timestamp, current_status, global_pos):

                    if not is_my_message:
                        return

                    menu = QMenu()


                    edit_icon = QIcon(resource_path("assets/edit4.png"))
                    delete_icon = QIcon(resource_path("assets/delete.png"))


                    edit_action = menu.addAction(edit_icon, "Edit Message")
                    delete_action = menu.addAction(delete_icon, "Delete Message")


                    action = menu.exec(global_pos)

                    if action == edit_action:
                        edit_message(message_text, message_timestamp, me_username, chat_id)
                    elif action == delete_action:
                        delete_message(message_text, message_timestamp, me_username, chat_id)

                def create_message_bubble(message: str, is_my_message: bool, timestamp: datetime, status=None, message_id=None, edited=False):
                    font = QFont()
                    display_message = hyphenate_text(message, font, 280)

                    bubble = MessageBubble(display_message, is_my_message, timestamp, status, message_id, edited)


                    bubble.long_pressed.connect(handle_message_long_press)

                    container = QWidget()
                    layout = QHBoxLayout(container)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(0)

                    if is_my_message:
                        layout.addStretch()
                        layout.addWidget(bubble)
                    else:
                        layout.addWidget(bubble)
                        layout.addStretch()

                    return container, bubble



                def update_ui(text, sender, timestamp, status="delivered", message_id=None, edited=False):
                    nonlocal last_message_date, message_widgets, message_data_map

                    if message_layout is None or message_layout.parent() is None:
                        return


                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)

                    local_tz = timezone(timedelta(hours=4))
                    local_timestamp = timestamp.astimezone(local_tz)

                    msg_date = local_timestamp.date()
                    label_text = format_date_label(local_timestamp)

                    if message_id is None:
                        actual_key = (local_timestamp.isoformat(), sender, text)
                    else:
                        actual_key = message_id

                    if actual_key in message_widgets:
                        bubble = message_widgets[actual_key]

                        if bubble.status != status:
                            bubble.set_status(status)


                        if bubble.message != text:
                            bubble.edited = edited
                            bubble.update_message_text(text)
                            if actual_key in message_data_map:
                                message_data_map[actual_key]["text"] = text
                                message_data_map[actual_key]["edited"] = edited
                        if actual_key in message_data_map:
                            message_data_map[actual_key]["status"] = status
                    else:
                        if last_message_date != msg_date:
                            date_label = create_date_label(label_text)
                            message_layout.addWidget(date_label)
                            last_message_date = msg_date

                        is_me = sender == me_username
                        print(edited)
                        msg_container, bubble = create_message_bubble(text, is_me, local_timestamp, status, message_id, edited)

                        message_widgets[actual_key] = bubble
                        message_data_map[actual_key] = {
                            "text": text,
                            "sender": sender,
                            "timestamp": local_timestamp.isoformat(),
                            "status": status,
                            "message_id": message_id,
                            "edited": edited
                        }

                        message_layout.addWidget(msg_container)

                    QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))




                local_messages = load_local_chat(me_username, receiver_username)
                for msg in local_messages:
                    try:
                        ts = datetime.fromisoformat(msg["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except Exception:
                        ts = datetime.now(timezone.utc)

                    update_ui(msg["text"], msg["sender"], ts, msg.get("status", "delivered"), msg.get("message_id"),msg.get("edited"))

                QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))


                last_local_ts = None
                if local_messages:
                    try:

                        last_local_ts = max(datetime.fromisoformat(m["timestamp"]) for m in local_messages)
                    except Exception:
                        last_local_ts = None

                def load_missing_messages():
                    query = messages_ref.order_by("timestamp")
                    if last_local_ts:
                        query = query.start_after([last_local_ts])

                    docs = query.get()
                    for doc in docs:
                        doc_dict = doc.to_dict()
                        message_id = doc.id

                        text = doc_dict.get("text", "")
                        sender = doc_dict.get("sender", "")
                        status = doc_dict.get("status", "delivered")
                        timestamp = doc_dict.get("timestamp")
                        edited = doc_dict.get("edited", False)

                        if hasattr(timestamp, "ToDatetime"):
                            timestamp = timestamp.ToDatetime()
                        elif isinstance(timestamp, dict) and "_seconds" in timestamp:
                            timestamp = datetime.fromtimestamp(timestamp["_seconds"], tz=timezone.utc)
                        elif timestamp is None:
                            timestamp = datetime.now(timezone.utc)

                        iso_ts = timestamp.isoformat()
                        key = (iso_ts, sender, text)


                        exists_locally = any(
                            m.get("timestamp") == iso_ts and m.get("text") == text and m.get("sender") == sender
                            for m in local_messages
                        )
                        if not exists_locally:
                            new_msg = {
                                "text": text,
                                "sender": sender,
                                "timestamp": iso_ts,
                                "status": status,
                                "message_id": message_id,
                                "edited": edited
                            }
                            local_messages.append(new_msg)
                            save_local_chat(me_username, receiver_username, local_messages)
                            update_ui(text, sender, timestamp, status, message_id, edited)

                    QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))

                load_missing_messages()

                def mark_seen_for_incoming_messages():
                    updated = False
                    for msg in local_messages:
                        if msg["sender"] != me_username and msg.get("status") != "seen":
                            msg["status"] = "seen"
                            updated = True


                            try:

                                if "message_id" in msg and msg["message_id"]:
                                    messages_ref.document(msg["message_id"]).update({"status": "seen"})
                                else:

                                    query = messages_ref \
                                        .where(filter=FieldFilter("timestamp", "==", datetime.fromisoformat(msg["timestamp"]))) \
                                        .where(filter=FieldFilter("sender", "==", msg["sender"])) \
                                        .get()

                                    for doc in query:
                                        try:
                                            messages_ref.document(doc.id).update({"status": "seen"})
                                        except Exception as e:
                                            print("Failed to update status for doc:", doc.id, "Error:", e)

                            except Exception as outer_e:
                                print("Failed to query/update messages for status update:", outer_e)

                    if updated:
                        save_local_chat(me_username, receiver_username, local_messages)

                mark_seen_for_incoming_messages()


                def send_message():
                    message = message_input.text().strip()
                    if message:
                        utc_now = datetime.now(timezone.utc)


                        try:
                            doc_ref = messages_ref.add({
                                "text": message,
                                "sender": me_username,
                                "timestamp": utc_now,
                                "status": "delivered"
                            })
                            message_id = doc_ref[1].id
                        except Exception as e:
                            print(f"Error adding message to Firebase: {e}")
                            message_id = None

                        msg_data = {
                            "text": message,
                            "sender": me_username,
                            "timestamp": utc_now.isoformat(),
                            "status": "delivered",
                            "message_id": message_id,
                            "edited": False
                        }

                        local_messages.append(msg_data)
                        save_local_chat(me_username, receiver_username, local_messages)


                        update_ui(message, me_username, utc_now, "delivered", message_id,False)
                        QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))

                        message_input.clear()

                def send_typing_status():
                    try:
                        typing_status_ref.set({"typing": True}, merge=True)
                        typing_timer.start()
                    except Exception as e:
                        print("Failed to send typing status:", e)

                def stop_typing_status():
                    try:
                        typing_status_ref.set({"typing": False}, merge=True)
                    except Exception as e:
                        print("Failed to stop typing status:", e)

                typing_timer.timeout.connect(stop_typing_status)

                def handle_user_typing():
                    if not typing_timer.isActive():
                        send_typing_status()
                    typing_timer.start()

                message_input.textChanged.connect(handle_user_typing)

                send_btn.clicked.connect(send_message)

                input_layout = QHBoxLayout()
                input_layout.setContentsMargins(0, 0, 0, 0)
                input_layout.setSpacing(6)
                input_layout.addWidget(message_input)
                input_layout.addWidget(send_btn)

                v.addLayout(input_layout)


                listener_unsubscribe = None
                typing_unsubscribe = None

                def handle_typing(is_typing):
                    nonlocal wave_frame
                    if is_typing:
                        typing_label.setVisible(True)
                        def update_wave():
                            nonlocal wave_frame
                            dots = dot_wave_frames[wave_frame % len(dot_wave_frames)]
                            typing_label.setText(f"{receiver['first_name']} is typing {dots}")
                            wave_frame += 1

                        try:
                            wave_timer.timeout.disconnect()
                        except TypeError:
                            pass
                        wave_timer.timeout.connect(update_wave)
                        wave_timer.start()
                    else:
                        wave_timer.stop()
                        typing_label.setText("")
                        typing_label.setVisible(False)

                def listen_to_messages():
                    def on_snapshot(col_snapshot, changes, read_time):
                        for change in changes:

                            doc = change.document.to_dict()
                            message_id = change.document.id

                            if not doc:
                                continue

                            text = doc.get("text", "")
                            sender = doc.get("sender", "")
                            status = doc.get("status", "delivered")
                            timestamp = doc.get("timestamp")
                            edited = doc.get("edited", False)


                            if hasattr(timestamp, "ToDatetime"):
                                timestamp = timestamp.ToDatetime()
                            elif isinstance(timestamp, dict) and "_seconds" in timestamp:
                                timestamp = datetime.fromtimestamp(timestamp["_seconds"], tz=timezone.utc)
                            elif timestamp is None:
                                timestamp = datetime.now(timezone.utc)

                            iso_ts = timestamp.isoformat()
                            key = (iso_ts, sender, text)


                            if 'type' in dir(change):
                                if change.type == 'ADDED':

                                    exists_locally = any(
                                        m.get("timestamp") == iso_ts and m.get("sender") == sender and m.get("text") == text
                                        for m in local_messages
                                    )
                                    if not exists_locally:
                                        local_messages.append({
                                            "text": text,
                                            "sender": sender,
                                            "timestamp": iso_ts,
                                            "status": status,
                                            "message_id": message_id,
                                            "edited": edited
                                        })
                                        save_local_chat(me_username, receiver_username, local_messages)
                                        emitter.message_signal.emit(text, sender, timestamp, status, message_id, edited)
                                        if sender != me_username:
                                            QTimer.singleShot(0, mark_seen_for_incoming_messages)

                                elif change.type == 'MODIFIED':

                                    for i, msg in enumerate(local_messages):
                                        if msg.get("message_id") == message_id:
                                            local_messages[i]["text"] = text
                                            local_messages[i]["status"] = status
                                            save_local_chat(me_username, receiver_username, local_messages)

                                            emitter.message_signal.emit(text, sender, timestamp, status, message_id, edited)
                                            break

                                    for ui_key, bubble_widget in message_widgets.items():
                                        if ui_key[0] == iso_ts and ui_key[1] == sender and ui_key[2] == bubble_widget.message:
                                            bubble_widget.update_message_text(text)
                                            bubble_widget.set_status(status)
                                            break

                                elif change.type == 'REMOVED':
                                    message_id = change.document.id


                                    local_messages[:] = [msg for msg in local_messages if msg.get("message_id") != message_id]
                                    save_local_chat(me_username, receiver_username, local_messages)


                                    if message_id in message_data_map:
                                        del message_data_map[message_id]

                                    if message_id in message_widgets:
                                        bubble_widget = message_widgets.pop(message_id)
                                        if bubble_widget:
                                            container_widget = bubble_widget.parentWidget().parentWidget()
                                            if container_widget:
                                                message_layout.removeWidget(container_widget)
                                                container_widget.setParent(None)
                                                container_widget.deleteLater()


                            if sender == receiver_username:
                                QTimer.singleShot(0, lambda: wave_timer.stop())
                                QTimer.singleShot(0, lambda: typing_label.setText(""))
                                QTimer.singleShot(0, lambda: typing_label.setVisible(False))

                    return messages_ref.order_by("timestamp").on_snapshot(on_snapshot)

                def listen_to_typing_status():
                    other_user_status_ref = db.collection("chats").document(chat_id).collection("status").document(receiver_username)

                    def on_status_snapshot(doc_snapshot, changes, read_time):
                        for doc in doc_snapshot:
                            data = doc.to_dict()
                            if data and data.get("typing", False):
                                emitter.typing_signal.emit(True)
                            else:
                                emitter.typing_signal.emit(False)

                    return other_user_status_ref.on_snapshot(on_status_snapshot)






                def edit_message(original_text, timestamp, sender, chat_id):
                    current_msg_data = None
                    message_id = None


                    for i, msg in enumerate(local_messages):
                        if msg.get("text") == original_text and msg.get("sender") == sender:
                            current_msg_data = msg
                            message_id = msg.get("message_id")
                            break

                    if not current_msg_data:
                        QMessageBox.warning(dlg, "Error", "Could not find message to edit locally.")
                        return


                    new_text, ok = QInputDialog.getText(dlg, "Edit Message", "Enter new message:",
                                                        QLineEdit.EchoMode.Normal, original_text)

                    if ok and new_text and new_text != original_text:
                        if current_msg_data:

                            current_msg_data["text"] = new_text
                            current_msg_data["edited"] = True

                            message_data_map[message_id]["text"] = new_text


                            print("setting edited to true")
                            message_data_map[message_id]["edited"] = True


                            update_ui(new_text, sender, timestamp, current_msg_data["status"], message_id,edited = True)


                            save_local_chat(me_username, receiver_username, local_messages)


                        try:
                            if message_id:

                                messages_ref.document(message_id).update({
                                    "text": new_text,
                                    "edited": True
                                })
                            else:

                                query = messages_ref \
                                    .where(filter=FieldFilter("timestamp", "==", timestamp)) \
                                    .where(filter=FieldFilter("sender", "==", sender)) \
                                    .get()
                                for doc in query:
                                    messages_ref.document(doc.id).update({
                                        "text": new_text,
                                        "edited": True
                                    })
                                    break
                        except Exception as e:
                            QMessageBox.critical(dlg, "Firebase Error", f"Failed to edit message in Firebase: {e}")




                def delete_message(message_text, timestamp, sender, chat_id):
                    reply = QMessageBox.question(
                        dlg,
                        "Delete Message",
                        "Are you sure you want to delete this message?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )

                    if reply == QMessageBox.StandardButton.Yes:
                        message_id_to_delete = None


                        for msg in local_messages:

                            if msg.get("text") == message_text and msg.get("sender") == sender:
                                message_id_to_delete = msg.get("message_id")
                                break

                        if not message_id_to_delete:
                            print("Error: Could not find message_id to delete.")
                            return




                        if message_id_to_delete in message_data_map:
                            del message_data_map[message_id_to_delete]

                        if message_id_to_delete in message_widgets:
                            bubble_widget = message_widgets.pop(message_id_to_delete)
                            if bubble_widget:

                                container_widget = bubble_widget.parentWidget().parentWidget()
                                if container_widget:
                                    message_layout.removeWidget(container_widget)
                                    container_widget.setParent(None)
                                    container_widget.deleteLater()


                        local_messages[:] = [msg for msg in local_messages if msg.get("message_id") != message_id_to_delete]
                        save_local_chat(me_username, receiver_username, local_messages)


                        try:
                            messages_ref.document(message_id_to_delete).delete()
                        except Exception as e:
                            QMessageBox.critical(dlg, "Firebase Error", f"Failed to delete message from Firebase: {e}")




                emitter.message_signal.connect(update_ui)
                emitter.typing_signal.connect(handle_typing)
                listener_unsubscribe = listen_to_messages()
                typing_unsubscribe = listen_to_typing_status()

                def cleanup():
                    emitter.message_signal.disconnect(update_ui)
                    wave_timer.stop()
                    if listener_unsubscribe:
                        listener_unsubscribe.unsubscribe()
                    if typing_unsubscribe:
                        typing_unsubscribe.unsubscribe()
                    try:
                        typing_status_ref.set({"typing": False}, merge=True)
                    except:
                        pass

                dlg.finished.connect(cleanup)
                dlg.exec()

            def top_aligned_label(text: str) -> QWidget:
                lbl = QLabel(text)
                lbl.setStyleSheet("color:gray;font-size:13pt;")
                lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
                wrapper = QWidget()
                v = QVBoxLayout(wrapper)
                v.setContentsMargins(10, 10, 10, 10)
                v.setAlignment(Qt.AlignmentFlag.AlignTop)
                v.addWidget(lbl)
                return wrapper

            page = QWidget()
            root = QVBoxLayout(page)
            root.setContentsMargins(20, 20, 20, 20)
            root.setSpacing(16)
            root.setAlignment(Qt.AlignmentFlag.AlignTop)
            page.setStyleSheet("background:#2e2e3e;color:#fff;")

            title = QLabel("My Trainers")
            title.setFont(QFont(sf_family, 20, QFont.Weight.Bold))
            root.addWidget(title)

            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.setSpacing(0)

            tab_buttons = QWidget()
            tab_buttons.setStyleSheet("""
                border-bottom: 1px solid #444;
            """)
            tab_btn_layout = QHBoxLayout(tab_buttons)
            tab_btn_layout.setContentsMargins(0, 0, 0, 0)
            tab_btn_layout.setSpacing(0)
            tab_btn_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)


            pending_btn = QPushButton("Pending Acceptance")
            accepted_btn = QPushButton("Accepted Trainers")
            for btn in (pending_btn, accepted_btn):
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        color: #ccc;
                        font-weight: 600;
                        padding: 8px 14px;
                        border: none;
                    }
                    QPushButton:hover {
                        color: #fff;
                    }
                """)

            tab_btn_layout.addWidget(pending_btn)
            tab_btn_layout.addWidget(accepted_btn)



            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            trainers = [u for u in users if u.get("role", "").lower() == "trainer"]
            def compute_hash(lst):
                """Helper function to compute hash of a list."""
                return hashlib.sha256(str(lst).encode('utf-8')).hexdigest()

            def detect_trainer_list_changes():
                nonlocal previous_trainers_hash, previous_chosentrainers_hash


                current_trainers = me.get("trainers", [])
                current_chosentrainers = me.get("chosentrainers", [])


                current_trainers_hash = compute_hash(current_trainers)
                current_chosentrainers_hash = compute_hash(current_chosentrainers)


                if current_trainers_hash != previous_trainers_hash or current_chosentrainers_hash != previous_chosentrainers_hash:

                    emitter.trainer_list_changed_signal.emit()


                    previous_trainers_hash = current_trainers_hash
                    previous_chosentrainers_hash = current_chosentrainers_hash


            def setup_change_detection_timer():
                timer = QTimer()
                timer.setInterval(500)
                timer.timeout.connect(detect_trainer_list_changes)
                timer.start()
            def get_trainer_blocks():
                pending_usernames = [u for u in me.get("chosentrainers", []) if u not in me.get("trainers", [])]
                accepted_usernames = me.get("trainers", [])

                pending = [t for t in trainers if t["username"] in pending_usernames]
                accepted = [t for t in trainers if t["username"] in accepted_usernames]

                def remove_pending(tr):

                    if tr["username"] in me.get("chosentrainers", []):
                        me["chosentrainers"].remove(tr["username"])

                    if me["username"] in tr.get("tobechosenclients", []):
                        tr["tobechosenclients"].remove(me["username"])

                    save_user_data()
                    refresh_blocks(stay_on_index=0)


                def remove_accepted(tr):

                    if tr["username"] in me.get("trainers", []):
                        me["trainers"].remove(tr["username"])

                    if me["username"] in tr.get("clients", []):
                        tr["clients"].remove(me["username"])

                    save_user_data()
                    refresh_blocks(stay_on_index=1)

                pb = wrap_scroll(flexible_block([trainer_card(t, True, remove_pending) for t in pending])) if pending else top_aligned_label("No pending trainers.")
                ab = wrap_scroll(flexible_block([trainer_card(t, False, remove_accepted) for t in accepted])) if accepted else top_aligned_label("No accepted trainers.")
                return pb, ab


            def save_user_data():
                try:
                    Path("user_data.json").write_text(json.dumps(users, indent=4))
                except Exception as e:
                    print("Failed saving user data:", e)

            def refresh_blocks(stay_on_index):
                nonlocal pending_block, accepted_block
                new_pending, new_accepted = get_trainer_blocks()
                stack.removeWidget(pending_block); pending_block.deleteLater()
                stack.removeWidget(accepted_block); accepted_block.deleteLater()
                stack.insertWidget(0, new_pending)
                stack.insertWidget(1, new_accepted)
                pending_block, accepted_block = new_pending, new_accepted
                stack.setCurrentIndex(stay_on_index)
                switch_tab(stay_on_index)

            pending_block, accepted_block = get_trainer_blocks()

            stack_container = QWidget()
            stack = QStackedLayout(stack_container)
            stack.setContentsMargins(0, 0, 0, 0)
            stack.addWidget(pending_block)
            stack.addWidget(accepted_block)

            tab_layout.addWidget(tab_buttons)
            tab_layout.addSpacing(12)
            tab_layout.addWidget(stack_container)

            def switch_tab(index: int):
                stack.setCurrentIndex(index)
                self.last_trainer_subtab_index = index

                for i, btn in enumerate((pending_btn, accepted_btn)):
                    if not isinstance(btn, QPushButton):
                        continue
                    if i == index:
                        btn.setStyleSheet("""
                            QPushButton {
                                background: transparent;
                                color: white;
                                font-weight: 600;
                                font-size: 14pt;
                                padding: 10px 18px;
                                border: none;
                                border-bottom: 5px solid #ffffff;
                            }
                        """)
                    else:
                        btn.setStyleSheet("""
                            QPushButton {
                                background: transparent;
                                color: #aaa;
                                font-weight: 500;
                                font-size: 13pt;
                                padding: 10px 18px;
                                border: none;
                            }
                            QPushButton:hover {
                                color: white;
                            }
                        """)




            pending_btn.clicked.connect(lambda: switch_tab(0))
            accepted_btn.clicked.connect(lambda: switch_tab(1))

            self.last_trainer_subtab_index = 0
            self.trainer_tab_switcher = switch_tab

            panel = QWidget()
            plo = QVBoxLayout(panel)
            plo.setContentsMargins(16, 0, 16, 16)
            plo.setSpacing(6)
            plo.setAlignment(Qt.AlignmentFlag.AlignTop)
            plo.addWidget(tab_widget)
            root.addWidget(panel)

            choose_btn = QPushButton("Choose Trainer(s)")
            choose_btn.setFixedSize(180, 38)
            choose_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            choose_btn.setStyleSheet("""
                QPushButton {
                    background-color: #5C6BC0;
                    color: white;
                    padding: 10px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3F51B5;
                }
            """)
            root.addWidget(choose_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

            def open_dialog() -> None:
                dlg = QDialog(self)
                dlg.setWindowTitle("Select Trainers")
                dlg.setFixedSize(460, 620)
                dlg.setStyleSheet("background:#1e1e2f;color:#fff;")
                dlg.setWindowIcon(QIcon(resource_path("assets/trainer.png")))
                v = QVBoxLayout(dlg)
                v.setContentsMargins(14, 14, 14, 14)
                v.setSpacing(12)

                search = QLineEdit()
                search.setPlaceholderText("Search trainers…")
                search.setStyleSheet("""
                    QLineEdit {
                        background:#2e2e3e;
                        border:1px solid #555;
                        border-radius:6px;
                        padding:7px;
                        color:#fff;
                    }
                """)
                v.addWidget(search)

                lst = QListWidget()
                lst.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
                lst.setMinimumHeight(420)
                lst.setStyleSheet("""
                    QListWidget::item {
                        background:#2e2e3e;
                        margin:3px;
                        padding:10px;
                        border-radius:6px;
                        color:#fff;
                    }
                    QListWidget::item:selected {
                        background:#00BCD4;
                        color:#000;
                    }
                    QScrollBar:vertical {
                        background:#1e1e2f;
                        width:8px;
                        border-radius:4px;
                    }
                    QScrollBar::handle:vertical {
                        background:#555;
                        border-radius:4px;
                        min-height:40px;
                    }
                    QScrollBar::add-line, QScrollBar::sub-line {
                        height:0;
                    }
                """)
                v.addWidget(lst)

                def populate(txt: str = "") -> None:
                    txt = txt.lower().strip()
                    lst.clear()
                    already = set(me.get("chosentrainers", []) + me.get("trainers", []))
                    for t in trainers:
                        fullname = f"{t['first_name']} {t['last_name']}"
                        if txt and txt not in fullname.lower() and txt not in t["username"].lower():
                            continue
                        it = QListWidgetItem()
                        cb = QCheckBox(fullname)
                        cb.setStyleSheet("""
                                            QCheckBox::indicator {
                                                width: 18px;
                                                height: 18px;
                                                border-radius: 9px; /* half of width/height for a perfect circle */
                                            }
                                            QCheckBox::indicator:unchecked {
                                                border: 2px solid #00BCD4;
                                                background: transparent;
                                            }
                                            QCheckBox::indicator:checked {
                                                background-color: #00BCD4;
                                                border: 2px solid #00BCD4;
                                            }
                                            QCheckBox {
                                                color: #fff;
                                            }
                                        """)


                        if t["username"] in already:
                            cb.setChecked(True)
                            cb.setEnabled(False)
                        lst.addItem(it)
                        lst.setItemWidget(it, cb)
                        it.setSizeHint(QSize(lst.viewport().width(), max(cb.sizeHint().height(), 50)))

                populate()
                search.textChanged.connect(populate)

                add = QPushButton("Add Trainer(s)")
                add.setFixedHeight(44)
                add.setStyleSheet("""
                    QPushButton {
                        background:#00BCD4;
                        border-radius:8px;
                        font-weight:600;
                    }
                    QPushButton:hover {
                        background:#0097a7;
                    }
                """)
                v.addWidget(add)

                def save_selection() -> None:
                    new_fullnames = [
                        lst.itemWidget(lst.item(i)).text()
                        for i in range(lst.count())
                        if lst.itemWidget(lst.item(i)).isChecked() and lst.itemWidget(lst.item(i)).isEnabled()
                    ]
                    new_usernames = [
                        t["username"] for t in trainers for full in new_fullnames
                        if full == f"{t['first_name']} {t['last_name']}"
                    ]


                    me.setdefault("chosentrainers", [])
                    for u in new_usernames:
                        if u not in me["chosentrainers"] and u not in me.get("trainers", []):
                            me["chosentrainers"].append(u)


                    for t in trainers:
                        if t["username"] in new_usernames:
                            t.setdefault("tobechosenclients", [])

                            if me["username"] not in t["tobechosenclients"]:
                                t["tobechosenclients"].append(me["username"])


                    try:
                        Path("user_data.json").write_text(json.dumps(users, indent=4))
                    except Exception as e:
                        print("write user_data.json:", e)


                    nonlocal pending_block, accepted_block
                    new_pending, new_accepted = get_trainer_blocks()
                    stack.removeWidget(pending_block); pending_block.deleteLater()
                    stack.removeWidget(accepted_block); accepted_block.deleteLater()
                    stack.insertWidget(0, new_pending); stack.insertWidget(1, new_accepted)
                    pending_block, accepted_block = new_pending, new_accepted
                    stack.setCurrentIndex(0)
                    switch_tab(0)

                    dlg.accept()


                add.clicked.connect(save_selection)

                dlg.exec()


            choose_btn.clicked.connect(open_dialog)

            switch_tab(self.last_trainer_subtab_index)
            emitter.trainer_list_changed_signal.connect(refresh_blocks)
            setup_change_detection_timer()
            return page

        def create_workout_log_tab() -> QWidget:
            filter_applied = [False]
            logs_per_page = 6
            current_page = [1]
            last_filtered = [None]
            workout_categories = {
                "Cardio": ["Running", "Jogging", "Walking", "Rowing", "Jump Rope"],
                "Cycling": ["Road Cycling", "Mountain Biking", "Stationary Bike"],
                "Strength Training": ["Weight Lifting", "Bodyweight Exercises", "Resistance Bands"],
                "Flexibility": ["Yoga", "Stretching", "Pilates"],
                "Sports": ["Basketball", "Soccer", "Tennis", "Swimming"],
                "Other": []
            }

            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            me.setdefault("workout_logs", [])

            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(12)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            title_bar = QWidget()
            title_layout = QHBoxLayout(title_bar)
            title_layout.setContentsMargins(0, 0, 0, 0)
            title_layout.setSpacing(4)

            title = QLabel("Workout Log")
            title.setFont(QFont(sf_family, 20, QFont.Weight.Bold))
            title.setStyleSheet("color:#fff;")

            filter_btn = QPushButton()
            filter_btn.setIcon(QIcon(resource_path("assets/filter.png")))
            filter_btn.setIconSize(QSize(40, 40))
            filter_btn.setFixedSize(48, 48)
            filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            filter_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background: transparent;
                }
                QPushButton:hover {
                    background-color: rgba(255,255,255,0.1);
                    border-radius:6px;
                }
                """)

            def open_filter_dialog():
                dlg = QDialog(self)
                dlg.setWindowTitle("Filter Workout Logs")
                dlg.setFixedSize(320, 320)
                dlg.setStyleSheet("background:#2e2e3e;color:#fff;")
                dlg.setWindowIcon(QIcon(resource_path("assets/filter.png")))
                v = QVBoxLayout(dlg)
                v.setContentsMargins(16, 16, 16, 16)
                v.setSpacing(10)

                category_box = QComboBox()
                category_box.addItem("All")
                category_box.addItems(workout_categories.keys())
                category_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                v.addWidget(QLabel("Filter by Exercise Type"))
                v.addWidget(category_box)

                sub_box = QComboBox()
                sub_box.addItem("All")
                sub_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                v.addWidget(QLabel("Filter by Exercise"))
                v.addWidget(sub_box)

                def update_subtypes():
                    sub_box.clear()
                    sub_box.addItem("All")
                    subs = workout_categories.get(category_box.currentText(), [])
                    if subs:
                        sub_box.addItems(subs)
                    else:
                        sub_box.addItem("Other")

                category_box.currentTextChanged.connect(update_subtypes)
                update_subtypes()


                start_date = QDateEdit()
                start_date.setCalendarPopup(True)
                start_date.setDisplayFormat("yyyy-MM-dd")
                start_date.setDate(QDate.currentDate().addMonths(-1))
                start_date.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                end_date = QDateEdit()
                end_date.setCalendarPopup(True)
                end_date.setDisplayFormat("yyyy-MM-dd")
                end_date.setDate(QDate.currentDate())
                end_date.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                v.addWidget(QLabel("Start Date"))
                v.addWidget(start_date)
                v.addWidget(QLabel("End Date"))
                v.addWidget(end_date)

                def apply_filter():
                    selected_cat = category_box.currentText()
                    selected_sub = sub_box.currentText()
                    start = start_date.date().toString("yyyy-MM-dd")
                    end = end_date.date().toString("yyyy-MM-dd")

                    def in_range(date_str):
                        return start <= date_str <= end

                    filtered = [
                        log for log in me["workout_logs"]
                        if (selected_cat == "All" or log["category"] == selected_cat) and
                        (selected_sub == "All" or log["exercise"] == selected_sub) and
                        in_range(log["date"])
                    ]

                    last_filtered[0] = filtered
                    refresh_logs(filtered)

                    clear_filter_btn.setVisible(True)
                    filter_applied[0] = True

                    dlg.accept()

                apply_btn = QPushButton("Apply Filter")
                apply_btn.setStyleSheet("background:#00BCD4;border-radius:6px;padding:8px;font-weight:bold;")
                apply_btn.clicked.connect(apply_filter)
                v.addWidget(apply_btn)

                dlg.exec()



            filter_btn.clicked.connect(open_filter_dialog)
            title_layout.addWidget(title)
            title_layout.addStretch()

            clear_filter_btn = QPushButton()
            clear_filter_btn.setIcon(QIcon(resource_path("assets/clear.png")))
            clear_filter_btn.setIconSize(QSize(16, 16))
            clear_filter_btn.setFixedSize(20, 20)

            clear_filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            clear_filter_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background: transparent;
                }
                QPushButton:hover {
                    background-color: rgba(255,255,255,0.1);
                    border-radius:10px;
                }
            """)
            clear_filter_btn.setVisible(False)

            def clear_filter():
                last_filtered[0] = None
                refresh_logs()
                clear_filter_btn.setVisible(False)
                filter_applied[0] = False
                highlight_buttons(None)

            clear_filter_btn.clicked.connect(clear_filter)


            filter_wrap = QWidget()
            filter_layout = QGridLayout(filter_wrap)
            filter_layout.setContentsMargins(0, 0, 0, 0)
            filter_layout.setSpacing(0)


            filter_layout.addWidget(filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            filter_layout.addWidget(clear_filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

            title_layout.addWidget(filter_wrap)
            layout.addWidget(title_bar)

            quick_row = QHBoxLayout()
            quick_row.setSpacing(6)
            quick_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
            def highlight_buttons(btn):
                for i in range(quick_row.count()):
                    w = quick_row.itemAt(i).widget()
                    w.setStyleSheet("background:#444;color:#fff;padding:4px 12px;border-radius:6px;")
                if btn:
                    btn.setStyleSheet("background:#00BCD4;color:#fff;padding:4px 12px;border-radius:6px;font-weight:bold;")

            def apply_date_range(days):
                cutoff = QDate.currentDate().addDays(-days)
                logs = me.get("workout_logs", [])
                filtered = [l for l in logs if QDate.fromString(l['date'], "yyyy-MM-dd") >= cutoff]
                last_filtered[0] = filtered
                filter_applied[0] = True
                clear_filter_btn.setVisible(True)
                highlight_buttons(None)
                current_page[0] = 1
                refresh_logs(filtered)

            for label, days in [("Day", 1), ("Week", 7), ("Month", 30), ("Year", 365)]:
                b = QPushButton(f"Past {label}")
                b.clicked.connect(lambda _, d=days, btn=b: [apply_date_range(d), highlight_buttons(btn)])
                quick_row.addWidget(b)

            highlight_buttons(None)
            layout.addLayout(quick_row)

            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)
            container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            scroll = wrap_scroll(container)
            layout.addWidget(scroll)

            pagination_layout = QHBoxLayout()
            pagination_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            prev_btn = QPushButton("←")
            next_btn = QPushButton("→")
            page_input = QLineEdit("1")
            page_input.setFixedWidth(40)
            page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
            page_input.setValidator(QIntValidator(1, 999999999))
            page_input.setStyleSheet("""
                QLineEdit {
                    color: #fff;
                    background: #222;
                    border-radius: 4px;
                    padding: 2px 4px;
                }
            """)

            total_pages_lbl = QLabel("/ 1")
            total_pages_lbl.setStyleSheet("color:#ccc;")

            for btn in (prev_btn, next_btn):
                btn.setFixedSize(32, 32)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet("""
                    QPushButton {
                        background: #5C6BC0;
                        color: #fff;
                        border-radius: 6px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background: #3F51B5;
                    }
                """)

            prev_btn.clicked.connect(lambda: [
                current_page.__setitem__(0, max(1, current_page[0] - 1)),
                refresh_logs(last_filtered[0] if last_filtered[0] else None)
            ])
            next_btn.clicked.connect(lambda: [
                current_page.__setitem__(0, current_page[0] + 1),
                refresh_logs(last_filtered[0] if last_filtered[0] else None)
            ])
            page_input.returnPressed.connect(lambda: [
                current_page.__setitem__(0, int(page_input.text()) if page_input.text().isdigit() else 1),
                refresh_logs(last_filtered[0] if last_filtered[0] else None)
            ])

            pagination_layout.addWidget(prev_btn)
            pagination_layout.addWidget(page_input)
            pagination_layout.addWidget(total_pages_lbl)
            pagination_layout.addWidget(next_btn)

            pagination_widget = QWidget()
            pagination_widget.setLayout(pagination_layout)
            pagination_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
            layout.addWidget(pagination_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
            def delete_log(log: dict):
                me["workout_logs"].remove(log)
                try:
                    Path("user_data.json").write_text(json.dumps(users, indent=4))
                except Exception as e:
                    print("write user_data.json:", e)
                refresh_logs()

            def open_edit_log_dialog(log: dict):
                dlg = create_log_dialog("Edit Workout Log", log)

                dlg.exec()


            def log_card(log: dict) -> QWidget:
                f = QFrame()
                f.setStyleSheet("background:#37474F;border-radius:10px;")
                f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                shadow_effect = QGraphicsDropShadowEffect()
                shadow_effect.setOffset(0, 4)
                shadow_effect.setBlurRadius(10)
                semi_transparent_white = QColor(0, 0, 0, int(255 * 0.3))
                shadow_effect.setColor(semi_transparent_white)
                f.setGraphicsEffect(shadow_effect)

                h = QVBoxLayout(f)
                h.setContentsMargins(12, 8, 12, 8)


                title_row = QHBoxLayout()
                title_lbl = QLabel(f"{log['category']} - {log['exercise']}")
                title_lbl.setStyleSheet("color:#00BCD4;font-size:14pt;font-weight:600;")

                edit_btn = QPushButton()
                edit_btn.setIcon(QIcon(resource_path("assets/edit.png")))
                edit_btn.setIconSize(QSize(20, 20))
                edit_btn.setFixedSize(28, 28)
                edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                edit_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,255,255,0.1);
                    }
                """)
                edit_btn.clicked.connect(lambda: open_edit_log_dialog(log))

                remove_btn = QPushButton()
                remove_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                remove_btn.setIconSize(QSize(20, 20))
                remove_btn.setFixedSize(28, 28)
                remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                remove_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,0,0,0.1);
                    }
                """)
                remove_btn.clicked.connect(lambda: delete_log(log))

                btn_wrap = QHBoxLayout()
                btn_wrap.setSpacing(4)
                btn_wrap.addWidget(edit_btn)
                btn_wrap.addWidget(remove_btn)

                title_row.addWidget(title_lbl)
                title_row.addStretch()
                title_row.addLayout(btn_wrap)
                h.addLayout(title_row)


                def format_duration(dstr: str) -> str:
                    try:
                        h, m, s = map(int, dstr.split(":"))
                        parts = []
                        if h: parts.append(f"{h} hr")
                        if m: parts.append(f"{m} min")
                        if s and not (h or m): parts.append(f"{s} sec")
                        return " ".join(parts) if parts else "0 min"
                    except:
                        return dstr or "N/A"

                duration = format_duration(log.get("duration", "00:00:00"))
                calories = log.get("calories", "N/A")
                weight = log.get("weight", "")
                heart_rate = log.get("heart_rate", "")

                detail_text = f"Date: {log['date']} | Duration: {duration} | Calories: {calories} kcal"
                if weight:
                    detail_text += f" | Weight: {weight}"
                if heart_rate:
                    detail_text += f" | HR: {heart_rate}"

                detail = QLabel(detail_text)
                detail.setStyleSheet("color:#ccc;font-size:10pt;")
                h.addWidget(detail)


                mood = log.get("mood", "")
                intensity = log.get("intensity", "")
                if mood or intensity:
                    extra = []
                    if intensity:
                        extra.append(f"Intensity: {intensity}")
                    if mood:
                        extra.append(f"Mood: {mood}")
                    extra_lbl = QLabel(" | ".join(extra))
                    extra_lbl.setStyleSheet("color:#aaa;font-size:9pt;")
                    h.addWidget(extra_lbl)


                if log.get("notes"):
                    notes = QLabel(f"Notes: {log['notes']}")
                    notes.setStyleSheet("color:#bbb;font-size:9pt;")
                    notes.setWordWrap(True)
                    h.addWidget(notes)

                return f

            def refresh_logs(filtered_logs=None):
                for i in reversed(range(container_layout.count())):
                    w = container_layout.itemAt(i).widget()
                    if w:
                        container_layout.removeWidget(w)
                        w.deleteLater()

                logs = filtered_logs if filtered_logs is not None else me.get("workout_logs", [])


                logs_sorted = sorted(logs, key=lambda x: x.get("date", ""), reverse=True)

                total = max(1, (len(logs_sorted) + logs_per_page - 1) // logs_per_page)
                current_page[0] = min(current_page[0], total)

                if not logs_sorted:
                    message = "No results found." if filtered_logs is not None else "No workouts logged yet."
                    lbl = QLabel(message)
                    lbl.setStyleSheet("color:#888;font-size:11pt;")
                    container_layout.addWidget(lbl)
                else:
                    start = (current_page[0] - 1) * logs_per_page
                    end = start + logs_per_page

                    for log in logs_sorted[start:end]:
                        container_layout.addWidget(log_card(log))

                page_input.setText(str(current_page[0]))
                total_pages_lbl.setText(f"/ {total}")
                prev_btn.setEnabled(current_page[0] > 1)
                next_btn.setEnabled(current_page[0] < total)


            def create_log_dialog(title_text: str, existing: dict = None) -> QDialog:
                dlg = QDialog(self)
                dlg.setWindowTitle(title_text)
                dlg.setStyleSheet("background:#2e2e3e;color:#fff;")
                dlg.setWindowIcon(QIcon(resource_path("assets/workoutlog.png")))

                layout = QVBoxLayout(dlg)
                layout.setContentsMargins(16, 16, 16, 16)
                layout.setSpacing(10)

                grid = QGridLayout()
                grid.setHorizontalSpacing(10)
                grid.setVerticalSpacing(10)


                date_input = QDateEdit(datetime.strptime(existing["date"], "%Y-%m-%d") if existing else datetime.today())
                date_input.setCalendarPopup(True)
                date_input.setDisplayFormat("yyyy-MM-dd")
                date_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                category_box = QComboBox()
                category_box.addItems(workout_categories.keys())
                category_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                category_other_input = QLineEdit()
                category_other_input.setPlaceholderText("Enter exercise type")
                category_other_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                sub_box = QComboBox()
                sub_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                sub_other_input = QLineEdit()
                sub_other_input.setPlaceholderText("Enter exercise")
                sub_other_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                calories_input = QLineEdit()
                calories_input.setPlaceholderText("Calories burned")
                calories_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                duration_input = QTimeEdit()
                duration_input.setDisplayFormat("HH:mm:ss")
                duration_input.setTime(QTime(0, 0, 0))
                duration_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                weight_input = QLineEdit()
                weight_input.setPlaceholderText("Current weight")
                weight_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                hr_input = QLineEdit()
                hr_input.setPlaceholderText("Heart rate")
                hr_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                intensity_box = QComboBox()
                intensity_box.addItems(["", "Low", "Medium", "High"])
                intensity_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                mood_box = QComboBox()
                mood_box.addItems(["", "Energized", "Normal", "Tired", "Exhausted"])
                mood_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")


                notes_input = QTextEdit()
                notes_input.setPlaceholderText("Notes")
                notes_input.setFixedHeight(100)
                notes_input.setStyleSheet("background:#444;color:#fff;border-radius:6px;")


                grid.addWidget(QLabel("Date"), 0, 0)
                grid.addWidget(date_input, 1, 0)
                grid.addWidget(QLabel("Exercise Type"), 0, 1)
                grid.addWidget(category_box, 1, 1)

                grid.addWidget(category_other_input, 2, 0, 1, 2)

                grid.addWidget(QLabel("Exercise"), 3, 0)
                grid.addWidget(sub_box, 4, 0)
                grid.addWidget(QLabel("Calories Burned"), 3, 1)
                grid.addWidget(calories_input, 4, 1)

                grid.addWidget(sub_other_input, 5, 0, 1, 2)

                grid.addWidget(QLabel("Duration"), 6, 0)
                grid.addWidget(duration_input, 7, 0)
                grid.addWidget(QLabel("Current Weight"), 6, 1)
                grid.addWidget(weight_input, 7, 1)

                grid.addWidget(QLabel("Heart Rate"), 8, 0)
                grid.addWidget(hr_input, 9, 0)
                grid.addWidget(QLabel("Intensity"), 8, 1)
                grid.addWidget(intensity_box, 9, 1)

                grid.addWidget(QLabel("Mood"), 10, 0)
                grid.addWidget(mood_box, 11, 0)

                layout.addLayout(grid)
                layout.addWidget(QLabel("Notes"))
                layout.addWidget(notes_input)


                save_btn = QPushButton("Save Log")
                save_btn.setFixedHeight(40)
                save_btn.setStyleSheet("""
                    QPushButton {
                        background:#00BCD4;
                        border-radius:8px;
                        font-weight:bold;
                    }
                    QPushButton:hover {
                        background:#0097a7;
                    }
                """)
                layout.addWidget(save_btn)


                def update_subtypes():
                    sub_box.clear()
                    selected_category = category_box.currentText()
                    subs = workout_categories.get(selected_category, [])
                    if subs:
                        sub_box.addItems(subs)
                    sub_box.addItem("Other")
                    category_other_input.setVisible(selected_category == "Other")
                    sub_other_input.setVisible(sub_box.currentText() == "Other")

                    dlg.resize(520, dlg.sizeHint().height())

                category_box.currentTextChanged.connect(update_subtypes)
                sub_box.currentTextChanged.connect(lambda: (sub_other_input.setVisible(sub_box.currentText() == "Other"), dlg.resize(520, dlg.sizeHint().height())))
                update_subtypes()

                if existing:
                    is_custom_category = existing["category"] not in workout_categories
                    category_box.setCurrentText(existing["category"] if not is_custom_category else "Other")
                    category_other_input.setText(existing["category"] if is_custom_category else "")
                    update_subtypes()
                    subs = workout_categories.get(category_box.currentText(), [])
                    is_custom_sub = existing["exercise"] not in subs
                    sub_box.setCurrentText(existing["exercise"] if not is_custom_sub else "Other")
                    sub_other_input.setText(existing["exercise"] if is_custom_sub else "")
                    calories_input.setText(existing.get("calories", ""))
                    weight_input.setText(existing.get("weight", ""))
                    hr_input.setText(existing.get("heart_rate", ""))
                    intensity_box.setCurrentText(existing.get("intensity", ""))
                    mood_box.setCurrentText(existing.get("mood", ""))
                    notes_input.setPlainText(existing.get("notes", ""))
                    try:
                        h, m, s = map(int, existing.get("duration", "00:00:00").split(":"))
                        duration_input.setTime(QTime(h, m, s))
                    except:
                        duration_input.setTime(QTime(0, 0, 0))



                def save_log():
                    updated = {
                        "date": date_input.date().toString("yyyy-MM-dd"),
                        "category": category_other_input.text() if category_box.currentText() == "Other" else category_box.currentText(),
                        "exercise": sub_other_input.text() if sub_box.currentText() == "Other" else sub_box.currentText(),
                        "calories": calories_input.text().strip(),
                        "duration": duration_input.time().toString("HH:mm:ss"),
                        "weight": weight_input.text().strip(),
                        "heart_rate": hr_input.text().strip(),
                        "intensity": intensity_box.currentText(),
                        "mood": mood_box.currentText(),
                        "notes": notes_input.toPlainText().strip()
                    }

                    if existing:
                        idx = me["workout_logs"].index(existing)
                        me["workout_logs"][idx] = updated
                    else:
                        me.setdefault("workout_logs", []).append(updated)

                    try:
                        Path("user_data.json").write_text(json.dumps(users, indent=4))
                    except Exception as e:
                        print("write user_data.json:", e)

                    dlg.accept()
                    refresh_logs()

                save_btn.clicked.connect(save_log)

                QTimer.singleShot(0, lambda: dlg.resize(520, dlg.sizeHint().height()))
                return dlg



            refresh_logs()


            add_btn = QPushButton("Add Workout Log")
            add_btn.setFixedSize(200, 40)
            add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            add_btn.setStyleSheet("""
                QPushButton {
                    background-color: #5C6BC0;
                    color: white;
                    padding: 10px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3F51B5;
                }
            """)
            layout.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignHCenter)


            add_btn.clicked.connect(lambda: show_non_blocking_dialog(create_log_dialog("New Workout Log")))


            return page

        def create_my_progress_tab() -> QWidget:
            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)

            def create_summary_tab() -> QWidget:

                summary_tab = QWidget()
                summary_tab.setStyleSheet("""
                    background-color: #222;
                    border-top-left-radius: 0px;
                    border-top-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border-bottom-right-radius: 12px;
                """)
                layout = QVBoxLayout(summary_tab)
                layout.setContentsMargins(10, 10, 10, 1)
                layout.setSpacing(16)

                summary_tab_bar = QTabWidget()
                summary_tab_bar.setStyleSheet("""
                    QTabWidget::pane {
                        background: #222;
                        border: none;
                    }

                    QTabBar::tab {
                        background: #222;
                        color: #ccc;
                        font-weight: 600;
                        padding: 10px 20px;
                        border: none;
                        border-bottom: 2px solid transparent;
                    }

                    QTabBar::tab:hover {
                        color: #fff;
                    }

                    QTabBar::tab:selected {
                        color: #fff;
                        border-bottom: 2px solid white;
                    }
                """)





                def group_logs_by_period(logs, period="weekly"):
                    grouped = defaultdict(list)

                    for log in logs:
                        date_str = log.get("date")
                        try:
                            date = datetime.strptime(date_str, "%Y-%m-%d")
                        except:
                            continue

                        if period == "weekly":
                            year, week, _ = date.isocalendar()
                            start_of_week = date - timedelta(days=date.weekday())
                            end_of_week = start_of_week + timedelta(days=6)
                            key = f"Week {week}\n({start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')})"
                        elif period == "monthly":
                            key = date.strftime("%B %Y")
                        else:
                            key = date.strftime("%Y-%m-%d")

                        grouped[key].append(log)

                    return grouped


                def create_chart_tab(metric: str, chart_type: str):
                    w = QWidget()
                    layout = QVBoxLayout(w)
                    layout.setContentsMargins(16, 16, 16, 16)
                    layout.setSpacing(10)

                    control_bar = QHBoxLayout()
                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    combo_style = f"""
                        /* ----- Shared Combo & DateEdit Style ----- */
                        QComboBox, QDateEdit {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 28px 6px 12px;
                            border: 1px solid #555;
                            border-radius: 6px;
                            min-width: 50px;
                        }}
                        QComboBox::drop-down, QDateEdit::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QComboBox::down-arrow, QDateEdit::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}

                        /* ----- Combo Dropdown List ----- */
                        QComboBox QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            border: none;
                            outline: none;
                            padding: 4px;
                        }}

                        /* ----- DateEdit Calendar Popup ----- */
                        QDateEdit::calendar-widget {{
                            background-color: #333;
                            border: 1px solid #555;
                        }}
                        QCalendarWidget QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            selection-color: white;
                        }}
                        QCalendarWidget QWidget {{
                            alternate-background-color: #333;
                            background: #333;
                            color: #fff;
                        }}
                        QCalendarWidget QToolButton {{
                            background-color: #333;
                            color: #fff;
                            border: none;
                            font-weight: bold;
                        }}
                        QCalendarWidget QToolButton:hover {{
                            background-color: #444;
                        }}

                        /* ----- Labels & Buttons for Consistency ----- */
                        QLabel {{
                            color: #ccc;
                            font-weight: bold;
                            margin-right: 4px;
                        }}
                        QPushButton {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 15px;
                            border: 1px solid #555;
                            border-radius: 6px;
                        }}
                        QPushButton:hover {{
                            background-color: #444;
                        }}
                    """
                    toggle = QComboBox()
                    toggle.addItems(["Daily", "Weekly", "Monthly"])
                    toggle.setFixedWidth(160)
                    toggle.setStyleSheet(combo_style)
                    control_bar.addWidget(toggle)
                    control_bar.addStretch()
                    control_bar.addWidget(QLabel("From:"))

                    logs = me.get("workout_logs", [])
                    date_objs = []
                    for log in logs:
                        try:
                            d = datetime.strptime(log["date"], "%Y-%m-%d").date()
                            date_objs.append(d)
                        except:
                            continue


                    min_date = min(date_objs) if date_objs else QDate.currentDate().toPyDate()
                    max_date = max(date_objs) if date_objs else QDate.currentDate().toPyDate()


                    from_date = QDateEdit()
                    from_date.setDate(QDate(min_date.year, min_date.month, min_date.day))
                    from_date.setDisplayFormat("yyyy-MM-dd")
                    from_date.setCalendarPopup(True)
                    from_date.setStyleSheet(combo_style)
                    control_bar.addWidget(from_date)

                    control_bar.addWidget(QLabel("To:"))
                    to_date = QDateEdit(QDate(max_date.year, max_date.month, max_date.day))
                    to_date.setDisplayFormat("yyyy-MM-dd")
                    to_date.setCalendarPopup(True)
                    to_date.setStyleSheet(combo_style)
                    control_bar.addWidget(to_date)

                    refresh_btn = QPushButton("Refresh")

                    refresh_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #5C6BC0;
                            color: white;
                            padding: 10px 24px;
                            border-radius: 8px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #3F51B5;
                        }
                    """)


                    refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                    control_bar.addWidget(refresh_btn)

                    layout.addLayout(control_bar)

                    chart_container = QWidget()
                    chart_layout = QVBoxLayout(chart_container)
                    chart_layout.setContentsMargins(0, 0, 0, 0)
                    layout.addWidget(chart_container)


                    pagination = QHBoxLayout()
                    pagination.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                    prev_btn = QPushButton("←")
                    next_btn = QPushButton("→")

                    for btn in (prev_btn, next_btn):
                        btn.setFixedSize(32, 32)
                        btn.setCursor(Qt.CursorShape.PointingHandCursor)
                        btn.setStyleSheet("""
                            QPushButton {
                                background: #5C6BC0;
                                color: #fff;
                                border-radius: 6px;
                                font-weight: bold;
                            }
                            QPushButton:hover {
                                background: #3F51B5;
                            }
                        """)

                    page_lbl = QLineEdit()
                    page_lbl.setFixedWidth(60)
                    page_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    page_lbl.setStyleSheet("""
                        QLineEdit {
                            background: transparent;
                            color: #ccc;
                            border: none;
                            font-weight: bold;
                        }
                    """)

                    current_page = [0]
                    per_page = 10

                    page_input = QLineEdit("1")
                    page_input.setFixedWidth(40)
                    page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    page_input.setValidator(QIntValidator(1, 999999999))
                    page_input.setStyleSheet("color:#fff;background:#222;border-radius:4px;padding:2px 4px;")

                    total_pages_lbl = QLabel("/ 1")
                    total_pages_lbl.setStyleSheet("color:#ccc;")

                    pagination.addWidget(prev_btn)
                    pagination.addWidget(page_input)
                    pagination.addWidget(total_pages_lbl)
                    pagination.addWidget(next_btn)

                    pagination_widget = QWidget()
                    pagination_widget.setLayout(pagination)
                    pagination_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
                    layout.addWidget(pagination_widget, alignment=Qt.AlignmentFlag.AlignHCenter)


                    total_pages = [1]

                    def paged_refresh():
                        page_input.setText(str(current_page[0] + 1))
                        prev_btn.setEnabled(current_page[0] > 0)
                        next_btn.setEnabled(current_page[0] < total_pages[0] - 1)
                        update_chart()

                    def update_chart():
                        chart = draw_chart(toggle.currentText())

                        while chart_layout.count():
                            item = chart_layout.takeAt(0)
                            widget = item.widget()
                            if widget:
                                widget.setParent(None)
                                widget.deleteLater()


                        chart_layout.addWidget(chart)

                    def draw_chart(period):
                        start = from_date.date().toPyDate()
                        end = to_date.date().toPyDate()

                        filtered = []
                        for log in logs:
                            try:
                                date = datetime.strptime(log["date"], "%Y-%m-%d").date()
                                if start <= date <= end:
                                    filtered.append(log)
                            except:
                                continue

                        grouped = group_logs_by_period(filtered, period=period.lower())
                        x = list(grouped.keys())
                        y = []


                        page_data = x[current_page[0] * per_page: (current_page[0] + 1) * per_page]
                        for group in page_data:
                            if metric == "duration":
                                vals = []
                                for log in grouped[group]:
                                    dstr = log.get("duration", "")
                                    try:
                                        h, m, s = map(int, dstr.split(":"))
                                        total_minutes = h * 60 + m + s / 60
                                        vals.append(total_minutes)
                                    except:
                                        continue
                            else:
                                vals = [float(log.get(metric, 0)) for log in grouped[group] if log.get(metric)]

                            value = (sum(vals) / len(vals)) if metric == "weight" and vals else sum(vals)
                            y.append(value)


                        total_pages[0] = (len(x) + per_page - 1) // per_page
                        total_pages_lbl.setText(f"/ {total_pages[0]}")


                        width = max(6, len(x) * 0.5)
                        dynamic_height = max(3, chart_container.height() / 100)
                        fig = Figure(figsize=(width, dynamic_height), dpi=100, facecolor="#222")
                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.25)

                        ax.set_facecolor("#1E1E2F")
                        ax.tick_params(colors='white', labelsize=8)
                        ax.set_title(f"{metric.capitalize()} ({period})", color="white", fontsize=12, pad=15)
                        ax.spines[:].set_color("white")
                        ax.grid(True, linestyle='--', alpha=0.2)

                        ax.set_xticks(range(len(page_data)))
                        ax.set_xticklabels(page_data, rotation=35, ha='right', fontsize=8, color='white')

                        if chart_type == "bar":
                            bars = ax.bar(range(len(page_data)), y, color="#7C4DFF", width=0.6)
                            for bar in bars:
                                bar.set_linewidth(0)
                                bar.set_alpha(0.9)
                                bar.set_edgecolor("none")
                                bar.set_zorder(3)
                            for i, bar in enumerate(bars):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width() / 2, height + 5, f"{int(height)}",
                                        ha='center', va='bottom', color='white', fontsize=8)

                        else:
                            ax.plot(range(len(page_data)), y, color="#7C4DFF", linewidth=2.5, marker='o',
                                    markerfacecolor="#7C4DFF", markeredgecolor="white", markersize=6)

                        return FigureCanvas(fig)

                    update_chart()

                    toggle.currentIndexChanged.connect(lambda: (current_page.__setitem__(0, 0), paged_refresh()))
                    refresh_btn.clicked.connect(lambda: (current_page.__setitem__(0, 0), paged_refresh()))


                    prev_btn.clicked.connect(lambda: (current_page.__setitem__(0, max(0, current_page[0] - 1)), paged_refresh()))
                    next_btn.clicked.connect(lambda: (current_page.__setitem__(0, min(current_page[0] + 1, total_pages[0] - 1)), paged_refresh()))


                    def handle_page_input():
                        try:
                            new_page = int(page_input.text()) - 1
                            if 0 <= new_page < total_pages[0]:
                                current_page[0] = new_page
                            else:
                                current_page[0] = total_pages[0] - 1
                        except ValueError:
                            current_page[0] = 0
                        paged_refresh()

                    page_input.returnPressed.connect(handle_page_input)

                    return w


                summary_tab_bar.addTab(create_chart_tab("weight", "line"), "Weight")
                summary_tab_bar.addTab(create_chart_tab("calories", "bar"), "Calories Burned")
                summary_tab_bar.addTab(create_chart_tab("duration", "bar"), "Workout Duration")

                layout.addWidget(summary_tab_bar)
                return summary_tab

            def create_comparison_tab() -> QWidget:


                tab = QWidget()
                tab.setStyleSheet("""
                    background-color: #222;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border-bottom-right-radius: 12px;
                """)
                layout = QVBoxLayout(tab)
                layout.setContentsMargins(10, 10, 10, 10)
                layout.setSpacing(10)




                control_bar = QHBoxLayout()
                control_bar.setSpacing(20)

                def field_group(label_text: str, input_widget: QWidget) -> QWidget:
                    container = QWidget()
                    layout = QHBoxLayout(container)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(6)

                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    combo_style = f"""
                        /* ----- Shared Combo & DateEdit Style ----- */
                        QComboBox, QDateEdit {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 12px 6px 12px;
                            border: 1px solid #555;
                            border-radius: 6px;
                            min-width: 50px;
                        }}
                        QComboBox::drop-down, QDateEdit::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QComboBox::down-arrow, QDateEdit::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}

                        /* ----- Combo Dropdown List ----- */
                        QComboBox QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            border: none;
                            outline: none;
                            padding: 4px;
                        }}

                        /* ----- DateEdit Calendar Popup ----- */
                        QDateEdit::calendar-widget {{
                            background-color: #333;
                            border: 1px solid #555;
                        }}
                        QCalendarWidget QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            selection-color: white;
                        }}
                        QCalendarWidget QWidget {{
                            alternate-background-color: #333;
                            background: #333;
                            color: #fff;
                        }}
                        QCalendarWidget QToolButton {{
                            background-color: #333;
                            color: #fff;
                            border: none;
                            font-weight: bold;
                        }}
                        QCalendarWidget QToolButton:hover {{
                            background-color: #444;
                        }}

                        /* ----- Labels & Buttons for Consistency ----- */
                        QLabel {{
                            color: #ccc;
                            font-weight: bold;
                            margin-right: 4px;
                        }}
                        QPushButton {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 15px;
                            border: 1px solid #555;
                            border-radius: 6px;
                        }}
                        QPushButton:hover {{
                            background-color: #444;
                        }}
                    """


                    label = QLabel(label_text)
                    label.setStyleSheet("color: white; font-size: 14px;")
                    label.setFixedWidth(70)

                    input_widget.setFixedSize(220, 30)
                    input_widget.setStyleSheet(combo_style)


                    layout.addWidget(label)
                    layout.addWidget(input_widget)
                    layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                    return container

                period_toggle = QComboBox()
                period_toggle.addItems(["Weekly", "Monthly"])

                metric_toggle = QComboBox()
                metric_toggle.addItems(["Weight", "Calories Burned", "Duration"])

                period_a_combo = QComboBox()

                period_b_combo = QComboBox()


                refresh_btn = QPushButton("Compare")
                refresh_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #5C6BC0;
                        color: white;
                        padding: 10px 12px;
                        border-radius: 8px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3F51B5;
                    }
                """)
                refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                refresh_btn.setFixedWidth(100)
                refresh_btn.setFixedHeight(32)

                control_bar.addWidget(field_group("Period:", period_toggle))
                control_bar.addWidget(field_group("Metric:", metric_toggle))
                control_bar.addWidget(field_group("Period A:", period_a_combo))
                control_bar.addWidget(field_group("Period B:", period_b_combo))
                control_bar.addWidget(refresh_btn)

                layout.addLayout(control_bar)


                chart_area = QVBoxLayout()
                layout.addLayout(chart_area)

                summary_label = QLabel("")
                summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                summary_label.setStyleSheet("font-size:16px;font-weight:bold;")
                layout.addWidget(summary_label)


                def group_logs(logs, mode):
                    grouped = defaultdict(list)
                    for log in logs:
                        try:
                            d = datetime.strptime(log["date"], "%Y-%m-%d")
                        except:
                            continue
                        if mode == "Weekly":
                            y, w, _ = d.isocalendar()
                            start = d - timedelta(days=d.weekday())
                            end = start + timedelta(days=6)
                            label = f"Week {w} ({start.strftime('%b %d')}–{end.strftime('%b %d')}, {y})"
                        else:
                            label = d.strftime("%B %Y")
                        grouped[label].append(log)
                    return grouped

                def update_dropdowns():
                    logs = me.get("workout_logs", [])
                    mode = period_toggle.currentText()
                    grouped = group_logs(logs, mode)

                    def sort_key(label):
                        if mode == "Weekly":
                            match = re.search(r"Week (\d+).*?(\d{4})", label)
                            return int(match.group(2)) * 100 + int(match.group(1)) if match else 0
                        elif mode == "Monthly":
                            try:
                                return datetime.strptime(label, "%B %Y")
                            except:
                                return datetime.min
                        return label

                    sorted_keys = sorted(grouped.keys(), key=sort_key)

                    period_a_combo.clear()
                    period_b_combo.clear()
                    period_a_combo.addItems(sorted_keys)
                    period_b_combo.addItems(sorted_keys)


                def extract_metric(logs, metric):
                    if not logs:
                        return 0
                    values = []
                    for log in logs:
                        if metric == "Duration":
                            try:
                                h, m, s = map(int, log.get("duration", "0:0:0").split(":"))
                                values.append(h * 60 + m + s / 60)
                            except:
                                continue
                        else:
                            try:
                                key = "calories" if metric == "Calories Burned" else metric.lower()
                                val = float(log.get(key, 0))
                                values.append(val)
                            except:
                                continue
                    if metric == "Weight":
                        return sum(values) / len(values) if values else 0
                    return sum(values)

                def draw_chart():
                    logs = me.get("workout_logs", [])
                    mode = period_toggle.currentText()
                    metric = metric_toggle.currentText()
                    period_a = period_a_combo.currentText()
                    period_b = period_b_combo.currentText()

                    grouped = group_logs(logs, mode)
                    a_value = extract_metric(grouped.get(period_a, []), metric)
                    b_value = extract_metric(grouped.get(period_b, []), metric)


                    fig = Figure(figsize=(6.5, 6.5), dpi=100, facecolor="#222")
                    ax = fig.add_subplot(111)
                    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.2)


                    ax.set_facecolor("#1E1E2F")
                    ax.spines[:].set_color("white")
                    ax.tick_params(colors="white", labelsize=9)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.2)


                    ax.set_title(f"{metric} Comparison", color="white", fontsize=14, pad=12)
                    ax.set_ylabel(metric, color="white", fontsize=11)
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["A", "B"], color="white", fontsize=10)


                    bar_colors = ["#7C4DFF", "#9575CD"]
                    bars = ax.bar([0, 1], [a_value, b_value], width=0.5, color=bar_colors)
                    for bar in bars:
                        bar.set_alpha(0.92)
                        bar.set_linewidth(0)
                        bar.set_edgecolor("none")


                    canvas = FigureCanvas(fig)

                    while chart_area.count():
                        child = chart_area.takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
                    chart_area.addWidget(canvas)


                    if a_value == 0:
                        summary = "Not enough data in Period A"
                        summary_label.setStyleSheet("color: orange; font-size:16px; font-weight:bold;")
                    else:
                        change = ((b_value - a_value) / a_value) * 100
                        direction = "increased" if change > 0 else "decreased"
                        color = "#4CAF50" if (change > 0 and metric != "Weight") or (change < 0 and metric == "Weight") else "#FF5252"
                        summary = f"{metric} {direction} by {abs(change):.1f}%"
                        summary_label.setStyleSheet(f"color: {color}; font-size:16px; font-weight:bold;")

                    summary_label.setText(summary)


                period_toggle.currentIndexChanged.connect(update_dropdowns)
                refresh_btn.clicked.connect(draw_chart)


                update_dropdowns()
                draw_chart()

                return tab


            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.setContentsMargins(16, 16, 16, 16)

            tabs = QTabWidget()


            tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: none;
                    background-color: transparent;
                }
                QTabBar::tab {
                    border: none;
                    background: #2e2e3e;
                    color: #ccc;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background: #222;
                    color: #fff;
                    border: none; 
                }
            """)


            tabs.addTab(create_summary_tab(), "Summary")
            tabs.addTab(create_comparison_tab(), "Comparison")


            layout.addWidget(tabs)
            return tab

        def create_workout_schedule_tab() -> QWidget:
            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            me.setdefault("workout_schedule", {})

            workout_categories = {
                "Cardio": ["Running", "Jogging", "Walking", "Rowing", "Jump Rope"],
                "Cycling": ["Road Cycling", "Mountain Biking", "Stationary Bike"],
                "Strength Training": ["Weight Lifting", "Bodyweight Exercises", "Resistance Bands"],
                "Flexibility": ["Yoga", "Stretching", "Pilates"],
                "Sports": ["Basketball", "Soccer", "Tennis", "Swimming"],
                "Other": []
            }

            page = QWidget()
            main_layout = QVBoxLayout(page)
            main_layout.setContentsMargins(20, 20, 20, 20)
            main_layout.setSpacing(12)
            main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            title = QLabel("Workout Schedule")
            title.setFont(QFont(sf_family, 20, QFont.Weight.Bold))
            title.setStyleSheet("color:#fff;")
            main_layout.addWidget(title)

            content_layout = QHBoxLayout()
            content_layout.setSpacing(20)


            calendar_column = QVBoxLayout()
            calendar_column.setAlignment(Qt.AlignmentFlag.AlignTop)

            calendar = CustomWorkoutCalendar(me["workout_schedule"])
            calendar.setMinimumSize(500, 500)
            calendar_column.addWidget(calendar)

            add_btn = QPushButton("Schedule New Workout")
            add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            add_btn.setStyleSheet("""
                QPushButton {
                    background-color: #5C6BC0;
                    color: white;
                    padding: 10px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3F51B5;
                }
            """)

            add_btn.clicked.connect(lambda: open_schedule_dialog("Schedule Workout", None))
            calendar_column.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

            content_layout.addLayout(calendar_column, 2)


            scroll_container = QWidget()
            scroll_layout = QVBoxLayout(scroll_container)
            scroll_layout.setContentsMargins(0, 0, 0, 0)
            scroll_layout.setSpacing(10)
            scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            scroll = wrap_scroll(scroll_container)
            scroll.setMinimumWidth(300)
            scroll.setStyleSheet("""
                QScrollBar:vertical {
                    background: #2e2e3e;
                    width: 10px;
                    margin: 4px 0;
                    border-radius: 5px;
                }
                QScrollBar::handle:vertical {
                    background: #444;
                    min-height: 20px;
                    border-radius: 5px;
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0;
                }
            """)

            content_layout.addWidget(scroll, 1)


            main_layout.addLayout(content_layout)


            def save_schedule():
                try:
                    Path("user_data.json").write_text(json.dumps(users, indent=4))
                except Exception as e:
                    print("save_schedule error:", e)

            def schedule_card(entry: dict, edit_cb, delete_cb) -> QWidget:
                f = QFrame()
                f.setStyleSheet("background:#37474F;border-radius:10px;")
                f.setMinimumHeight(70)
                shadow_effect = QGraphicsDropShadowEffect()
                shadow_effect.setOffset(0, 4)
                shadow_effect.setBlurRadius(10)
                semi_transparent_white = QColor(0, 0, 0, int(255 * 0.3))
                shadow_effect.setColor(semi_transparent_white)
                f.setGraphicsEffect(shadow_effect)

                l = QVBoxLayout(f)
                l.setContentsMargins(12, 8, 12, 8)


                row = QHBoxLayout()
                t = QLabel(f"{entry['category']} - {entry['exercise']}")
                t.setStyleSheet("color:#00BCD4;font-weight:600;font-size:14pt;")
                row.addWidget(t)
                row.addStretch()

                edit_btn = QPushButton()
                edit_btn.setIcon(QIcon(resource_path("assets/edit.png")))
                edit_btn.setFixedSize(24, 24)
                edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                edit_btn.setStyleSheet("border:none;background:transparent;")
                edit_btn.clicked.connect(edit_cb)

                del_btn = QPushButton()
                del_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                del_btn.setFixedSize(24, 24)
                del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                del_btn.setStyleSheet("border:none;background:transparent;")
                del_btn.clicked.connect(delete_cb)

                row.addWidget(edit_btn)
                row.addWidget(del_btn)

                l.addLayout(row)


                details = f"Date: {entry['date']} | Duration: {entry['duration']} | Calories: {entry['calories']} kcal"
                if entry.get("priority"):
                    details += f" | Priority: {entry['priority']}"
                detail_lbl = QLabel(details)
                detail_lbl.setStyleSheet("color:#ccc;font-size:10pt;")
                l.addWidget(detail_lbl)


                loc_line = []
                if entry.get("location"):
                    loc_line.append(f"Location: {entry['location']}")
                if entry.get("equipment"):
                    loc_line.append(f"Equipment: {entry['equipment']}")
                if loc_line:
                    loc_lbl = QLabel(" | ".join(loc_line))
                    loc_lbl.setStyleSheet("color:#aaa;font-size:9pt;")
                    l.addWidget(loc_lbl)


                if entry.get("notes"):
                    notes_lbl = QLabel("Notes: " + entry["notes"])
                    notes_lbl.setStyleSheet("color:#bbb;font-size:9pt;")
                    notes_lbl.setWordWrap(True)
                    l.addWidget(notes_lbl)


                if "status" in entry:
                    status_dropdown = QComboBox()
                    status_dropdown.addItems(["Planned", "Completed", "Skipped"])
                    status_dropdown.setCurrentText(entry["status"])
                    status_dropdown.setCursor(Qt.CursorShape.PointingHandCursor)
                    status_dropdown.setFixedWidth(120)
                    status_dropdown.setStyleSheet("""
                        QComboBox {
                            background: #444;
                            color: white;
                            padding: 4px;
                            border-radius: 6px;
                        }
                        QComboBox::drop-down {
                            border: none;
                        }
                    """)

                    def update_status_color(status: str):
                        color = {
                            "Planned": "#FFA000",
                            "Completed": "#4CAF50",
                            "Skipped": "#F44336"
                        }.get(status, "#777")
                        status_dropdown.setStyleSheet(f"""
                            QComboBox {{
                                background: {color};
                                color: white;
                                padding: 4px;
                                border-radius: 6px;
                            }}
                            QComboBox::drop-down {{
                                border: none;
                            }}
                        """)

                    update_status_color(entry["status"])

                    def save_status_change(new_status: str):
                        entry["status"] = new_status
                        update_status_color(new_status)
                        try:

                            for e in me["workout_schedule"].get(entry["date"], []):
                                if e == entry:
                                    e["status"] = new_status
                                    break
                            Path("user_data.json").write_text(json.dumps(users, indent=4))
                            calendar.workout_data = me["workout_schedule"]
                            calendar.draw_calendar()

                        except Exception as e:
                            print("Failed to update status:", e)

                    status_dropdown.currentTextChanged.connect(save_status_change)
                    l.addWidget(status_dropdown, alignment=Qt.AlignmentFlag.AlignRight)

                return f

            def open_schedule_dialog(title_text: str, existing=None):
                dlg = QDialog(self)
                dlg.setWindowTitle(title_text)
                dlg.setStyleSheet("background:#2e2e3e;color:#fff;")
                dlg.setWindowIcon(QIcon(resource_path("assets/schedule.png")))

                layout = QVBoxLayout(dlg)
                layout.setContentsMargins(16, 16, 16, 16)
                layout.setSpacing(10)

                grid = QGridLayout()
                grid.setHorizontalSpacing(10)
                grid.setVerticalSpacing(10)
                grid.setColumnStretch(0, 1)
                grid.setColumnStretch(1, 1)


                date_input = QDateEdit(QDate.fromString(existing["date"], "yyyy-MM-dd") if existing else calendar.selected_date)
                date_input.setCalendarPopup(True)
                date_input.setDisplayFormat("yyyy-MM-dd")
                date_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                category_box = QComboBox()
                category_box.addItems(workout_categories.keys())
                category_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                category_other_input = QLineEdit()
                category_other_input.setPlaceholderText("Enter exercise type")
                category_other_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                sub_box = QComboBox()
                sub_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                sub_other_input = QLineEdit()
                sub_other_input.setPlaceholderText("Enter exercise")
                sub_other_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                calories_input = QLineEdit()
                calories_input.setPlaceholderText("Calories burned")
                calories_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                duration_input = QTimeEdit()
                duration_input.setDisplayFormat("HH:mm:ss")
                duration_input.setTime(QTime(0, 0, 0))
                duration_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                time_input = QTimeEdit()
                time_input.setDisplayFormat("HH:mm")
                time_input.setTime(QTime.currentTime())
                time_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                location_box = QComboBox()
                location_box.addItems(["", "Home", "Gym", "Outdoors", "Other"])
                location_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                location_other_input = QLineEdit()
                location_other_input.setPlaceholderText("Enter location")
                location_other_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                priority_box = QComboBox()
                priority_box.addItems(["Low", "Medium", "High"])
                priority_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                equipment_display = QTextEdit()
                equipment_display.setPlaceholderText("Equipment (click to select)")
                equipment_display.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                equipment_display.setCursor(Qt.CursorShape.PointingHandCursor)
                equipment_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                equipment_display.setFixedHeight(50)
                equipment_display.setReadOnly(True)

                selected_equipment = []

                def open_equipment_popup():
                    popup = QDialog(dlg)
                    popup.setWindowTitle("Select Equipment")
                    popup.setStyleSheet("background:#2e2e3e;color:#fff;")
                    layout = QVBoxLayout(popup)

                    equipment_options = [
                        "None", "Mat", "Dumbbells", "Treadmill", "Resistance Bands", "Bench", "Stationary Bike", "Foam Roller",
                        "Kettlebell", "Barbell", "Pull-up Bar", "Elliptical", "Jump Rope", "Medicine Ball", "TRX", "Rowing Machine",
                        "Cycle", "Box", "Balance Ball", "Yoga Block", "Step Platform", "Battle Rope", "Punching Bag", "Swim Gear",
                        "Agility Ladder", "Resistance Sled", "Core Wheel", "Weighted Vest"
                    ]

                    checkboxes = {}
                    grid = QGridLayout()

                    for i, eq in enumerate(equipment_options):
                        cb = QCheckBox(eq)
                        cb.setStyleSheet("""
                            QCheckBox::indicator:checked {
                                background-color: #00BCD4;
                                border: 1px solid white;
                            }
                            QCheckBox {
                                padding: 4px;
                                color: white;
                            }
                        """)
                        if eq in selected_equipment:
                            cb.setChecked(True)
                        checkboxes[eq] = cb
                        grid.addWidget(cb, i // 4, i % 4)

                    layout.addLayout(grid)


                    other_input = QLineEdit()
                    other_input.setPlaceholderText("Other")
                    other_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                    layout.addWidget(other_input)

                    if any(e not in equipment_options for e in selected_equipment):
                        other_input.setText(", ".join([e for e in selected_equipment if e not in equipment_options]))


                    def on_none_checked(state):
                        if checkboxes["None"].isChecked():

                            for eq, cb in checkboxes.items():
                                if eq != "None":
                                    cb.blockSignals(True)
                                    cb.setChecked(False)
                                    cb.blockSignals(False)
                            other_input.blockSignals(True)
                            other_input.clear()
                            other_input.blockSignals(False)

                    def on_other_checked():
                        if checkboxes["None"].isChecked():
                            checkboxes["None"].blockSignals(True)
                            checkboxes["None"].setChecked(False)
                            checkboxes["None"].blockSignals(False)

                    def on_other_text_changed():
                        if other_input.text().strip() and checkboxes["None"].isChecked():
                            checkboxes["None"].blockSignals(True)
                            checkboxes["None"].setChecked(False)
                            checkboxes["None"].blockSignals(False)


                    checkboxes["None"].stateChanged.connect(on_none_checked)
                    for eq, cb in checkboxes.items():
                        if eq != "None":
                            cb.stateChanged.connect(on_other_checked)
                    other_input.textChanged.connect(on_other_text_changed)


                    btn_layout = QHBoxLayout()
                    clear_btn = QPushButton("Clear")
                    clear_btn.setStyleSheet("background:#555;color:white;padding:6px;border-radius:6px;")

                    def clear_all():
                        for cb in checkboxes.values():
                            cb.setChecked(False)
                        other_input.clear()

                    clear_btn.clicked.connect(clear_all)

                    save_btn = QPushButton("Save")
                    save_btn.setStyleSheet("background:#00BCD4;padding:6px;border-radius:6px;")

                    def save_equipment():
                        selected_equipment.clear()
                        for eq, cb in checkboxes.items():
                            if cb.isChecked():
                                selected_equipment.append(eq)
                        if other_input.text().strip():
                            selected_equipment.append(other_input.text().strip())
                        equipment_display.setText(", ".join(selected_equipment))
                        popup.accept()

                    save_btn.clicked.connect(save_equipment)
                    btn_layout.addWidget(clear_btn)
                    btn_layout.addStretch()
                    btn_layout.addWidget(save_btn)

                    layout.addLayout(btn_layout)
                    popup.setWindowModality(Qt.WindowModality.NonModal)
                    popup.setWindowFlags(popup.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
                    popup.show()

                equipment_display.mousePressEvent = lambda e: open_equipment_popup()

                notes_input = QTextEdit()
                notes_input.setPlaceholderText("Notes")
                notes_input.setFixedHeight(100)
                notes_input.setStyleSheet("background:#444;color:#fff;border-radius:6px;")


                grid.addWidget(QLabel("Date"), 0, 0)
                grid.addWidget(date_input, 1, 0)
                grid.addWidget(QLabel("Exercise Type"), 0, 1)
                grid.addWidget(category_box, 1, 1)
                grid.addWidget(category_other_input, 2, 0, 1, 2)

                grid.addWidget(QLabel("Exercise"), 3, 0)
                grid.addWidget(sub_box, 4, 0)
                grid.addWidget(QLabel("Calories Burned"), 3, 1)
                grid.addWidget(calories_input, 4, 1)
                grid.addWidget(sub_other_input, 5, 0, 1, 2)

                grid.addWidget(QLabel("Duration"), 6, 0)
                grid.addWidget(duration_input, 7, 0)
                grid.addWidget(QLabel("Time"), 6, 1)
                grid.addWidget(time_input, 7, 1)

                grid.addWidget(QLabel("Location"), 8, 0)
                grid.addWidget(location_box, 9, 0)
                grid.addWidget(QLabel("Priority"), 8, 1)
                grid.addWidget(priority_box, 9, 1)

                grid.addWidget(location_other_input, 10, 0, 1, 2)

                layout.addLayout(grid)
                layout.addWidget(QLabel("Equipment"))
                layout.addWidget(equipment_display)
                layout.addWidget(QLabel("Notes"))
                layout.addWidget(notes_input)


                status_box = None
                if existing:
                    layout.addWidget(QLabel("Status"))
                    status_box = QComboBox()
                    status_box.addItems(["Planned", "Completed", "Skipped"])
                    status_box.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                    status_box.setCurrentText(existing.get("status", "Planned"))
                    layout.addWidget(status_box)


                save_btn = QPushButton("Save Schedule")
                save_btn.setFixedHeight(40)
                save_btn.setStyleSheet("""
                    QPushButton {
                        background:#00BCD4;
                        border-radius:8px;
                        font-weight:bold;
                    }
                    QPushButton:hover {
                        background:#0097a7;
                    }
                """)
                layout.addWidget(save_btn)

                def update_subtypes():
                    sub_box.clear()
                    selected_category = category_box.currentText()
                    subs = workout_categories.get(selected_category, [])
                    if subs:
                        sub_box.addItems(subs)
                    sub_box.addItem("Other")
                    category_other_input.setVisible(selected_category == "Other")
                    sub_other_input.setVisible(sub_box.currentText() == "Other")
                    dlg.resize(520, dlg.sizeHint().height())

                def update_location_visibility():
                    location_other_input.setVisible(location_box.currentText() == "Other")
                    dlg.resize(520, dlg.sizeHint().height())

                category_box.currentTextChanged.connect(update_subtypes)
                sub_box.currentTextChanged.connect(lambda: (sub_other_input.setVisible(sub_box.currentText() == "Other"), dlg.resize(520, dlg.sizeHint().height())))
                location_box.currentTextChanged.connect(update_location_visibility)
                update_subtypes()
                update_location_visibility()

                if existing:

                    category = existing["category"]
                    is_custom_cat = category not in workout_categories
                    category_box.setCurrentText(category if not is_custom_cat else "Other")
                    category_other_input.setText(category if is_custom_cat else "")
                    update_subtypes()
                    exercise = existing["exercise"]
                    is_custom_ex = exercise not in workout_categories.get(category, [])
                    sub_box.setCurrentText(exercise if not is_custom_ex else "Other")
                    sub_other_input.setText(exercise if is_custom_ex else "")
                    calories_input.setText(existing.get("calories", ""))
                    h, m, s = map(int, existing.get("duration", "00:00:00").split(":"))
                    duration_input.setTime(QTime(h, m, s))
                    time_str = existing.get("time", "")
                    try:
                        h, m = map(int, time_str.split(":"))
                        time_input.setTime(QTime(h, m))
                    except: pass
                    location_box.setCurrentText(existing.get("location", ""))
                    location_other_input.setText(existing.get("location_other", ""))
                    priority_box.setCurrentText(existing.get("priority", "Medium"))
                    selected_equipment[:] = existing.get("equipment", "").split(", ")
                    equipment_display.setText(", ".join(selected_equipment))
                    notes_input.setPlainText(existing.get("notes", ""))

                def save_log():
                    updated = {
                        "date": date_input.date().toString("yyyy-MM-dd"),
                        "category": category_other_input.text() if category_box.currentText() == "Other" else category_box.currentText(),
                        "exercise": sub_other_input.text() if sub_box.currentText() == "Other" else sub_box.currentText(),
                        "calories": calories_input.text().strip(),
                        "duration": duration_input.time().toString("HH:mm:ss"),
                        "time": time_input.time().toString("HH:mm"),
                        "location": location_other_input.text() if location_box.currentText() == "Other" else location_box.currentText(),
                        "priority": priority_box.currentText(),
                        "equipment": ", ".join(selected_equipment),
                        "notes": notes_input.toPlainText().strip(),
                        "status": status_box.currentText() if existing else "Planned"
                    }

                    date_str = updated["date"]
                    me["workout_schedule"].setdefault(date_str, [])
                    if existing:
                        idx = me["workout_schedule"][date_str].index(existing)
                        me["workout_schedule"][date_str][idx] = updated
                    else:
                        me["workout_schedule"][date_str].append(updated)

                    try:
                        Path("user_data.json").write_text(json.dumps(users, indent=4))
                    except Exception as e:
                        print("write user_data.json:", e)

                    dlg.accept()
                    refresh_schedule()
                    calendar.workout_data = me["workout_schedule"]
                    calendar.draw_calendar()

                save_btn.clicked.connect(save_log)

                QTimer.singleShot(0, lambda: dlg.resize(520, dlg.sizeHint().height()))

                dlg.exec()




            def refresh_schedule():
                for i in reversed(range(scroll_layout.count())):
                    w = scroll_layout.itemAt(i).widget()
                    if w:
                        scroll_layout.removeWidget(w)
                        w.deleteLater()

                day = calendar.selected_date.toString("yyyy-MM-dd")
                entries = me["workout_schedule"].get(day, [])

                if not entries:
                    lbl = QLabel("No workouts scheduled.")
                    lbl.setStyleSheet("color:#888;font-size:11pt;")
                    scroll_layout.addWidget(lbl)
                else:
                    for e in entries:
                        def make_delete_cb(entry, current_day):
                            def callback():
                                me["workout_schedule"][current_day].remove(entry)
                                save_schedule()
                                refresh_schedule()
                                calendar.workout_data = me["workout_schedule"]
                                calendar.draw_calendar()
                            return callback

                        def make_edit_cb(entry):
                            return lambda: open_schedule_dialog("Edit Scheduled Workout", entry)

                        card = schedule_card(
                            e,
                            edit_cb=make_edit_cb(e),
                            delete_cb=make_delete_cb(e, day)
                        )
                        scroll_layout.addWidget(card)



            calendar.dateSelected.connect(lambda _: refresh_schedule())

            QTimer.singleShot(0, refresh_schedule)

            return page

        def create_nutrition_log_tab() -> QWidget:


            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            logs = me.get("nutrition_log", []) if me else []

            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(16)

            tabs = QTabWidget()
            tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: none;
                    background-color: transparent;
                }
                QTabBar::tab {
                    border: none;
                    background: #2e2e3e;
                    color: #ccc;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background: #222;
                    color: #fff;
                    border: none; 
                }
            """)
            def create_summary_tab() -> QWidget:

                summary_tab = QWidget()
                summary_tab.setStyleSheet("""
                    background-color: #222;
                    border-top-left-radius: 0px;
                    border-top-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border-bottom-right-radius: 12px;
                """)
                layout = QVBoxLayout(summary_tab)
                layout.setContentsMargins(10, 10, 10, 10)
                layout.setSpacing(16)


                inner_tabs = QTabWidget()
                inner_tabs.setStyleSheet("""
                    QTabWidget::pane {
                        background: #222;
                        border: none;
                    }

                    QTabBar::tab {
                        background: #222;
                        color: #ccc;
                        font-weight: 600;
                        padding: 10px 20px;
                        border: none;
                        border-bottom: 2px solid transparent;
                    }

                    QTabBar::tab:hover {
                        color: #fff;
                    }

                    QTabBar::tab:selected {
                        color: #fff;
                        border-bottom: 2px solid white;
                    }
                """)

                def create_calorie_chart_tab() -> QWidget:
                    matplotlib.use("QtAgg")
                    current_page = [0]
                    per_page = 10
                    total_pages = [1]

                    logs = me.get("nutrition_log", [])
                    calorie_targets = me.get("needed_calories", [])

                    if not logs:
                        tab = QWidget()
                        layout = QVBoxLayout(tab)
                        layout.addWidget(QLabel("No nutrition data found.", alignment=Qt.AlignmentFlag.AlignCenter))
                        return tab

                    date_list = []
                    for log in logs:
                        try:
                            d = datetime.strptime(log["date"], "%Y-%m-%d").date()
                            date_list.append(d)
                        except:
                            continue

                    if not date_list:
                        tab = QWidget()
                        layout = QVBoxLayout(tab)
                        layout.addWidget(QLabel("No valid dates found in nutrition log.", alignment=Qt.AlignmentFlag.AlignCenter))
                        return tab

                    min_date = min(date_list)
                    max_date = max(date_list)

                    tab = QWidget()
                    tab_layout = QVBoxLayout(tab)
                    tab_layout.setContentsMargins(16, 16, 16, 16)
                    tab_layout.setSpacing(10)

                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    combo_style = f"""
                        /* ----- Shared Combo & DateEdit Style ----- */
                        QComboBox, QDateEdit {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 28px 6px 12px;
                            border: 1px solid #555;
                            border-radius: 6px;
                            min-width: 50px;
                        }}
                        QComboBox::drop-down, QDateEdit::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QComboBox::down-arrow, QDateEdit::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}

                        /* ----- Combo Dropdown List ----- */
                        QComboBox QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            border: none;
                            outline: none;
                            padding: 4px;
                        }}

                        /* ----- DateEdit Calendar Popup ----- */
                        QDateEdit::calendar-widget {{
                            background-color: #333;
                            border: 1px solid #555;
                        }}
                        QCalendarWidget QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            selection-color: white;
                        }}
                        QCalendarWidget QWidget {{
                            alternate-background-color: #333;
                            background: #333;
                            color: #fff;
                        }}
                        QCalendarWidget QToolButton {{
                            background-color: #333;
                            color: #fff;
                            border: none;
                            font-weight: bold;
                        }}
                        QCalendarWidget QToolButton:hover {{
                            background-color: #444;
                        }}

                        /* ----- Labels & Buttons for Consistency ----- */
                        QLabel {{
                            color: #ccc;
                            font-weight: bold;
                            margin-right: 4px;
                        }}
                        QPushButton {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 15px;
                            border: 1px solid #555;
                            border-radius: 6px;
                        }}
                        QPushButton:hover {{
                            background-color: #444;
                        }}
                    """
                    control_bar = QHBoxLayout()
                    period_combo = QComboBox()
                    period_combo.addItems(["Daily", "Weekly", "Monthly"])
                    period_combo.setFixedWidth(140)
                    period_combo.setStyleSheet(combo_style)
                    control_bar.addWidget(period_combo)

                    control_bar.addStretch()
                    control_bar.addWidget(QLabel("From:"))

                    from_date = QDateEdit(QDate(min_date.year, min_date.month, min_date.day))
                    from_date.setCalendarPopup(True)
                    from_date.setDisplayFormat("yyyy-MM-dd")
                    from_date.setStyleSheet(combo_style)
                    control_bar.addWidget(from_date)

                    control_bar.addWidget(QLabel("To:"))
                    to_date = QDateEdit(QDate(max_date.year, max_date.month, max_date.day))
                    to_date.setCalendarPopup(True)
                    to_date.setDisplayFormat("yyyy-MM-dd")
                    to_date.setStyleSheet(combo_style)
                    control_bar.addWidget(to_date)

                    refresh_btn = QPushButton("Refresh")

                    refresh_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #5C6BC0;
                            color: white;
                            padding: 10px 24px;
                            border-radius: 8px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #3F51B5;
                        }
                    """)


                    refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    control_bar.addWidget(refresh_btn)

                    tab_layout.addLayout(control_bar)

                    chart_container = QWidget()
                    chart_layout = QVBoxLayout(chart_container)
                    chart_layout.setContentsMargins(0, 0, 0, 0)
                    tab_layout.addWidget(chart_container)

                    pagination = QHBoxLayout()
                    pagination.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                    prev_btn = QPushButton("←")
                    next_btn = QPushButton("→")

                    for btn in (prev_btn, next_btn):
                        btn.setFixedSize(32, 32)
                        btn.setCursor(Qt.CursorShape.PointingHandCursor)
                        btn.setStyleSheet("""
                            QPushButton {
                                background: #5C6BC0;
                                color: #fff;
                                border-radius: 6px;
                                font-weight: bold;
                            }
                            QPushButton:hover {
                                background: #3F51B5;
                            }
                        """)

                    page_input = QLineEdit("1")
                    page_input.setFixedWidth(40)
                    page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    page_input.setValidator(QIntValidator(1, 9999))
                    page_input.setStyleSheet("color:#fff;background:#222;border-radius:4px;padding:2px 4px;")

                    total_pages_lbl = QLabel("/ 1")
                    total_pages_lbl.setStyleSheet("color:#ccc;")

                    pagination.addWidget(prev_btn)
                    pagination.addWidget(page_input)
                    pagination.addWidget(total_pages_lbl)
                    pagination.addWidget(next_btn)

                    pagination_widget = QWidget()
                    pagination_widget.setLayout(pagination)
                    tab_layout.addWidget(pagination_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

                    def get_needed_calories_by_date():
                        entries = sorted(
                            (datetime.strptime(e["date"], "%Y-%m-%d").date(), e["needed_calories"])
                            for e in calorie_targets if "date" in e and "needed_calories" in e
                        )

                        def lookup(date):
                            for i in range(len(entries) - 1, -1, -1):
                                if date >= entries[i][0]:
                                    return entries[i][1]
                            return entries[0][1] if entries else None

                        return lookup

                    def group_logs_by_period(logs, needed_lookup, start, end, period="Daily"):
                        grouped_actual = defaultdict(float)
                        grouped_needed = defaultdict(float)
                        actual_by_date = defaultdict(float)

                        for log in logs:
                            try:
                                d = datetime.strptime(log.get("date", ""), "%Y-%m-%d").date()
                                cal = float(log.get("calories", 0))
                                actual_by_date[d] += cal
                            except:
                                continue

                        if period == "Daily":
                            current = start
                            while current <= end:
                                key = current.strftime("%Y-%m-%d")
                                needed = needed_lookup(current)
                                if needed is not None:
                                    grouped_actual[key] = actual_by_date.get(current, 0)
                                    grouped_needed[key] = needed
                                current += timedelta(days=1)
                            sorted_keys = sorted(grouped_actual.keys(), key=lambda k: datetime.strptime(k, "%Y-%m-%d"))

                        elif period == "Weekly":
                            current = start
                            while current <= end:
                                year, week, _ = current.isocalendar()
                                start_of_week = current - timedelta(days=current.weekday())
                                end_of_week = start_of_week + timedelta(days=6)
                                key = f"Week {week}\n({start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')})"
                                grouped_actual[key] += actual_by_date.get(current, 0)
                                grouped_needed[key] += needed_lookup(current) or 0
                                current += timedelta(days=1)
                            sorted_keys = list(grouped_actual.keys())

                        else:
                            current = start
                            while current <= end:
                                key = current.strftime("%B %Y")
                                grouped_actual[key] += actual_by_date.get(current, 0)
                                grouped_needed[key] += needed_lookup(current) or 0
                                current += timedelta(days=1)
                            sorted_keys = list(grouped_actual.keys())

                        return sorted_keys, grouped_actual, grouped_needed

                    def show_log_dialog(title, period_str, period, logs):
                        dialog = QDialog()
                        dialog.setWindowTitle(f"Nutrition Details: {period_str}")
                        dialog.setStyleSheet("background:#2e2e3e; color:white; font-size:13px;")
                        dialog.setMinimumWidth(400)
                        layout = QVBoxLayout(dialog)

                        period_logs = []
                        for log in logs:
                            try:
                                d = datetime.strptime(log.get("date", ""), "%Y-%m-%d").date()
                                if period == "Daily":
                                    if d.strftime("%Y-%m-%d") == title:
                                        period_logs.append(log)
                                elif period == "Weekly":
                                    week_num = int(re.search(r"Week (\d+)", title).group(1))
                                    year = int(re.search(r"\((\d{4})", title).group(1))
                                    if d.isocalendar()[1] == week_num and d.isocalendar()[0] == year:
                                        period_logs.append(log)
                                elif period == "Monthly":
                                    if d.strftime("%B %Y") == title:
                                        period_logs.append(log)
                            except:
                                continue

                        if not period_logs:
                            text = "No entries for this period."
                        else:
                            text = "\n".join(
                                f"{l['meal_type']} - {l['food_name']}: {l['calories']} kcal ({l.get('time', 'N/A')})"
                                for l in period_logs
                            )
                        editor = QTextEdit(text)
                        editor.setReadOnly(True)
                        layout.addWidget(editor)
                        dialog.setWindowModality(Qt.WindowModality.NonModal)
                        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
                        dialog.show()

                    def draw_chart():
                        start = from_date.date().toPyDate()
                        end = to_date.date().toPyDate()
                        period = period_combo.currentText()

                        filtered = [log for log in logs if start <= datetime.strptime(log.get("date", "0000-00-00"), "%Y-%m-%d").date() <= end]
                        if not filtered and period == "Daily":
                            return QLabel("No nutrition data to show.", alignment=Qt.AlignmentFlag.AlignCenter)

                        needed_lookup = get_needed_calories_by_date()
                        x_full, y_actual_full, y_needed_full = group_logs_by_period(filtered, needed_lookup, start, end, period)


                        page = current_page[0]
                        start_i = page * per_page
                        end_i = start_i + per_page
                        x = x_full[start_i:end_i]
                        y_actual = {k: y_actual_full[k] for k in x}
                        y_needed = {k: y_needed_full[k] for k in x}


                        total_pages[0] = (len(x_full) + per_page - 1) // per_page
                        page_input.setText(str(current_page[0] + 1))
                        total_pages_lbl.setText(f"/ {total_pages[0]}")


                        if not x:
                            return QLabel("No nutrition data to show.", alignment=Qt.AlignmentFlag.AlignCenter)

                        y_vals = [y_actual[k] for k in x]
                        y_targets = [y_needed[k] for k in x]

                        width = max(6, len(x) * 0.5)
                        dynamic_height = max(3, chart_container.height() / 100)
                        fig = Figure(figsize=(width, dynamic_height), dpi=100, facecolor="#222")

                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.25)
                        ax.set_facecolor("#1E1E2F")
                        ax.tick_params(colors='white', labelsize=8)
                        ax.set_title("Calorie Intake", color="white", fontsize=12, pad=15)
                        ax.spines[:].set_color("white")
                        ax.grid(True, linestyle='--', alpha=0.2)
                        ax.set_xticks(range(len(x)))
                        ax.set_xticklabels(x, rotation=35, ha='right', color="white", fontsize=8)

                        bar_colors = ["#00C853" if y_vals[i] >= y_targets[i] else "#D32F2F" for i in range(len(x))]
                        bars = ax.bar(range(len(x)), y_vals, color=bar_colors, width=0.6, label="Actual")

                        ax.plot(range(len(x)), y_targets, color="#FFCA28", linewidth=1.5, label="Needed")

                        for i in range(len(x)):
                            ax.text(i, y_vals[i] + 20, str(int(y_vals[i])), ha='center', va='bottom', color='white', fontsize=8)

                        ax.legend(loc='upper right', fontsize=9, facecolor="#2e2e3e", edgecolor="#fff", labelcolor="white")

                        canvas = FigureCanvas(fig)

                        def on_click(event):
                            if not event.inaxes:
                                return
                            for i, bar in enumerate(bars):
                                if bar.contains(event)[0]:
                                    show_log_dialog(x[i], x[i], period, filtered)

                        canvas.mpl_connect("button_press_event", on_click)
                        return canvas

                    def update_chart():
                        while chart_layout.count():
                            w = chart_layout.takeAt(0).widget()
                            if w:
                                w.setParent(None)
                                w.deleteLater()
                        chart_widget = draw_chart()
                        chart_layout.addWidget(chart_widget)

                    def paged_refresh():
                        page_input.setText(str(current_page[0] + 1))
                        prev_btn.setEnabled(current_page[0] > 0)
                        next_btn.setEnabled(current_page[0] < total_pages[0] - 1)
                        update_chart()
                    refresh_btn.clicked.connect(lambda: (current_page.__setitem__(0, 0), paged_refresh()))
                    period_combo.currentIndexChanged.connect(update_chart)
                    period_combo.currentIndexChanged.connect(lambda: (current_page.__setitem__(0, 0), paged_refresh()))
                    update_chart()

                    prev_btn.clicked.connect(lambda: (current_page.__setitem__(0, max(0, current_page[0] - 1)), paged_refresh()))
                    next_btn.clicked.connect(lambda: (current_page.__setitem__(0, min(total_pages[0] - 1, current_page[0] + 1)), paged_refresh()))

                    def handle_page_input():
                        try:
                            new_page = int(page_input.text()) - 1
                            if 0 <= new_page < total_pages[0]:
                                current_page[0] = new_page
                            else:
                                current_page[0] = total_pages[0] - 1
                        except ValueError:
                            current_page[0] = 0
                        paged_refresh()

                    page_input.returnPressed.connect(handle_page_input)

                    return tab


                def create_past_summary_tab() -> QWidget:
                    logs = me.get("nutrition_log", [])
                    calorie_targets = me.get("needed_calories", [])

                    tab = QWidget()
                    layout = QVBoxLayout(tab)
                    layout.setContentsMargins(10, 10, 10, 10)
                    layout.setSpacing(0)

                    today = datetime.today().date()


                    def lookup(date):
                        entries = sorted(
                            [(datetime.strptime(e["date"], "%Y-%m-%d").date(), e["needed_calories"]) for e in calorie_targets],
                            key=lambda x: x[0]
                        )
                        for i in range(len(entries) - 1, -1, -1):
                            if date >= entries[i][0]:
                                return entries[i][1]
                        return None













                    ranges = {
                        "Past Day": (today, today),
                        "Past 7 Days": (today - timedelta(days=6), today),
                        "Past 30 Days": (today - timedelta(days=29), today),
                        "Past 90 Days": (today - timedelta(days=90), today),
                        "Past 365 Days": (today - timedelta(days=364), today),
                    }
                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    combo_style = f"""
                            /* ----- Shared Combo & DateEdit Style ----- */
                            QComboBox, QDateEdit {{
                                background-color: #333;
                                color: #fff;
                                padding: 6px 12px 6px 12px;
                                border: 1px solid #555;
                                border-radius: 6px;
                                min-width: 50px;
                            }}
                            QComboBox::drop-down, QDateEdit::drop-down {{
                                subcontrol-origin: padding;
                                subcontrol-position: top right;
                                width: 24px;
                                border-left: 1px solid #555;
                            }}
                            QComboBox::down-arrow, QDateEdit::down-arrow {{
                                image: url("{down_arrow_path}");
                                width: 14px;
                                height: 14px;
                            }}

                            /* ----- Combo Dropdown List ----- */
                            QComboBox QAbstractItemView {{
                                background-color: #333;
                                color: #fff;
                                selection-background-color: #444;
                                border: none;
                                outline: none;
                                padding: 4px;
                            }}

                            /* ----- DateEdit Calendar Popup ----- */
                            QDateEdit::calendar-widget {{
                                background-color: #333;
                                border: 1px solid #555;
                            }}
                            QCalendarWidget QAbstractItemView {{
                                background-color: #333;
                                color: #fff;
                                selection-background-color: #444;
                                selection-color: white;
                            }}
                            QCalendarWidget QWidget {{
                                alternate-background-color: #333;
                                background: #333;
                                color: #fff;
                            }}
                            QCalendarWidget QToolButton {{
                                background-color: #333;
                                color: #fff;
                                border: none;
                                font-weight: bold;
                            }}
                            QCalendarWidget QToolButton:hover {{
                                background-color: #444;
                            }}

                            /* ----- Labels & Buttons for Consistency ----- */
                            QLabel {{
                                color: #ccc;
                                font-weight: bold;
                                margin-right: 4px;
                            }}
                            QPushButton {{
                                background-color: #333;
                                color: #fff;
                                padding: 6px 15px;
                                border: 1px solid #555;
                                border-radius: 6px;
                            }}
                            QPushButton:hover {{
                                background-color: #444;
                            }}
                        """

                    selector = QComboBox()
                    selector.setStyleSheet(combo_style)
                    selector.addItems(ranges.keys())
                    selector.setFixedWidth(140)

                    layout.addWidget(selector, alignment=Qt.AlignmentFlag.AlignLeft)


                    chart_container = QWidget()
                    chart_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    chart_area = QVBoxLayout(chart_container)
                    chart_area.setContentsMargins(0, 0, 0, 0)
                    chart_area.setSpacing(10)
                    layout.addWidget(chart_container)


                    info_label = QLabel("")
                    info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    info_label.setWordWrap(True)
                    info_label.setStyleSheet("""
                        QLabel {
                            color: #f0f0f0;
                            font-size: 15px;
                            padding-top: 0px;
                        }
                    """)
                    layout.addWidget(info_label, alignment=Qt.AlignmentFlag.AlignHCenter)

                    def show_summary(period):
                        start, end = ranges[period]
                        filtered = [l for l in logs if start <= datetime.strptime(l["date"], "%Y-%m-%d").date() <= end]

                        daily_actuals = defaultdict(float)
                        for log in filtered:
                            try:
                                d = datetime.strptime(log["date"], "%Y-%m-%d").date()
                                daily_actuals[d] += float(log.get("calories", 0))
                            except:
                                continue

                        total_actual = 0
                        total_needed = 0
                        met_goal_days = 0

                        for day, actual in daily_actuals.items():
                            needed = lookup(day)
                            if needed is not None:
                                total_actual += actual
                                total_needed += needed
                                if actual >= needed:
                                    met_goal_days += 1

                        avg_day = total_actual / len(daily_actuals) if daily_actuals else 0

                        by_meal = defaultdict(float)
                        food_counter = defaultdict(float)
                        for log in filtered:
                            by_meal[log.get("meal_type", "Other")] += float(log.get("calories", 0))
                            food_counter[log.get("food_name", "Unknown")] += float(log.get("calories", 0))

                        top_meal = max(by_meal, key=by_meal.get, default="N/A")
                        top_food = max(food_counter, key=food_counter.get, default="N/A")


                        while chart_area.count():
                            w = chart_area.takeAt(0).widget()
                            if w:
                                w.deleteLater()


                        fig = Figure(facecolor="#222", figsize=(6, 6), dpi=100)
                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.25)

                        ax.set_facecolor("#1E1E2F")
                        ax.spines[:].set_color("white")
                        ax.tick_params(colors="white", labelsize=9)
                        ax.set_title("Calorie Intake", color="white", fontsize=13, pad=12)


                        ax.set_xticks([0])
                        ax.set_xticklabels([period], color="white", fontsize=10)
                        ax.grid(True, linestyle="--", alpha=0.2)

                        color = "#00C853" if total_actual >= total_needed else "#D32F2F"
                        ax.bar([0], [total_actual], color=color, width=0.4)
                        ax.plot([0], [total_needed], color="#FFCA28", linewidth=2.2)

                        ax.text(0, total_actual + 20, f"{int(total_actual)} kcal", ha="center", color="white", fontsize=10)

                        canvas = FigureCanvas(fig)
                        canvas.setMinimumHeight(500)
                        chart_area.addWidget(canvas)



                        info_label.setText(
                            f"<b>Date Range:</b> {start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}<br>"
                            f"<b>Total Intake:</b> {int(total_actual)} kcal<br>"
                            f"<b>Needed:</b> {int(total_needed)} kcal<br>"
                            f"<b>Avg/day:</b> {int(avg_day)} kcal<br>"
                            f"<b>Top Meal:</b> {top_meal}<br>"
                            f"<b>Top Food:</b> {top_food}<br>"
                            f"<b>Days Met Goal:</b> {met_goal_days} / {len(daily_actuals)}"
                        )

                    selector.currentTextChanged.connect(show_summary)
                    show_summary(selector.currentText())

                    return tab




                calorie_chart_tab = create_calorie_chart_tab()
                past_summary_tab = create_past_summary_tab()


                inner_tabs.addTab(calorie_chart_tab, "Calorie Chart")
                inner_tabs.addTab(past_summary_tab, "Past Summary")

                layout.addWidget(inner_tabs)
                return summary_tab

            def create_fitness_calculators_tab():

                def create_bmi_calculator_tab(user_gender: str = "Male") -> QWidget:
                    tab = QWidget()

                    main_layout = QHBoxLayout(tab)
                    main_layout.setContentsMargins(10, 10, 10, 10)
                    main_layout.setSpacing(40)
                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    combo_style = f"""
                        /* ----- Shared Input Style ----- */
                        QComboBox, QDateEdit, QLineEdit {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 12px;
                            border: 1px solid #555;
                            border-radius: 6px;
                            min-width: 50px;
                        }}
                        QComboBox::drop-down, QDateEdit::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QComboBox::down-arrow, QDateEdit::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}

                        /* ----- Combo Dropdown List ----- */
                        QComboBox QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            border: none;
                            outline: none;
                            padding: 4px;
                        }}

                        /* ----- DateEdit Calendar Popup ----- */
                        QDateEdit::calendar-widget {{
                            background-color: #333;
                            border: 1px solid #555;
                        }}
                        QCalendarWidget QAbstractItemView {{
                            background-color: #333;
                            color: #fff;
                            selection-background-color: #444;
                            selection-color: white;
                        }}
                        QCalendarWidget QWidget {{
                            alternate-background-color: #333;
                            background: #333;
                            color: #fff;
                        }}
                        QCalendarWidget QToolButton {{
                            background-color: #333;
                            color: #fff;
                            border: none;
                            font-weight: bold;
                        }}
                        QCalendarWidget QToolButton:hover {{
                            background-color: #444;
                        }}

                        /* ----- Labels & Buttons for Consistency ----- */
                        QLabel {{
                            color: #ccc;
                            font-weight: bold;
                            margin-right: 4px;
                        }}
                        QPushButton {{
                            background-color: #333;
                            color: #fff;
                            padding: 6px 15px;
                            border: 1px solid #555;
                            border-radius: 6px;
                        }}
                        QPushButton:hover {{
                            background-color: #444;
                        }}
                    """

                    weight_input = QLineEdit()
                    weight_input.setPlaceholderText("Weight (kg)")
                    weight_input.setFixedWidth(120)

                    height_input = QLineEdit()
                    height_input.setPlaceholderText("Height (cm)")
                    height_input.setFixedWidth(120)

                    gender_input = QComboBox()
                    gender_input.addItems(["Male", "Female"])
                    gender_input.setCurrentText(user_gender if user_gender in ["Male", "Female"] else "Male")
                    gender_input.setFixedWidth(120)


                    def create_vertical_field(label_text: str, input_widget: QWidget) -> QWidget:
                        wrapper = QWidget()
                        layout = QVBoxLayout(wrapper)
                        layout.setSpacing(2)
                        layout.setContentsMargins(0, 0, 0, 0)

                        label = QLabel(label_text)
                        label.setStyleSheet("font-size: 14px; color: white;")
                        input_widget.setFixedSize(120, 30)
                        input_widget.setStyleSheet(combo_style)

                        layout.addWidget(label)
                        layout.addWidget(input_widget)
                        return wrapper


                    form_container = QWidget()
                    form_layout = QVBoxLayout(form_container)
                    form_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
                    form_layout.setSpacing(10)
                    form_layout.setContentsMargins(0, 0, 0, 0)

                    form_layout.addWidget(create_vertical_field("Weight:", weight_input))
                    form_layout.addWidget(create_vertical_field("Height:", height_input))
                    form_layout.addWidget(create_vertical_field("Gender:", gender_input))

                    main_layout.addWidget(form_container, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)


                    content_container = QWidget()
                    content_layout = QVBoxLayout(content_container)
                    content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    content_layout.setSpacing(20)

                    image = QLabel()
                    image.setAlignment(Qt.AlignmentFlag.AlignCenter)

                    bmi_label = QLabel("")
                    bmi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    bmi_label.setFont(QFont(sf_family, 16, weight=600))

                    classification_label = QLabel("")
                    classification_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    classification_label.setFont(QFont(sf_family, 16, weight=600))

                    calculate_btn = QPushButton("Calculate BMI")
                    calculate_btn.setFixedSize(160, 44)
                    calculate_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #5C6BC0;
                            color: white;
                            padding: 10px 24px;
                            border-radius: 8px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #3F51B5;
                        }
                    """)

                    calculate_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                    content_layout.addWidget(image)
                    content_layout.addWidget(bmi_label)
                    content_layout.addWidget(classification_label)
                    btn_container = QWidget()
                    btn_layout = QHBoxLayout(btn_container)
                    btn_layout.setContentsMargins(0, 0, 0, 0)
                    btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    btn_layout.addWidget(calculate_btn)
                    content_layout.addWidget(btn_container)


                    main_layout.addWidget(content_container, stretch=1)


                    def update_image():
                        selected_gender = gender_input.currentText().lower()
                        path = resource_path(f"assets/{selected_gender}bmi.png")
                        if not os.path.exists(path):
                            image.clear()
                            return
                        pixmap = QPixmap(path).scaledToWidth(600, Qt.TransformationMode.SmoothTransformation)
                        image.setPixmap(pixmap)


                    def calculate_bmi():
                        try:
                            weight = float(weight_input.text())
                            height = float(height_input.text()) / 100
                            if height <= 0:
                                raise ValueError()
                            bmi = round(weight / (height ** 2), 1)
                        except:
                            bmi_label.setText("Invalid input.")
                            bmi_label.setStyleSheet("color: orange;")
                            classification_label.setText("")
                            return

                        bmi_label.setText(f"Your BMI: {bmi}")
                        bmi_label.setStyleSheet("color: white;")


                        if bmi < 18.5:
                            category = "Underweight"
                            color = "cyan"
                        elif bmi < 25:
                            category = "Normal"
                            color = "green"
                        elif bmi < 30:
                            category = "Overweight"
                            color = "yellow"
                        elif bmi < 35:
                            category = "Obese"
                            color = "orange"
                        else:
                            category = "Extremely Obese"
                            color = "red"

                        classification_label.setText(f"Classification: {category}")
                        classification_label.setStyleSheet(f"color: {color};")

                        update_image()


                    gender_input.currentIndexChanged.connect(update_image)
                    calculate_btn.clicked.connect(calculate_bmi)

                    update_image()
                    return tab

                def create_calorie_calculator_tab() -> QWidget:

                    def create_result_badge(label, calories, note="", percent="", is_bmr=False, group=None):
                        badge = QFrame()
                        badge.setStyleSheet("""
                            QFrame {
                                border: 1px solid #444;
                                border-radius: 6px;
                                background-color: #2d2d2d;
                            }
                        """)
                        badge.setFixedHeight(80)
                        badge_layout = QHBoxLayout(badge)
                        badge_layout.setContentsMargins(6, 0, 0, 0)
                        badge_layout.setSpacing(6)

                        radio = None
                        if not is_bmr:
                            radio = QRadioButton()
                            radio.setStyleSheet("""
                                QRadioButton { spacing: 6px; }
                                QRadioButton::indicator {
                                    width: 18px; height: 18px;
                                    border-radius: 9px;
                                    border: 2px solid #aaa;
                                    background: transparent;
                                }
                                QRadioButton::indicator:checked {
                                    background-color: #78b56d;
                                    border: 2px solid white;
                                }
                            """)
                            if group:
                                group.addButton(radio)
                            badge_layout.addWidget(radio)


                        left = QFrame()
                        left.setStyleSheet("border: none; background: transparent;")
                        left_layout = QVBoxLayout(left)
                        left_layout.setContentsMargins(0, 0, 0, 0)
                        left_layout.setSpacing(2)

                        label_lbl = QLabel(label)
                        label_lbl.setStyleSheet(
                            "font-size:15px; font-weight:600; color:white; background:transparent; border:none;"
                        )
                        left_layout.addWidget(label_lbl)
                        if note:
                            note_lbl = QLabel(note)
                            note_lbl.setStyleSheet(
                                "font-size:12px; color:#d3d3d3; background:transparent; border:none;"
                            )
                            left_layout.addWidget(note_lbl)


                        right = TriangleBadge()

                        right_layout = QVBoxLayout(right)
                        right_layout.setContentsMargins(32, 4, 12, 4)
                        right_layout.setSpacing(0)

                        cal_lbl = QLabel(f"{calories:,}")
                        cal_lbl.setStyleSheet(
                            "font-size:18px; font-weight:bold; color:white; background:transparent; border:none;"
                        )
                        right_layout.addWidget(cal_lbl)

                        suffix_lbl = QLabel("Calories/day")
                        suffix_lbl.setStyleSheet(
                            "font-size:12px; color:#78b56d; background:transparent; border:none;"
                        )
                        right_layout.addWidget(suffix_lbl)

                        if percent:
                            percent_lbl = QLabel(percent)
                            percent_lbl.setStyleSheet(
                                "font-size:12px; color:#d3d3d3; background:transparent; border:none;"
                            )
                            right_layout.addWidget(percent_lbl)

                        badge_layout.addWidget(left, stretch=1)
                        badge_layout.addWidget(right)

                        return badge, radio

                    def calculate_bmr(gender, weight, height, age):
                        if gender == "Male":
                            return 10 * weight + 6.25 * height - 5 * age + 5
                        else:
                            return 10 * weight + 6.25 * height - 5 * age - 161

                    tab = QWidget()
                    main_layout = QHBoxLayout(tab)
                    main_layout.setContentsMargins(16, 16, 16, 16)
                    main_layout.setSpacing(12)


                    result_box = QFrame()
                    result_box.setStyleSheet("background-color: rgb(66, 66, 66); border-radius: 12px;")
                    result_box.setFixedWidth(400)

                    result_layout = QVBoxLayout(result_box)
                    result_title = QLabel("Choose Plan:")
                    result_title.setStyleSheet("font-size:16px; font-weight:bold; color:white;")
                    result_title.setVisible(False)
                    result_layout.addWidget(result_title)

                    result_container = QWidget()
                    result_container.setStyleSheet("background: transparent;")
                    result_vbox = QVBoxLayout(result_container)
                    result_vbox.setContentsMargins(0, 0, 0, 0)
                    result_vbox.setSpacing(10)
                    result_layout.addWidget(result_container)

                    save_button = QPushButton("Save Selected Plan")
                    save_button.setVisible(False)
                    save_button.setCursor(Qt.CursorShape.PointingHandCursor)
                    save_button.setStyleSheet("""
                        QPushButton {
                            background-color: #78b56d;
                            color: white;
                            padding: 10px 20px;
                            font-size: 16px;
                            font-weight: bold;
                            border-radius: 8px;
                        }
                        QPushButton:hover {
                            background-color: #98cf89;
                        }
                    """)
                    result_layout.addWidget(save_button, alignment=Qt.AlignmentFlag.AlignCenter)
                    result_layout.addStretch()


                    form_container = QWidget()
                    form_layout = QVBoxLayout(form_container)
                    form_layout.setContentsMargins(0, 0, 0, 0)
                    form_layout.setSpacing(0)


                    switcher = QHBoxLayout()
                    switcher.setSpacing(0)

                    metric_btn = QPushButton("Metric")
                    imperial_btn = QPushButton("Imperial")

                    for btn in (metric_btn, imperial_btn):
                        btn.setCheckable(True)
                        btn.setMinimumHeight(42)
                        btn.setMinimumWidth(140)
                        btn.setStyleSheet("""
                            QPushButton {
                                background-color: #111111;
                                color: white;
                                padding: 8px 16px;
                                font-size: 16px;
                                font-weight: bold;
                                border-top-left-radius: 10px;
                                border-top-right-radius: 10px;
                                border-bottom-left-radius: 0px;
                                border-bottom-right-radius: 0px;
                                background: transparent;
                                border: none;
                            }
                            QPushButton:checked {
                                background-color: rgb(66, 66, 66);
                            }
                        """)
                        switcher.addWidget(btn)
                    switcher.addStretch()
                    form_layout.addLayout(switcher)

                    INPUT_STYLE = """
                        QLineEdit, QComboBox, QDateEdit {
                            background-color: rgb(48, 48, 48);
                            color: white;
                            font-size: 15px;
                            padding: 4px 10px;
                            border: 1px solid white;
                            border-radius: 5px;
                            min-height: 30px;
                            min-width: 200px;
                        }

                        QComboBox::drop-down, QDateEdit::drop-down {
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid white;
                        }

                        QComboBox::down-arrow, QDateEdit::down-arrow {
                            image: none;
                            width: 0;
                            height: 0;
                        }
                    """


                    box = QFrame()
                    box.setStyleSheet("""
                        background-color: rgb(66, 66, 66);
                        border-top-left-radius: 0px;
                        border-top-right-radius: 12px;
                        border-bottom-left-radius: 12px;
                        border-bottom-right-radius: 12px;

                        border: none;
                    """)
                    box_layout = QVBoxLayout(box)
                    box_layout.setContentsMargins(16, 16, 16, 16)
                    box_layout.setSpacing(10)

                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll.setMaximumHeight(560)
                    scroll.setStyleSheet("""
                        QScrollArea { border: none; }
                        QScrollBar:vertical {
                            border: none;
                            background: transparent;
                            width: 10px;
                            margin: 4px 0;
                        }
                        QScrollBar::handle:vertical {
                            background-color: #888;
                            min-height: 30px;
                            border-radius: 5px;
                        }
                        QScrollBar::handle:vertical:hover {
                            background-color: #aaa;
                        }
                        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                            height: 0px;
                        }
                        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                            background: none;
                        }
                    """)
                    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                    scroll.setWidget(box)
                    form_layout.addWidget(scroll)

                    stacked = QStackedWidget()

                    def create_unit_widget(is_metric=True):
                        w = QWidget()
                        layout = QVBoxLayout(w)
                        layout.setSpacing(10)

                        label_style = "color:white; font-size:15px; font-weight:600;"

                        def styled_lineedit(placeholder, fixed_width=200):
                            field = QLineEdit()
                            field.setPlaceholderText(placeholder)
                            field.setStyleSheet(INPUT_STYLE)
                            field.setMinimumHeight(30)
                            field.setFixedWidth(fixed_width)
                            return field


                        age = styled_lineedit("Years")
                        layout.addWidget(QLabel("Age:", styleSheet=label_style))
                        layout.addWidget(age)


                        layout.addWidget(QLabel("Height:", styleSheet=label_style))
                        height_layout = QHBoxLayout()
                        height_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                        height_layout.setSpacing(8)

                        if is_metric:
                            height_cm = styled_lineedit("cm")
                            height_layout.addWidget(height_cm)
                            height_ft = height_in = None
                        else:
                            height_ft = styled_lineedit("ft", 95)
                            height_in = styled_lineedit("in", 95)
                            height_layout.addWidget(height_ft)
                            height_layout.addWidget(height_in)
                            height_cm = None

                        layout.addLayout(height_layout)


                        weight = styled_lineedit("kg" if is_metric else "lb")
                        layout.addWidget(QLabel("Weight:", styleSheet=label_style))
                        layout.addWidget(weight)

                        return {
                            "widget": w,
                            "age": age,
                            "height_cm": height_cm,
                            "height_ft": height_ft,
                            "height_in": height_in,
                            "weight": weight
                        }

                    metric = create_unit_widget(True)
                    imperial = create_unit_widget(False)
                    stacked.addWidget(metric["widget"])
                    stacked.addWidget(imperial["widget"])
                    box_layout.addWidget(stacked)


                    bottom_fields = QWidget()
                    bottom_layout = QVBoxLayout(bottom_fields)
                    bottom_layout.setContentsMargins(8, 0, 0, 0)
                    bottom_layout.setSpacing(10)


                    gender_label = QLabel("Gender:")
                    gender_label.setStyleSheet("color:white; font-size:16px; font-weight:600;")


                    gbox = QHBoxLayout()
                    gbox.setSpacing(16)
                    gbox.addWidget(gender_label)
                    male = QRadioButton("Male")
                    female = QRadioButton("Female")
                    male.setChecked(True)
                    for r in (male, female):
                        r.setStyleSheet("""
                            QRadioButton {
                                color: white;
                                font-size: 14px;
                                padding: 2px 6px;
                            }
                            QRadioButton::indicator {
                                width: 16px;
                                height: 16px;
                                border-radius: 8px;
                                border: 2px solid white;
                                background: rgb(48,48,48);
                            }
                            QRadioButton::indicator:checked {
                                background-color: white;
                            }
                        """)
                    gender_group = QButtonGroup()
                    gender_group.addButton(male)
                    gender_group.addButton(female)
                    gbox.addWidget(male)
                    gbox.addWidget(female)
                    bottom_layout.addLayout(gbox)


                    activity_label = QLabel("Activity:")
                    activity_label.setStyleSheet("color:white; font-size:16px; font-weight:600;")
                    activity = QComboBox()
                    activity.addItems([
                        "Basal Metabolic Rate (BMR)",
                        "Sedentary: little or no exercise",
                        "Light: 1–3x/week",
                        "Moderate: 4–5x/week",
                        "Active: daily or 3–4 intense/week",
                        "Very Active: intense 6–7x/week",
                        "Extra Active: intense daily or job"
                    ])
                    activity.setStyleSheet(INPUT_STYLE)
                    bottom_layout.addWidget(activity_label)
                    bottom_layout.addWidget(activity)


                    date_label = QLabel("Date:")
                    date_label.setStyleSheet("color:white; font-size:16px; font-weight:600;")
                    date_input = QDateEdit(QDate.currentDate())
                    date_input.setCalendarPopup(True)
                    date_input.setDisplayFormat("yyyy-MM-dd")
                    date_input.setStyleSheet(INPUT_STYLE)
                    bottom_layout.addWidget(date_label)
                    bottom_layout.addWidget(date_input)


                    calc = QPushButton("Calculate ▶")
                    calc.setCursor(Qt.CursorShape.PointingHandCursor)
                    calc.setStyleSheet("""
                        QPushButton {
                            background-color: rgb(48, 48, 48);
                            color: white;
                            padding: 10px 20px;
                            font-size: 16px;
                            font-weight: bold;
                            border-radius: 8px;
                        }
                        QPushButton:hover {
                            background-color: rgb(60, 60, 60);
                        }
                    """)
                    calc.setFixedWidth(180)
                    bottom_layout.addWidget(calc, alignment=Qt.AlignmentFlag.AlignCenter)


                    box_layout.addWidget(bottom_fields)



                    button_group = QButtonGroup()
                    selected_plan_calories = {"calories": None}

                    def calculate():
                        is_metric = stacked.currentIndex() == 0

                        for i in reversed(range(result_vbox.count())):
                            widget = result_vbox.itemAt(i).widget()
                            if widget:
                                widget.setParent(None)

                        try:

                            if is_metric:
                                age = int(metric["age"].text())
                                weight = float(metric["weight"].text())
                                height = float(metric["height_cm"].text())
                            else:
                                age = int(imperial["age"].text())
                                weight = float(imperial["weight"].text()) * 0.453592
                                ft = float(imperial["height_ft"].text())
                                inch = float(imperial["height_in"].text())
                                height = (ft * 12 + inch) * 2.54

                            gender = "Male" if male.isChecked() else "Female"
                            bmr = calculate_bmr(gender, weight, height, age)
                            act = activity.currentIndex()
                            multipliers = [1.0, 1.2, 1.375, 1.55, 1.725, 1.9, 2.0]
                            cal = round(bmr * multipliers[act])


                            result_title.setVisible(True)
                            save_button.setVisible(True)

                            if act == 0:
                                badge, _ = create_result_badge(
                                    "Basal Metabolic Rate (BMR)", int(bmr), is_bmr=True
                                )
                                save_button.setText("Save Plan")
                                badge.setProperty("calories", int(bmr))
                                result_vbox.addWidget(badge)
                            else:
                                values = [
                                    ("Maintain weight", cal, "", "100%"),
                                    ("Mild weight loss", max(cal-250, 0), "0.25 kg/week", f"{int((cal-250)/cal*100)}%"),
                                    ("Weight loss", max(cal-500, 0), "0.5 kg/week", f"{int((cal-500)/cal*100)}%"),
                                    ("Extreme weight loss", max(cal-1000, 0), "1 kg/week", f"{int((cal-1000)/cal*100)}%"),
                                ]
                                for label, kcal, note, pct in values:
                                    badge, radio = create_result_badge(label, kcal, note, pct, group=button_group)
                                    badge.setProperty("calories", kcal)
                                    result_vbox.addWidget(badge)
                                save_button.setText("Save Selected Plan")

                        except Exception:
                            error = QLabel("⚠ Please enter all fields correctly.")
                            error.setStyleSheet("color: orange; font-size: 14px; font-weight: bold;")
                            error.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            result_title.setVisible(False)
                            save_button.setVisible(False)
                            result_vbox.addWidget(error)


                    def save_selected():
                        kcal = None
                        for i in range(result_vbox.count()):
                            item = result_vbox.itemAt(i).widget()
                            r = item.findChild(QRadioButton)
                            if r:
                                if r.isChecked():
                                    kcal = item.property("calories")
                                    break
                            else:
                                kcal = item.property("calories")
                                break

                        if kcal is not None:
                            entry = {"date": date_input.date().toString("yyyy-MM-dd"), "needed_calories": kcal}
                            me["needed_calories"] = [e for e in me.get("needed_calories", []) if e["date"] != entry["date"]]
                            me["needed_calories"].append(entry)
                            Path("user_data.json").write_text(json.dumps(users, indent=4))
                            QMessageBox.information(tab, "Saved", f"Saved {kcal} kcal for {entry['date']}." )
                        else:
                            QMessageBox.warning(tab, "No Selection", "Please select a plan before saving.")



                    save_button.clicked.connect(save_selected)
                    calc.clicked.connect(calculate)
                    def update_box_style(is_metric):
                        if is_metric:
                            box.setStyleSheet("""
                                background-color: rgb(66, 66, 66);
                                border-top-left-radius: 0px;
                                border-top-right-radius: 12px;
                                border-bottom-left-radius: 12px;
                                border-bottom-right-radius: 12px;
                                border: none;
                            """)
                        else:
                            box.setStyleSheet("""
                                background-color: rgb(66, 66, 66);
                                border-radius: 12px;
                                border: none;
                            """)

                    metric_btn.clicked.connect(lambda: (
                        metric_btn.setChecked(True),
                        imperial_btn.setChecked(False),
                        stacked.setCurrentIndex(0),
                        update_box_style(True)
                    ))
                    imperial_btn.clicked.connect(lambda: (
                        imperial_btn.setChecked(True),
                        metric_btn.setChecked(False),
                        stacked.setCurrentIndex(1),
                        update_box_style(False)
                    ))

                    metric_btn.click()

                    main_layout.addWidget(form_container, stretch=1)
                    main_layout.addWidget(result_box)

                    return tab

                user_gender = me.get("gender", "Male") if me else "Male"
                tab = QWidget()
                tab.setStyleSheet("""
                    background-color: #222;
                    border-radius:12px;
                """)
                layout = QVBoxLayout(tab)
                layout.setContentsMargins(10, 10, 0, 0)

                tabs = QTabWidget()


                tabbar = tabs.tabBar()
                tabbar.setStyleSheet("""
                    QTabWidget::pane {
                        background: #222;
                        border: none;
                    }

                    QTabBar::tab {
                        background: #222;
                        color: #ccc;
                        font-weight: 600;
                        padding: 10px 20px;
                        border: none;
                        border-bottom: 2px solid transparent;
                    }

                    QTabBar::tab:hover {
                        color: #fff;
                    }

                    QTabBar::tab:selected {
                        color: #fff;
                        border-bottom: 2px solid white;
                    }
                """)


                tabs.addTab(create_calorie_calculator_tab(), "Calorie Calculator")
                tabs.addTab(create_bmi_calculator_tab(user_gender), "BMI")

                layout.addWidget(tabs)
                return tab

            def create_food_guide_tab() -> QWidget:
                tab = QWidget()
                layout = QHBoxLayout(tab)
                layout.setSpacing(0)
                layout.setContentsMargins(0, 0, 0, 0)


                sidebar = QListWidget()
                sidebar.setFixedWidth(280)
                sidebar.setContentsMargins(10, 10, 10, 10)
                sidebar.setStyleSheet("""
                    QListWidget {
                        background-color: #222;
                        color: white;
                        border: none;
                        font-size: 15px;
                        border-top-left-radius: 14px;
                        border-bottom-left-radius: 14px;
                    }
                    QListWidget::item {
                        margin: 8px;
                        padding: 12px;
                        border-radius: 12px;
                        background-color: #2a2a2a;
                    }
                    QListWidget::item:selected {
                        background-color: #023e7a;
                        color: white;
                    }
                """)

                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.VLine)
                divider.setLineWidth(1)
                divider.setStyleSheet("background-color: white; width: 1px;")

                sections = [
                    "Calories in Common Foods",
                    "Sample Meal Plans",
                    "Calories Burned from Exercises",
                    "Energy from Food Components"
                ]
                for section in sections:
                    QListWidgetItem(section, sidebar)


                content_stack = QStackedWidget()
                content_stack.setStyleSheet("background-color: #121212; color: #ddd;")

                def load_json(filename):
                    try:
                        with open(resource_path(f"assets/calorieinfo/{filename}"), "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception as e:
                        return {"error": str(e)}

                def create_table_section(title, headers, rows):
                    box = QWidget()
                    box_layout = QVBoxLayout(box)
                    box_layout.setContentsMargins(32, 32, 32, 32)
                    box_layout.setSpacing(24)
                    box_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

                    header_label = QLabel(title)
                    header_label.setFont(QFont(sf_family, 22, QFont.Weight.Bold))
                    header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                    header_label.setStyleSheet("color: white;")
                    box_layout.addWidget(header_label)

                    table = QWidget()
                    table_layout = QVBoxLayout(table)
                    table_layout.setSpacing(0)
                    table_layout.setContentsMargins(0, 0, 0, 0)
                    table_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

                    def make_row(data, is_header=False, alt=False):
                        row = QWidget()
                        row_layout = QHBoxLayout(row)
                        row_layout.setContentsMargins(0, 0, 0, 0)
                        row_layout.setSpacing(0)

                        bg_color = "#333333" if alt else "#1f1f1f"
                        if is_header:
                            bg_color = "#023e7a"
                        row.setStyleSheet(f"background-color: {bg_color}; border: none;")

                        for i, cell in enumerate(data):
                            wrapper = QWidget()
                            wrapper_layout = QVBoxLayout(wrapper)
                            wrapper_layout.setContentsMargins(16, 12, 16, 12)
                            wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

                            label = QLabel(str(cell))
                            label.setWordWrap(True)
                            label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

                            if not is_header and cell and all(str(x).strip() == "" for x in data[1:]):

                                label.setStyleSheet("color: white; font-weight: bold; text-decoration: underline;")
                                label.setFont(QFont(sf_family, 16, QFont.Weight.Bold))
                            else:
                                label.setStyleSheet("color: white;" if is_header else "color: #ccc;")
                                label.setFont(QFont(sf_family, 15, QFont.Weight.Bold if is_header else QFont.Weight.Normal))

                            wrapper_layout.addWidget(label)
                            wrapper.setFixedWidth(230)
                            row_layout.addWidget(wrapper)

                            if i < len(data) - 1:
                                line = QFrame()
                                line.setFrameShape(QFrame.Shape.VLine)
                                line.setLineWidth(1)
                                line.setStyleSheet("background-color: rgba(255, 255, 255, 80);")
                                row_layout.addWidget(line)


                        return row


                    table_layout.addWidget(make_row(headers, is_header=True))
                    for i, row_data in enumerate(rows):
                        table_layout.addWidget(make_row(row_data, alt=(i % 2 == 0)))

                    box_layout.addWidget(table)

                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll.setWidget(box)
                    scroll.setStyleSheet("""
                        QScrollBar:vertical {
                            border: none;
                            background: #2e2e2e;
                            width: 8px;
                            margin: 0px;
                            border-radius: 4px;
                        }
                        QScrollBar::handle:vertical {
                            background: #555;
                            min-height: 20px;
                            border-radius: 4px;
                        }
                        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                            height: 0px;
                        }

                        QScrollBar:horizontal {
                            border: none;
                            background: #2e2e2e;
                            height: 8px;
                            margin: 0px;
                            border-radius: 4px;
                        }
                        QScrollBar::handle:horizontal {
                            background: #555;
                            min-width: 20px;
                            border-radius: 4px;
                        }
                        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                            width: 0px;
                        }
                    """)

                    return scroll


                food_data = load_json("common_foods.json")
                if "categories" in food_data:
                    food_rows = []
                    for category in food_data["categories"]:
                        food_rows.append([category["category"], "", "", ""])
                        for item in category["items"]:
                            food_rows.append([
                                item["food"], item["serving"], item["calories"], item["kJ"]
                            ])
                    content_stack.addWidget(create_table_section(
                        "Calories in Common Foods",
                        ["Food", "Serving Size", "Calories", "kJ"],
                        food_rows
                    ))
                else:
                    content_stack.addWidget(QLabel("Error loading common_foods.json"))


                plans_data = load_json("meal_plans.json")
                if "meal_plans" in plans_data:
                    plans = plans_data["meal_plans"]
                    meal_rows = []
                    for meal in ["Breakfast", "Snack", "Lunch", "Snack 2", "Dinner"]:
                        meal_rows.append([
                            meal,
                            "\n".join(plans["1200"].get(meal, [])),
                            "\n".join(plans["1500"].get(meal, [])),
                            "\n".join(plans["2000"].get(meal, []))
                        ])
                    meal_rows.append([
                        "Total",
                        f"{plans['1200']['Total']} Calories",
                        f"{plans['1500']['Total']} Calories",
                        f"{plans['2000']['Total']} Calories"
                    ])
                    content_stack.addWidget(create_table_section(
                        "Sample Meal Plans",
                        ["Meal", "1200 Cal Plan", "1500 Cal Plan", "2000 Cal Plan"],
                        meal_rows
                    ))
                else:
                    content_stack.addWidget(QLabel("Error loading meal_plans.json"))


                burn_data = load_json("exercise_burn.json")
                if "exercises" in burn_data:
                    burn_rows = [
                        [
                            item["activity"],
                            item["calories_burned"]["125_lb"],
                            item["calories_burned"]["155_lb"],
                            item["calories_burned"]["185_lb"]
                        ] for item in burn_data["exercises"]
                    ]
                    content_stack.addWidget(create_table_section(
                        "Calories Burned from Common Exercises",
                        ["Activity (1 hour)", "125 lb", "155 lb", "185 lb"],
                        burn_rows
                    ))
                else:
                    content_stack.addWidget(QLabel("Error loading exercise_burn.json"))


                comp_data = load_json("energy_components.json")
                if "components" in comp_data:
                    comp_rows = [
                        [
                            item["name"],
                            item["kJ_per_gram"],
                            item["kcal_per_gram"],
                            item["kJ_per_ounce"],
                            item["kcal_per_ounce"]
                        ] for item in comp_data["components"]
                    ]
                    content_stack.addWidget(create_table_section(
                        "Energy from Common Food Components",
                        ["Component", "kJ/g", "kcal/g", "kJ/oz", "kcal/oz"],
                        comp_rows
                    ))
                else:
                    content_stack.addWidget(QLabel("Error loading energy_components.json"))


                sidebar.currentRowChanged.connect(content_stack.setCurrentIndex)
                sidebar.setCurrentRow(0)

                layout.addWidget(sidebar)
                layout.addWidget(divider)
                layout.addWidget(content_stack)
                return tab

            def create_log_nutrition_tab() -> QWidget:
                last_filtered_logs = [None]
                MAX_PER_PAGE = 4

                try:
                    users = json.loads(Path("user_data.json").read_text())
                except Exception as e:
                    print("user_data.json:", e)
                    users = []

                me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
                me.setdefault("nutrition_log", [])

                page = QWidget()
                page.setStyleSheet("""
                    background-color: #222;
                    border-radius:12px;
                """)
                layout = QVBoxLayout(page)
                layout.setContentsMargins(20, 20, 20, 20)
                layout.setSpacing(12)
                layout.setAlignment(Qt.AlignmentFlag.AlignTop)


                title_bar = QWidget()
                title_layout = QHBoxLayout(title_bar)
                title_layout.setContentsMargins(0, 0, 0, 0)
                title_layout.setSpacing(4)

                title = QLabel("Nutrition Log")
                title.setFont(QFont(sf_family, 20, QFont.Weight.Bold))
                title.setStyleSheet("color:#fff;")

                filter_btn = QPushButton()
                filter_btn.setIcon(QIcon(resource_path("assets/filter.png")))
                filter_btn.setIconSize(QSize(40, 40))
                filter_btn.setFixedSize(48, 48)
                filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                filter_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,255,255,0.1);
                        border-radius:6px;
                    }
                """)

                clear_filter_btn = QPushButton()
                clear_filter_btn.setIcon(QIcon(resource_path("assets/clear.png")))
                clear_filter_btn.setIconSize(QSize(16, 16))
                clear_filter_btn.setFixedSize(20, 20)
                clear_filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                clear_filter_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,255,255,0.1);
                        border-radius:10px;
                    }
                """)
                clear_filter_btn.setVisible(False)

                filter_applied = [False]

                filter_wrap = QWidget()
                filter_layout = QGridLayout(filter_wrap)
                filter_layout.setContentsMargins(0, 0, 0, 0)
                filter_layout.setSpacing(0)
                filter_layout.addWidget(filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
                filter_layout.addWidget(clear_filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

                title_layout.addWidget(title)
                title_layout.addStretch()
                title_layout.addWidget(filter_wrap)
                layout.addWidget(title_bar)


                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(8)
                container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

                quick_row = QHBoxLayout()
                quick_row.setContentsMargins(0, 0, 0, 0)
                quick_row.setSpacing(6)
                quick_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
                selected_filter = [None]

                scroll = wrap_scroll(container)
                layout.addLayout(quick_row)
                layout.addWidget(scroll)


                def highlight_buttons(active_btn):
                    for i in range(quick_row.count()):
                        btn = quick_row.itemAt(i).widget()
                        if isinstance(btn, QPushButton):
                            if active_btn is not None and btn == active_btn:
                                btn.setStyleSheet("""background:#00BCD4;color:#fff;padding:4px 12px;
                                                    border-radius:6px;font-weight:bold;""")
                            else:
                                btn.setStyleSheet("""background:#444;color:#fff;padding:4px 12px;
                                                    border-radius:6px;""")


                for label, days in [("Day", 1), ("Week", 7), ("Month", 30), ("Year", 365)]:
                    btn = QPushButton(f"Past {label}")
                    btn.setStyleSheet("background:#444;color:#fff;padding:4px 12px;border-radius:6px;")
                    def handler(checked=False, d=days, b=btn):
                        selected_filter[0] = d
                        apply_date_range(d)
                        highlight_buttons(b)
                    btn.clicked.connect(handler)
                    quick_row.addWidget(btn)



                pagination = QHBoxLayout()
                pagination.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                prev_btn = QPushButton("←")
                next_btn = QPushButton("→")
                page_lbl = QLineEdit()
                page_lbl.setFixedWidth(60)
                page_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                page_lbl.setStyleSheet("""
                    QLineEdit {
                        background: transparent;
                        color: #ccc;
                        border: none;
                        font-weight: bold;
                    }
                """)


                current_page = [0]

                for btn in (prev_btn, next_btn):
                    btn.setFixedSize(32, 32)
                    btn.setStyleSheet("""
                        QPushButton {
                            background: #5C6BC0;
                            color: #fff;
                            border-radius: 6px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background: #3F51B5;
                        }
                    """)


                page_input = QLineEdit("1")
                page_input.setFixedWidth(40)
                page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
                page_input.setValidator(QIntValidator(1, 999999999))
                page_input.setStyleSheet("color:#fff;background:#222;border-radius:4px;padding:2px 4px;")

                total_pages_lbl = QLabel("/ 1")
                total_pages_lbl.setStyleSheet("color:#ccc;")

                pagination.addWidget(prev_btn)
                pagination.addWidget(page_input)
                pagination.addWidget(total_pages_lbl)
                pagination.addWidget(next_btn)
                pagination_widget = QWidget()
                pagination_widget.setLayout(pagination)
                pagination_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
                layout.addWidget(pagination_widget, alignment=Qt.AlignmentFlag.AlignHCenter)





                add_btn = QPushButton("Add Nutrition Log")
                add_btn.setFixedSize(200, 40)
                add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                add_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #5C6BC0;
                        color: white;
                        padding: 10px 24px;
                        border-radius: 8px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3F51B5;
                    }
                """)
                layout.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
                add_btn.clicked.connect(lambda: open_nutrition_dialog("New Nutrition Log"))

                def open_nutrition_dialog(title_text: str, existing: dict = None):
                    dlg = QDialog()
                    dlg.setWindowTitle(title_text)
                    dlg.setWindowIcon(QIcon(resource_path("assets/nutrition.png")))
                    dlg.setStyleSheet("""
                        QDialog {
                            background: #2e2e3e;
                            color: #fff;
                            border: none;
                        }
                        QScrollBar:vertical {
                            background: #2e2e3e;
                            width: 12px;
                            margin: 0;
                            border: none;
                        }
                        QScrollBar::handle:vertical {
                            background: #555;
                            border-radius: 6px;
                            min-height: 20px;
                        }
                        QScrollBar::handle:vertical:hover {
                            background: #888;
                        }
                        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                            height: 0;
                        }
                    """)

                    layout = QVBoxLayout(dlg)
                    layout.setContentsMargins(16, 16, 16, 16)
                    layout.setSpacing(10)


                    date_input = QDateEdit(QDate.currentDate())
                    date_input.setCalendarPopup(True)
                    date_input.setDisplayFormat("yyyy-MM-dd")
                    time_input = QTimeEdit(QTime.currentTime())
                    time_input.setDisplayFormat("HH:mm")
                    for w in (date_input, time_input):
                        w.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                    top_grid = QGridLayout()
                    top_grid.addWidget(QLabel("Date"), 0, 0)
                    top_grid.addWidget(date_input, 1, 0)
                    top_grid.addWidget(QLabel("Time"), 0, 1)
                    top_grid.addWidget(time_input, 1, 1)
                    layout.addLayout(top_grid)


                    food_rows = []
                    food_container = QWidget()
                    food_layout = QVBoxLayout(food_container)
                    food_layout.setSpacing(16)
                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll.setWidget(food_container)
                    scroll.setStyleSheet("background:#2e2e3e;border: 1px solid #555;border-radius: 6px;")
                    layout.addWidget(scroll)

                    def add_food_entry(data=None):
                        box = QGroupBox()
                        box.setStyleSheet("QGroupBox { border: none; }")
                        vbox = QVBoxLayout(box)

                        meal_box = QComboBox(); meal_box.addItems(["Breakfast", "Lunch", "Dinner", "Snack", "Other"])
                        meal_other = QLineEdit(); meal_other.setPlaceholderText("Custom meal"); meal_other.setVisible(False)
                        food_input = QLineEdit(); food_input.setPlaceholderText("e.g. Apple")
                        qty_input = QLineEdit(); qty_input.setPlaceholderText("Qty")
                        unit_box = QComboBox(); unit_box.addItems(["g", "ml", "slice", "cup", "tbsp", "oz", "piece", "serving", "bowl", "Other"])
                        unit_other = QLineEdit(); unit_other.setPlaceholderText("Custom unit"); unit_other.setVisible(False)
                        cal_input = ClearableLineEdit("0")

                        for w in (meal_box, meal_other, food_input, qty_input, unit_box, unit_other, cal_input):
                            w.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")

                        def Label(text):
                            l = QLabel(text)
                            l.setStyleSheet("border: none; color: #fff;")
                            return l

                        form = QGridLayout()
                        form.addWidget(Label("Meal Type"), 0, 0)
                        form.addWidget(meal_box, 1, 0)
                        form.addWidget(Label("Food Name"), 0, 1)
                        form.addWidget(food_input, 1, 1)
                        form.addWidget(meal_other, 2, 0)
                        form.addWidget(Label("Quantity"), 3, 0)
                        form.addWidget(qty_input, 4, 0)
                        form.addWidget(Label("Unit"), 3, 1)
                        form.addWidget(unit_box, 4, 1)
                        form.addWidget(unit_other, 5, 1)
                        form.addWidget(Label("Calories"), 6, 0)
                        form.addWidget(cal_input, 7, 0)

                        vbox.addLayout(form)
                        food_layout.addWidget(box)

                        separator = QFrame()
                        separator.setFrameShape(QFrame.Shape.HLine)
                        separator.setStyleSheet("color: #555; background-color: #555; height: 1px;")
                        food_layout.addWidget(separator)

                        food_rows.append({
                            "meal_box": meal_box,
                            "meal_other": meal_other,
                            "food_input": food_input,
                            "qty_input": qty_input,
                            "unit_box": unit_box,
                            "unit_other": unit_other,
                            "cal_input": cal_input,
                        })

                        def upd_vis():
                            meal_other.setVisible(meal_box.currentText() == "Other")
                            unit_other.setVisible(unit_box.currentText() == "Other")
                            QTimer.singleShot(0, lambda: dlg.resize(560, dlg.sizeHint().height()))

                        meal_box.currentTextChanged.connect(upd_vis)
                        unit_box.currentTextChanged.connect(upd_vis)
                        upd_vis()

                        if data:
                            mt = data.get("meal_type", "Other")
                            meal_box.setCurrentText(mt if mt in [meal_box.itemText(i) for i in range(meal_box.count())] else "Other")
                            meal_other.setText(mt if meal_box.currentText() == "Other" else "")
                            food_input.setText(data.get("food_name", ""))
                            qty_input.setText(str(data.get("quantity", "")))
                            u = data.get("unit", "Other")
                            unit_box.setCurrentText(u if u in [unit_box.itemText(i) for i in range(unit_box.count())] else "Other")
                            unit_other.setText(u if unit_box.currentText() == "Other" else "")
                            cal_input.setText(str(data.get("calories", 0)))


                    if existing:
                        date_input.setDate(QDate.fromString(existing["date"], "yyyy-MM-dd"))
                        time_input.setTime(QTime.fromString(existing["time"], "HH:mm"))
                        for f in existing.get("foods", []): add_food_entry(f)
                    else:
                        add_food_entry()

                    add_btn = QPushButton("+ Add Another Food")
                    add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    add_btn.setStyleSheet("""
                        QPushButton { background:#666;padding:8px;border-radius:6px; }
                        QPushButton:hover { background:#888; }
                    """)
                    add_btn.clicked.connect(lambda: add_food_entry())
                    layout.addWidget(add_btn)

                    calc_btn = QPushButton("Calculate Calories for All")
                    calc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    calc_btn.setStyleSheet("""
                        QPushButton { background:#5a9;color:#fff;padding:8px;border-radius:6px; }
                        QPushButton:hover { background:#6bcaa3; }
                    """)
                    layout.addWidget(calc_btn)

                    def calculate_all():
                        entries, refs = [], []

                        for idx, row in enumerate(food_rows):
                            name = row["food_input"].text().strip()
                            try:
                                qty = float(row["qty_input"].text().strip())
                                assert qty > 0
                            except:
                                continue
                            unit = (
                                row["unit_other"].text().strip()
                                if row["unit_box"].currentText() == "Other"
                                else row["unit_box"].currentText()
                            )
                            if not name or not unit:
                                continue

                            query = f"{qty} {unit} of {name}"
                            entries.append(query)
                            refs.append(row["cal_input"])

                        if not entries:
                            return

                        try:
                            print("Nutritionix Query:", ", ".join(entries))
                            resp = requests.post(
                                "https://trackapi.nutritionix.com/v2/natural/nutrients",
                                headers=nutritionix_headers,
                                json={"query": ", ".join(entries)}
                            ).json()
                            print("Nutritionix Response:", json.dumps(resp, separators=(',', ':')))

                            foods = resp.get("foods", [])
                            total = 0

                            for i, item in enumerate(foods):
                                cal = item.get("nf_calories")
                                if i >= len(refs):
                                    continue
                                if cal is None:
                                    refs[i].setText("Couldn't calculate calorie")
                                    refs[i].setStyleSheet("color:#f55;background:#444;padding:6px;border-radius:6px;")
                                else:
                                    refs[i].setText(str(int(cal)))
                                    refs[i].setStyleSheet("color:#fff;background:#444;padding:6px;border-radius:6px;")
                                    total += int(cal)

                            for j in range(len(foods), len(refs)):
                                refs[j].setText("Couldn't calculate calorie")
                                refs[j].setStyleSheet("color:#f55;background:#444;padding:6px;border-radius:6px;")

                            total_calories.setText(str(total))

                        except Exception as e:
                            print("Nutritionix error:", e)



                    calc_btn.clicked.connect(calculate_all)

                    total_calories = QLineEdit("0")
                    total_calories.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                    notes_input = QTextEdit()
                    notes_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                    notes_input.setPlaceholderText("Notes...")
                    notes_input.setFixedHeight(120)
                    layout.addWidget(QLabel("Total Calories"))
                    layout.addWidget(total_calories)
                    layout.addWidget(QLabel("Notes"))
                    layout.addWidget(notes_input)

                    save_btn = QPushButton("Save Log")
                    save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    save_btn.setStyleSheet("""
                        QPushButton { background:#00BCD4;padding:10px;border-radius:8px;font-weight:bold; }
                        QPushButton:hover { background:#0097a7; }
                    """)
                    layout.addWidget(save_btn)

                    def save():
                        confirm = QMessageBox.question(dlg, "Confirm Save", "Are you sure you want to save this nutrition log?",
                                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        if confirm != QMessageBox.StandardButton.Yes:
                            return

                        date = date_input.date().toString("yyyy-MM-dd")
                        time = time_input.time().toString("HH:mm")
                        notes = notes_input.toPlainText().strip()

                        entries = []
                        for row in food_rows:
                            meal = row["meal_other"].text().strip() if row["meal_box"].currentText() == "Other" else row["meal_box"].currentText()
                            unit = row["unit_other"].text().strip() if row["unit_box"].currentText() == "Other" else row["unit_box"].currentText()
                            food_name = row["food_input"].text().strip()
                            try:
                                quantity = float(row["qty_input"].text().strip() or 0)
                                calories = int(row["cal_input"].text().strip() or 0)
                            except:
                                continue

                            entry = {
                                "meal_type": meal,
                                "food_name": food_name,
                                "quantity": quantity,
                                "unit": unit,
                                "calories": calories,
                                "date": date,
                                "time": time,
                                "notes": notes
                            }
                            entries.append(entry)

                        if existing:
                            me["nutrition_log"] = [e for e in me["nutrition_log"] if e not in existing.get("foods", [])]

                        me.setdefault("nutrition_log", []).extend(entries)
                        Path("user_data.json").write_text(json.dumps(users, indent=4))
                        dlg.accept()
                        paged_refresh()

                    save_btn.clicked.connect(save)
                    QTimer.singleShot(0, lambda: dlg.resize(560, dlg.sizeHint().height()))

                    dlg.exec()
                def delete_log(log):
                    me["nutrition_log"].remove(log)
                    Path("user_data.json").write_text(json.dumps(users, indent=4))
                    paged_refresh()

                def log_card(log):
                    f = QFrame()
                    f.setStyleSheet("background:#37474F;border-radius:10px;")
                    f.setMinimumHeight(70)
                    shadow_effect = QGraphicsDropShadowEffect()
                    shadow_effect.setOffset(0, 4)
                    shadow_effect.setBlurRadius(10)
                    semi_transparent_white = QColor(0, 0, 0, int(255 * 0.3))
                    shadow_effect.setColor(semi_transparent_white)
                    f.setGraphicsEffect(shadow_effect)

                    layout = QVBoxLayout(f)
                    layout.setContentsMargins(12, 8, 12, 8)

                    top = QHBoxLayout()
                    title = QLabel(f"{log['meal_type']} – {log['food_name']} ({log['quantity']} {log['unit']})")
                    title.setStyleSheet("color:#00BCD4;font-size:14pt;font-weight:600;")

                    edit_btn = QPushButton()
                    edit_btn.setIcon(QIcon(resource_path("assets/edit.png")))
                    edit_btn.setIconSize(QSize(20, 20))
                    edit_btn.setFixedSize(28, 28)
                    edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    edit_btn.setStyleSheet("border:none;background:transparent;")

                    remove_btn = QPushButton()
                    remove_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                    remove_btn.setIconSize(QSize(20, 20))
                    remove_btn.setFixedSize(28, 28)
                    remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    remove_btn.setStyleSheet("border:none;background:transparent;")

                    edit_btn.clicked.connect(lambda: open_nutrition_dialog("Edit Nutrition Log", log))
                    remove_btn.clicked.connect(lambda: delete_log(log))

                    btns = QHBoxLayout()
                    btns.addWidget(edit_btn)
                    btns.addWidget(remove_btn)

                    top.addWidget(title)
                    top.addStretch()
                    top.addLayout(btns)
                    layout.addLayout(top)

                    details = f"Date: {log['date']} | Time: {log['time']} | Calories: {log['calories']} kcal"
                    lbl = QLabel(details)
                    lbl.setStyleSheet("color:#ccc;font-size:10pt;")
                    layout.addWidget(lbl)

                    if log.get("notes"):
                        notes_lbl = QLabel(f"Notes: {log['notes']}")
                        notes_lbl.setStyleSheet("color:#aaa;font-size:9pt;")
                        notes_lbl.setWordWrap(True)
                        layout.addWidget(notes_lbl)

                    return f

                def refresh_logs(filtered=None):

                    last_filtered_logs[0] = filtered
                    for i in reversed(range(container_layout.count())):
                        w = container_layout.itemAt(i).widget()
                        if w:
                            container_layout.removeWidget(w)
                            w.deleteLater()


                    full_logs = me["nutrition_log"]
                    logs = filtered if filtered is not None else full_logs
                    logs = sorted(logs, key=lambda x: (x["date"], x["time"]), reverse=True)


                    if not logs:
                        msg = QLabel()
                        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        msg.setStyleSheet("color:#888;font-style:italic;padding:20px;")

                        if not full_logs:
                            msg.setText("You haven't logged any nutrition entries yet.")
                        else:
                            msg.setText("No nutrition logs found for the selected range.")

                        container_layout.addWidget(msg)
                        page_input.setText("0")
                        total_pages_lbl.setText("/ 0")
                        return


                    total_pages = max(1, (len(logs) + MAX_PER_PAGE - 1) // MAX_PER_PAGE)
                    current_page[0] = max(0, min(current_page[0], total_pages - 1))
                    page_input.setText(str(current_page[0] + 1))
                    total_pages_lbl.setText(f"/ {total_pages}")

                    start = current_page[0] * MAX_PER_PAGE
                    for log in logs[start:start + MAX_PER_PAGE]:
                        container_layout.addWidget(log_card(log))

                def jump_to_page():
                    try:

                        logs = me["nutrition_log"]
                        logs = sorted(logs, key=lambda x: (x["date"], x["time"]), reverse=True)
                        total_pages = max(1, (len(logs) + MAX_PER_PAGE - 1) // MAX_PER_PAGE)


                        requested = int(page_input.text()) - 1
                        if requested >= total_pages:
                            requested = total_pages - 1
                        if requested < 0:
                            requested = 0

                        current_page[0] = requested
                        paged_refresh()

                    except ValueError:

                        page_input.setText(str(current_page[0] + 1))
                page_input.editingFinished.connect(jump_to_page)




                def apply_date_range(days):
                    cutoff = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
                    filtered = [l for l in me["nutrition_log"] if l["date"] >= cutoff]
                    filter_applied[0] = True
                    clear_filter_btn.setVisible(True)
                    refresh_logs(filtered)

                def open_filter_dialog():
                    dlg = QDialog(self)
                    dlg.setWindowTitle("Filter Nutrition Log")
                    dlg.resize(420, dlg.sizeHint().height())
                    dlg.setStyleSheet("background:#2e2e3e;color:#fff;")
                    v = QVBoxLayout(dlg)
                    v.setContentsMargins(16, 16, 16, 16)

                    meal_box = QComboBox()
                    meal_box.addItem("All")
                    meal_box.addItems(["Breakfast", "Lunch", "Dinner", "Snack"])
                    v.addWidget(QLabel("Meal Type"))
                    v.addWidget(meal_box)
                    calorie_box = QComboBox()
                    calorie_box.addItems(["Any", "≥", "≤", "="])
                    v.addWidget(QLabel("Calories"))
                    v.addWidget(calorie_box)

                    calorie_input = QLineEdit()
                    calorie_input.setPlaceholderText("e.g. 200")
                    calorie_input.setValidator(QIntValidator(0, 10000))
                    calorie_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                    v.addWidget(calorie_input)


                    date_start = QDateEdit()
                    date_start.setDisplayFormat("yyyy-MM-dd")
                    date_start.setCalendarPopup(True)
                    date_start.setDate(QDate.currentDate().addDays(-7))
                    v.addWidget(QLabel("Start Date"))
                    v.addWidget(date_start)

                    date_end = QDateEdit()
                    date_end.setDisplayFormat("yyyy-MM-dd")
                    date_end.setCalendarPopup(True)
                    date_end.setDate(QDate.currentDate())
                    v.addWidget(QLabel("End Date"))
                    v.addWidget(date_end)
                    def apply():
                        s = date_start.date().toString("yyyy-MM-dd")
                        e = date_end.date().toString("yyyy-MM-dd")
                        m = meal_box.currentText()
                        rel = calorie_box.currentText()
                        try:
                            cal_val = int(calorie_input.text()) if rel != "Any" and calorie_input.text() else None
                        except:
                            cal_val = None

                        def in_range(d): return s <= d <= e

                        def matches_cal(c):
                            if cal_val is None:
                                return True
                            if rel == "≥":
                                return c >= cal_val
                            if rel == "≤":
                                return c <= cal_val
                            if rel == "=":
                                return c == cal_val
                            return True

                        filtered = [
                            l for l in me["nutrition_log"]
                            if (m == "All" or l["meal_type"] == m)
                            and in_range(l["date"])
                            and matches_cal(l["calories"])
                        ]

                        refresh_logs(filtered)
                        clear_filter_btn.setVisible(True)
                        filter_applied[0] = True
                        selected_filter[0] = None
                        highlight_buttons(None)
                        dlg.accept()


                    btn = QPushButton("Apply Filter")
                    btn.setStyleSheet("background:#00BCD4;border-radius:6px;padding:8px;font-weight:bold;")
                    btn.clicked.connect(apply)
                    v.addWidget(btn)

                    dlg.exec()


                def clear_filter():
                    refresh_logs()
                    filter_applied[0] = False
                    clear_filter_btn.setVisible(False)
                    selected_filter[0] = None
                    highlight_buttons(None)

                def paged_refresh():
                    if selected_filter[0]:
                        apply_date_range(selected_filter[0])
                    elif filter_applied[0]:
                        refresh_logs(last_filtered_logs[0])
                    else:
                        refresh_logs()


                prev_btn.clicked.connect(lambda: (current_page.__setitem__(0, max(0, current_page[0] - 1)), paged_refresh()))
                next_btn.clicked.connect(lambda: (current_page.__setitem__(0, current_page[0] + 1), paged_refresh()))

                filter_btn.clicked.connect(open_filter_dialog)
                clear_filter_btn.clicked.connect(clear_filter)

                refresh_logs()
                return page


            tabs.addTab(create_summary_tab(), "Summary")
            tabs.addTab(create_food_guide_tab(), "Food Guide")
            tabs.addTab(create_fitness_calculators_tab(), "Fitness Calculators")
            tabs.addTab(create_log_nutrition_tab(), "Log Nutrition")

            layout.addWidget(tabs)
            return page

        def create_fitness_goals_tab() -> QWidget:

            DATA_FILE = Path("user_data.json")


            try:
                users = json.loads(DATA_FILE.read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []


            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            if me is None:
                me = {}
            me.setdefault("goals", {})



            class ConcentricActivityRings(QWidget):
                def __init__(self, goals_progress, colors):
                    super().__init__()
                    self.goals_progress = goals_progress
                    self.colors = colors
                    self.setFixedSize(400, 400)

                def paintEvent(self, event):
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    center = self.rect().center()

                    ring_thickness = 32
                    spacing = 8
                    base_radius = 176

                    base_circumference = 2 * pi * (base_radius - ring_thickness / 2 + 18)
                    base_deg_per_px = 360 / base_circumference
                    fixed_spacing_px = 2
                    padding_deg = fixed_spacing_px * base_deg_per_px

                    for i, (label, (value, goal)) in enumerate(self.goals_progress.items()):
                        progress = min(value / goal, 1.0) if goal > 0 else 0
                        angle_span = int(progress * 360 * 16)
                        radius = base_radius - i * (ring_thickness + spacing)

                        rect = QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)

                        muted = QColor(self.colors[i % len(self.colors)])
                        muted.setAlpha(60)
                        pen_bg = QPen(muted, ring_thickness)
                        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
                        painter.setPen(pen_bg)
                        painter.drawArc(rect, 0, 360 * 16)

                        pen_fg = QPen(QColor(self.colors[i % len(self.colors)]), ring_thickness)
                        pen_fg.setCapStyle(Qt.PenCapStyle.RoundCap)
                        painter.setPen(pen_fg)
                        painter.drawArc(rect, 90 * 16, -angle_span)

                        painter.setFont(QFont("Arial", 10, QFont.Weight.ExtraBold))
                        painter.setPen(QColor("white"))

                        text = f"{label.upper()} {int(progress * 100)}%"
                        font_metrics = QFontMetrics(painter.font())

                        text_radius = radius - ring_thickness / 2 + 18
                        circumference = 2 * pi * text_radius
                        deg_per_px = 360 / circumference

                        char_infos = []
                        for ch in text:
                            ch_width = font_metrics.horizontalAdvance(ch)
                            ch_arc_deg = ch_width * deg_per_px
                            char_infos.append((ch, ch_arc_deg))

                        angle_end_deg = 90 - progress * 360
                        current_angle_deg = angle_end_deg

                        for ch, ch_arc_deg in reversed(char_infos):
                            current_angle_deg += ch_arc_deg / 2
                            angle_rad = current_angle_deg * pi / 180

                            x = center.x() + text_radius * cos(angle_rad)
                            y = center.y() - text_radius * sin(angle_rad)

                            painter.save()
                            painter.translate(x, y)
                            painter.rotate(-current_angle_deg + 90)
                            painter.drawText(QRectF(-20, -10, 40, 20), Qt.AlignmentFlag.AlignCenter, ch)
                            painter.restore()

                            current_angle_deg += ch_arc_deg / 2 + padding_deg

            def create_weekly_trend_chart(field_key, title, daily_goals_list, daily_completed_list):
                today = datetime.now().date()

                def week_start(date):
                    days_since_sunday = (date.weekday() + 1) % 7
                    return date - timedelta(days=days_since_sunday)

                current_week_start = week_start(today)
                current = current_week_start
                animation_running = False

                container = QWidget()
                layout = QVBoxLayout(container)
                layout.setContentsMargins(8, 8, 8, 8)

                label = QLabel(title)
                label.setStyleSheet("color:#ccc; font-weight:bold; font-size:12pt;")
                layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)

                week_range_label = QLabel("")
                week_range_label.setStyleSheet("color:#aaa; font-size:9pt;")
                layout.addWidget(week_range_label, alignment=Qt.AlignmentFlag.AlignCenter)

                canvas_container = QWidget()
                canvas_container.setFixedSize(400, 150)


                h_center_layout = QHBoxLayout()
                h_center_layout.addStretch()
                h_center_layout.addWidget(canvas_container)
                h_center_layout.addStretch()
                layout.addLayout(h_center_layout)

                fig1 = Figure(figsize=(4, 1.5), dpi=100, facecolor="#333")
                fig1.subplots_adjust(left=0.15, bottom=0.4)
                ax1 = fig1.add_subplot(111)
                canvas1 = FigureCanvas(fig1)
                canvas1.setParent(canvas_container)
                canvas1.move(0, 0)
                canvas1.resize(canvas_container.size())

                fig2 = Figure(figsize=(4, 1.5), dpi=100, facecolor="#333")
                fig2.subplots_adjust(left=0.15, bottom=0.4)
                ax2 = fig2.add_subplot(111)
                canvas2 = FigureCanvas(fig2)
                canvas2.setParent(canvas_container)
                canvas2.move(canvas_container.width(), 0)
                canvas2.resize(canvas_container.size())

                nav_layout = QHBoxLayout()
                btn_prev = QPushButton()
                btn_prev.setIcon(QIcon(resource_path("assets/prev.png")))

                btn_next = QPushButton()
                btn_next.setIcon(QIcon(resource_path("assets/next.png")))

                for btn in (btn_prev, btn_next):
                    btn.setIconSize(QSize(30, 30))
                    btn.setFixedSize(36, 36)
                    btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn.setStyleSheet("""
                        QPushButton {
                            border: none;
                            background-color: transparent;
                            border-radius: 8px;
                        }
                        QPushButton:hover {
                            background-color: rgba(36,146,255,50);  /* light gray on hover */
                        }
                    """)

                nav_layout.addWidget(btn_prev)

                spacer = QWidget()
                spacer.setFixedWidth(400)
                nav_layout.addWidget(spacer)
                nav_layout.addWidget(btn_next)


                nav_container = QWidget()
                nav_container.setLayout(nav_layout)
                nav_container.setFixedWidth(460)
                h_nav_layout = QHBoxLayout()
                h_nav_layout.addStretch()
                h_nav_layout.addWidget(nav_container)
                h_nav_layout.addStretch()
                layout.addLayout(h_nav_layout)

                def get_entry_for_date(entry_list, date):
                    for entry in entry_list:
                        if entry.get("date") == date.strftime("%Y-%m-%d"):
                            return entry.get(field_key, 0)
                    return 0

                def update_week_range_label(start_date):
                    end_date = start_date + timedelta(days=6)
                    week_range_label.setText(f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}")

                def draw_week(ax, canvas, week_start_date):
                    ax.clear()
                    ax.set_facecolor("#333")
                    ax.grid(True, linestyle='--', alpha=0.2, color="#fff")
                    ax.tick_params(colors='white', labelsize=8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')

                    dates = [week_start_date + timedelta(days=i) for i in range(7)]
                    labels = [d.strftime("%a") for d in dates]

                    targets = [get_entry_for_date(daily_goals_list, d) for d in dates]
                    completed = [get_entry_for_date(daily_completed_list, d) for d in dates]

                    max_target = max(targets) if targets else 1
                    y_max = max_target * 1.2 if max_target > 0 else 1
                    ax.set_ylim(0, y_max)

                    bar_colors = []
                    for i, d in enumerate(dates):
                        if d > today:
                            bar_colors.append("#888")
                        elif completed[i] >= targets[i]:
                            bar_colors.append("#4CAF50")
                        else:
                            bar_colors.append("#F44336")

                    ax.bar(range(7), completed, color=bar_colors, alpha=0.8, width=0.5, label="Completed")

                    target_points = [(i, t) for i, t in enumerate(targets) if t > 0]

                    if target_points:
                        x_vals, y_vals = zip(*target_points)
                        ax.plot(x_vals, y_vals, color="#FFEB3B", linewidth=2, label="Target")


                    ax.set_xticks(range(7))
                    ax.set_xticklabels(labels, color="white")
                    ax.set_ylabel(field_key.replace('_', ' ').title(), color="white")

                    for i, val in enumerate(completed):
                        ax.text(i, val + y_max * 0.03, str(val), ha='center', color='white', fontsize=7)

                    ax.legend(loc='upper center', bbox_to_anchor=(0.29, 1.31), fontsize=7,
                                facecolor="#222", edgecolor="#444", labelcolor="white", ncol=2)

                    canvas.draw()

                draw_week(ax1, canvas1, current)
                update_week_range_label(current)

                animation_running = False

                def slide_to_week(new_week_start, direction=1):
                    nonlocal current, animation_running
                    if animation_running:
                        return
                    animation_running = True

                    draw_week(ax2, canvas2, new_week_start)

                    width = canvas1.width()
                    canvas1.move(0, 0)
                    canvas2.move(direction * width, 0)

                    anim1 = QPropertyAnimation(canvas1, b"pos", container)
                    anim1.setDuration(400)
                    anim1.setStartValue(canvas1.pos())
                    anim1.setEndValue(QPoint(-direction * width, 0))
                    anim1.setEasingCurve(QEasingCurve.Type.InOutQuad)

                    anim2 = QPropertyAnimation(canvas2, b"pos", container)
                    anim2.setDuration(400)
                    anim2.setStartValue(canvas2.pos())
                    anim2.setEndValue(QPoint(0, 0))
                    anim2.setEasingCurve(QEasingCurve.Type.InOutQuad)

                    def on_anim_finished():
                        nonlocal ax1, ax2, canvas1, canvas2, current, animation_running
                        ax1.clear()
                        canvas1.move(width, 0)

                        current = new_week_start
                        update_week_range_label(current)

                        ax1, ax2 = ax2, ax1
                        canvas1, canvas2 = canvas2, canvas1

                        animation_running = False

                    anim2.finished.connect(on_anim_finished)

                    anim1.start()
                    anim2.start()

                    container._anim1 = anim1
                    container._anim2 = anim2

                def prev_week():
                    new_week = current - timedelta(days=7)
                    slide_to_week(new_week, direction=-1)

                def next_week():
                    new_week = current + timedelta(days=7)
                    if new_week <= current_week_start:
                        slide_to_week(new_week, direction=1)

                btn_prev.clicked.connect(prev_week)
                btn_next.clicked.connect(next_week)

                return container

            page = QWidget()
            overlay = CelebrationOverlay()

            def show_celebration_if_needed():
                today_str = QDate.currentDate().toString("yyyy-MM-dd")
                if me["goals"].get("daily_celebration", {}).get("date") != today_str:
                    me["goals"]["daily_celebration"] = {"date": today_str, "seen": True}
                    save_to_file()
                    overlay.show_celebration_for("daily", duration_ms=5000)

            main_layout = QVBoxLayout(page)
            main_layout.setContentsMargins(16, 16, 16, 16)
            main_layout.setSpacing(10)

            title = QLabel("Daily Goals")
            title.setStyleSheet("color:#8AB6D6; font-size:20pt; font-weight:bold;")
            main_layout.addWidget(title)


            btn_frame = QFrame()
            btn_layout = QHBoxLayout(btn_frame)
            btn_layout.setSpacing(20)

            btn_style = """
                QPushButton {
                    background-color: #5C6BC0;
                    color: white;
                    padding: 10px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3F51B5;
                }
            """

            set_target_btn = QPushButton("Set Target")
            update_goals_btn = QPushButton("Update Goals")
            for btn in (set_target_btn, update_goals_btn):
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet(btn_style)
                btn_layout.addWidget(btn)
            main_layout.addWidget(btn_frame, alignment=Qt.AlignmentFlag.AlignLeft)


            today = datetime.now().strftime("%Y-%m-%d")
            daily_goals_list = me["goals"].get("daily", [])
            daily_completed_list = me["goals"].get("daily_completed", [])
            me["goals"]["daily"] = daily_goals_list
            me["goals"]["daily_completed"] = daily_completed_list

            def get_today_entry(entry_list):
                for entry in entry_list:
                    if entry.get("date") == today:
                        return entry
                new_entry = {"date": today, "steps": 0, "exercise_duration": 0, "calories": 0, "water_intake": 0}
                entry_list.append(new_entry)
                return new_entry

            daily_goals = get_today_entry(daily_goals_list)
            daily_completed = get_today_entry(daily_completed_list)

            def save_to_file():
                try:
                    me["goals"]["daily"] = daily_goals_list
                    me["goals"]["daily_completed"] = daily_completed_list
                    DATA_FILE.write_text(json.dumps(users, indent=4))
                except Exception as e:
                    print("Error saving user_data.json:", e)

            save_to_file()

            def show_input_dialog(title_text, initial_data, max_values=None, is_progress=False, on_done=None):
                dlg = QDialog(page)
                dlg.setWindowTitle(title_text)

                dlg.setWindowIcon(QIcon(resource_path("assets/goals.png")))

                dlg_layout = QVBoxLayout(dlg)
                grid = QGridLayout()
                grid.setSpacing(12)
                inputs = {}

                def add_input(label, row, unit=""):
                    key = label.lower().replace(" ", "_")
                    lbl = QLabel(label)
                    lbl.setStyleSheet("color:#ddd;")
                    spin = QSpinBox()
                    spin.setRange(0, max_values.get(key, 100000) if max_values else 100000)
                    spin.setStyleSheet("""
                        QSpinBox {
                            background:#333;
                            color:#888;
                            border:1px solid #555;
                            border-radius:2px;
                        }
                        QSpinBox:focus {
                            color:#fff;
                        }
                    """)
                    spin.setValue(initial_data.get(key, 0))
                    spin.lineEdit().setPlaceholderText("0")
                    spin.setButtonSymbols(QSpinBox.ButtonSymbols.PlusMinus)
                    inputs[key] = spin
                    grid.addWidget(lbl, row, 0)
                    grid.addWidget(spin, row, 1)
                    if unit:
                        suffix_lbl = QLabel(f"{unit} / {max_values.get(key, '')}" if max_values else unit)
                        suffix_lbl.setStyleSheet("color:#aaa;")
                        grid.addWidget(suffix_lbl, row, 2)

                add_input("Steps", 0, "steps")
                add_input("Exercise Duration", 1, "min")
                add_input("Calories", 2, "kcal")
                add_input("Water Intake", 3, "ml")

                dlg_layout.addLayout(grid)

                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                dlg_layout.addWidget(buttons)

                def on_accept():
                    result = {k: spin.value() for k, spin in inputs.items()}
                    if on_done:
                        on_done(result)
                    dlg.close()

                def on_reject():
                    if on_done:
                        on_done(None)
                    dlg.close()

                buttons.accepted.connect(on_accept)
                buttons.rejected.connect(on_reject)

                dlg.exec()
                return dlg


            def on_set_target():
                def handle_result(result):
                    if result:
                        daily_goals.update(result)
                        daily_goals["date"] = today
                        save_to_file()
                        refresh_rings()

                show_input_dialog("Set Target Goals", daily_goals, on_done=handle_result)

            def on_update_goals():
                max_values = {
                    "steps": daily_goals.get("steps", 0),
                    "exercise_duration": daily_goals.get("exercise_duration", 0),
                    "calories": daily_goals.get("calories", 0),
                    "water_intake": daily_goals.get("water_intake", 0),
                }
                current = {
                    "steps": daily_completed.get("steps", 0),
                    "exercise_duration": daily_completed.get("exercise_duration", 0),
                    "calories": daily_completed.get("calories", 0),
                    "water_intake": daily_completed.get("water_intake", 0),
                }

                def handle_result(result):
                    if result:
                        daily_completed.update(result)
                        daily_completed["date"] = today
                        save_to_file()
                        refresh_rings()

                show_input_dialog("Update Completed Progress", current, max_values, is_progress=True, on_done=handle_result)

            set_target_btn.clicked.connect(on_set_target)
            update_goals_btn.clicked.connect(on_update_goals)

            def get_goals_progress():
                return {
                    "Steps": (daily_completed.get("steps", 0), daily_goals.get("steps", 1)),
                    "Duration": (daily_completed.get("exercise_duration", 0), daily_goals.get("exercise_duration", 1)),
                    "Calories": (daily_completed.get("calories", 0), daily_goals.get("calories", 1)),
                    "Water": (daily_completed.get("water_intake", 0), daily_goals.get("water_intake", 1)),
                }


            ring_colors = ["#4CAF50", "#C39200", "#F44336", "#00ACC1"]
            concentric_rings = ConcentricActivityRings(get_goals_progress(), ring_colors)

            def refresh_rings():
                concentric_rings.goals_progress = get_goals_progress()
                concentric_rings.update()
                show_celebration_if_needed()



            content_layout = QHBoxLayout()
            content_layout.setSpacing(12)
            content_layout.setContentsMargins(0, 0, 0, 0)

            ring_and_label_layout = QVBoxLayout()
            ring_and_label_layout.setSpacing(5)
            ring_and_label_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            ring_and_label_layout.addWidget(concentric_rings, alignment=Qt.AlignmentFlag.AlignHCenter)

            todays_goals_label = QLabel("Today's Goals")
            todays_goals_label.setStyleSheet("color:#aaa; font-size:13pt; font-weight:600;")
            ring_and_label_layout.addWidget(todays_goals_label, alignment=Qt.AlignmentFlag.AlignHCenter)


            left_side = QVBoxLayout()
            left_side.addStretch()
            left_side.addLayout(ring_and_label_layout)
            left_side.addStretch()


            left_widget = QWidget()
            left_widget.setLayout(left_side)
            content_layout.addWidget(left_widget)

            right_side = QVBoxLayout()

            trends_title = QLabel("Trends (Weekly)")
            trends_title.setStyleSheet("color:#8AB6D6; font-size:16pt; font-weight:bold;")
            right_side.addWidget(trends_title, alignment=Qt.AlignmentFlag.AlignLeft)

            trends_grid = QGridLayout()
            trends_grid.setSpacing(8)
            trends_grid.setContentsMargins(0, 0, 0, 0)

            def make_card(widget):
                card = QFrame()
                card.setStyleSheet("background-color: #333; border-radius: 10px;")
                layout = QVBoxLayout(card)
                layout.setContentsMargins(8, 8, 8, 8)
                layout.setSpacing(4)
                layout.addWidget(widget)
                return card

            steps_chart = make_card(create_weekly_trend_chart("steps", "Steps", daily_goals_list, daily_completed_list))
            calories_chart = make_card(create_weekly_trend_chart("calories", "Calories", daily_goals_list, daily_completed_list))
            water_chart = make_card(create_weekly_trend_chart("water_intake", "Hydration", daily_goals_list, daily_completed_list))
            exercise_chart = make_card(create_weekly_trend_chart("exercise_duration", "Exercise", daily_goals_list, daily_completed_list))

            trends_grid.addWidget(steps_chart, 0, 0)
            trends_grid.addWidget(calories_chart, 0, 1)
            trends_grid.addWidget(water_chart, 1, 0)
            trends_grid.addWidget(exercise_chart, 1, 1)

            trends_widget = QWidget()
            trends_widget.setLayout(trends_grid)
            right_side.addWidget(trends_widget)

            right_widget = QWidget()
            right_widget.setLayout(right_side)
            content_layout.addWidget(right_widget)


            main_layout.addLayout(content_layout)
            overlay.raise_()


            return page

        def create_sleep_tracker_tab() -> QWidget:
            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            me.setdefault("sleep_log", [])

            def create_sleep_log_tab() -> QWidget:
                logs_per_page = 6
                current_page = [1]
                last_filtered = [None]

                tab = QWidget()
                tab.setStyleSheet("""
                    background-color: #222;
                    border-top-left-radius: 0px;
                    border-top-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border-bottom-right-radius: 12px;
                """)
                layout = QVBoxLayout(tab)
                layout.setSpacing(12)
                layout.setContentsMargins(20, 20, 20, 20)
                layout.setAlignment(Qt.AlignmentFlag.AlignTop)


                title_bar = QWidget()
                title_layout = QHBoxLayout(title_bar)
                title_layout.setContentsMargins(0, 0, 0, 0)
                title_layout.setSpacing(4)
                title = QLabel("Sleep Log")
                title.setStyleSheet("color:#8AB6D6; font-size:20pt; font-weight:bold;")

                filter_btn = QPushButton()
                filter_btn.setIcon(QIcon(resource_path("assets/filter.png")))
                filter_btn.setIconSize(QSize(40, 40))
                filter_btn.setFixedSize(48, 48)
                filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                filter_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,255,255,0.1);
                        border-radius:6px;
                    }
                """)

                clear_filter_btn = QPushButton()
                clear_filter_btn.setIcon(QIcon(resource_path("assets/clear.png")))
                clear_filter_btn.setIconSize(QSize(16, 16))
                clear_filter_btn.setFixedSize(20, 20)
                clear_filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                clear_filter_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,255,255,0.1);
                        border-radius:10px;
                    }
                """)
                clear_filter_btn.setVisible(False)

                filter_wrap = QWidget()
                filter_layout = QGridLayout(filter_wrap)
                filter_layout.setContentsMargins(0, 0, 0, 0)
                filter_layout.setSpacing(0)
                filter_layout.addWidget(filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
                filter_layout.addWidget(clear_filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

                title_layout.addWidget(title)
                title_layout.addStretch()
                title_layout.addWidget(filter_wrap)
                layout.addWidget(title_bar)


                quick_row = QHBoxLayout()
                quick_row.setSpacing(6)
                quick_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
                selected_filter = [None]

                def highlight_buttons(btn):
                    for i in range(quick_row.count()):
                        w = quick_row.itemAt(i).widget()
                        w.setStyleSheet("background:#444;color:#fff;padding:4px 12px;border-radius:6px;")
                    if btn:
                        btn.setStyleSheet("background:#00BCD4;color:#fff;padding:4px 12px;border-radius:6px;font-weight:bold;")

                def apply_date_range(days):
                    cutoff = QDate.currentDate().addDays(-days)
                    logs = me.get("sleep_log", [])
                    filtered = [l for l in logs if QDate.fromString(l['date'], "yyyy-MM-dd") >= cutoff]
                    last_filtered[0] = filtered
                    clear_filter_btn.setVisible(True)
                    highlight_buttons(None)
                    current_page[0] = 1
                    refresh_sleep_logs(filtered)

                for label, days in [("Day", 1), ("Week", 7), ("Month", 30), ("Year", 365)]:
                    b = QPushButton(f"Past {label}")
                    b.clicked.connect(lambda _, d=days, btn=b: [apply_date_range(d), highlight_buttons(btn)])
                    quick_row.addWidget(b)

                highlight_buttons(None)
                layout.addLayout(quick_row)

                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setSpacing(8)
                container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
                scroll = wrap_scroll(container)
                layout.addWidget(scroll)

                pagination_layout = QHBoxLayout()
                pagination_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                prev_btn = QPushButton("←")
                next_btn = QPushButton("→")
                page_input = QLineEdit("1")
                page_input.setFixedWidth(40)
                page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
                page_input.setValidator(QIntValidator(1, 999999999))
                page_input.setStyleSheet("""
                    QLineEdit {
                        color: #fff;
                        background: #222;
                        border-radius: 4px;
                        padding: 2px 4px;
                    }
                """)

                total_pages_lbl = QLabel("/ 1")
                total_pages_lbl.setStyleSheet("color:#ccc;")

                for btn in (prev_btn, next_btn):
                    btn.setFixedSize(32, 32)
                    btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn.setStyleSheet("""
                        QPushButton {
                            background: #5C6BC0;
                            color: #fff;
                            border-radius: 6px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background: #3F51B5;
                        }
                    """)

                pagination_layout.addWidget(prev_btn)
                pagination_layout.addWidget(page_input)
                pagination_layout.addWidget(total_pages_lbl)
                pagination_layout.addWidget(next_btn)
                pagination_widget = QWidget()
                pagination_widget.setLayout(pagination_layout)
                pagination_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
                layout.addWidget(pagination_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

                def open_filter_dialog():
                    dlg = QDialog()
                    dlg.setWindowTitle("Filter Sleep Logs")
                    dlg.setWindowIcon(QIcon(resource_path("assets/filter.png")))
                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    dlg.setFixedWidth(350)
                    dlg.setStyleSheet(f"""
                        QDialog {{
                            background-color: #2e2e3e;
                            color: #fff;
                            border-radius: 12px;
                        }}
                        QLabel {{
                            color: #ccc;
                            font-size: 11pt;
                        }}
                        QComboBox, QDateEdit, QLineEdit {{
                            background: #444;
                            color: #fff;
                            border: none;
                            border-radius: 6px;
                            padding: 6px;
                        }}
                        QComboBox::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QComboBox::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}
                        QComboBox QAbstractItemView {{
                            background: #333;
                            color: #fff;
                            selection-background-color: #5C6BC0;
                            border: none;
                            outline: none;
                            padding: 4px;
                        }}
                        QDateEdit::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QDateEdit::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}
                        QPushButton {{
                            background-color: #5C6BC0;
                            color: white;
                            font-weight: bold;
                            padding: 8px;
                            border-radius: 6px;
                        }}
                        QPushButton:hover {{
                            background-color: #3F51B5;
                        }}
                    """)
                    layout = QVBoxLayout(dlg)
                    layout.setContentsMargins(20, 20, 20, 20)
                    layout.setSpacing(12)


                    start_date = QDateEdit()
                    start_date.setCalendarPopup(True)
                    start_date.setDate(QDate(2000, 1, 1))
                    end_date = QDateEdit()
                    end_date.setCalendarPopup(True)
                    end_date.setDate(QDate.currentDate())
                    layout.addWidget(QLabel("Start Date:"))
                    layout.addWidget(start_date)
                    layout.addWidget(QLabel("End Date:"))
                    layout.addWidget(end_date)


                    duration_operator = QComboBox()
                    duration_operator.addItems(["<", "=", ">"])

                    duration_input = QTimeEdit()
                    duration_input.setDisplayFormat("H:mm")
                    duration_input.setTime(QTime(0, 0))

                    duration_layout = QHBoxLayout()
                    duration_layout.addWidget(QLabel("Duration Filter:"))
                    duration_layout.addWidget(duration_operator)
                    duration_layout.addWidget(duration_input)
                    layout.addLayout(duration_layout)


                    session_combo = QComboBox()
                    session_combo.addItems(["All", "Nap", "Core Sleep", "Restorative", "Other"])
                    layout.addWidget(QLabel("Sleep Session:"))
                    layout.addWidget(session_combo)


                    apply_btn = QPushButton("Apply Filter")
                    layout.addWidget(apply_btn)

                    def apply_filters():
                        logs = me.get("sleep_log", [])
                        filtered = []

                        start_d = start_date.date()
                        end_d = end_date.date()
                        op = duration_operator.currentText()


                        dur_qtime = duration_input.time()
                        dur_val = dur_qtime.hour() * 60 + dur_qtime.minute()

                        sess = session_combo.currentText()

                        for log in logs:
                            d = QDate.fromString(log['date'], "yyyy-MM-dd")
                            if d < start_d or d > end_d:
                                continue


                            total_min = None
                            try:
                                h, m = map(int, log.get("duration", "0:0").split(":"))
                                total_min = h * 60 + m
                            except Exception:
                                pass

                            if dur_val is not None and total_min is not None:
                                if op == "<" and not (total_min < dur_val):
                                    continue
                                elif op == "=" and not (total_min == dur_val):
                                    continue
                                elif op == ">" and not (total_min > dur_val):
                                    continue


                            if sess != "All" and log.get("session", "Other") != sess:
                                continue

                            filtered.append(log)

                        last_filtered[0] = filtered
                        clear_filter_btn.setVisible(True)
                        current_page[0] = 1
                        refresh_sleep_logs(filtered)
                        dlg.accept()

                    apply_btn.clicked.connect(apply_filters)
                    dlg.exec()

                filter_btn.clicked.connect(open_filter_dialog)
                clear_filter_btn.clicked.connect(lambda: [
                    last_filtered.__setitem__(0, None),
                    refresh_sleep_logs(),
                    clear_filter_btn.setVisible(False),
                    highlight_buttons(None)
                ])

                def open_sleep_log_dialog(existing_log=None):
                    dlg = QDialog()
                    dlg.setWindowTitle("Log Sleep")
                    dlg.setWindowIcon(QIcon(resource_path("assets/sleeplog.png")))
                    dlg.setFixedWidth(300)
                    dlg.setStyleSheet("""
                        QDialog {
                            background:#2e2e3e;
                            color:#fff;
                        }
                        QLabel {
                            background: transparent;
                        }
                        QTimeEdit, QDateEdit, QComboBox, QTextEdit {
                            background:#444;
                            color:#fff;
                            padding:6px;
                            border-radius:6px;
                        }
                        QPushButton {
                            background: #5C6BC0;
                            color: #fff;
                            font-weight: bold;
                            padding: 8px;
                            border-radius: 6px;
                        }
                        QPushButton:hover {
                            background: #3F51B5;
                        }
                    """)

                    layout = QVBoxLayout(dlg)
                    layout.setContentsMargins(16, 16, 16, 16)

                    date_input = QDateEdit()
                    date_input.setCalendarPopup(True)
                    date_input.setDate(QDate.currentDate())

                    start_input = QTimeEdit()
                    end_input = QTimeEdit()
                    start_input.setDisplayFormat("hh:mm AP")
                    end_input.setDisplayFormat("hh:mm AP")

                    quality_input = QComboBox()
                    quality_input.addItems(["Excellent", "Good", "Fair", "Poor"])

                    session_input = QComboBox()
                    session_input.addItems(["Nap", "Core Sleep", "Restorative", "Other"])

                    note_input = QTextEdit()
                    note_input.setPlaceholderText("Optional note")
                    note_input.setFixedHeight(80)

                    layout.addWidget(QLabel("Date:"))
                    layout.addWidget(date_input)
                    layout.addWidget(QLabel("Start Time:"))
                    layout.addWidget(start_input)
                    layout.addWidget(QLabel("End Time:"))
                    layout.addWidget(end_input)
                    layout.addWidget(QLabel("Sleep Quality:"))
                    layout.addWidget(quality_input)
                    layout.addWidget(QLabel("Sleep Session:"))
                    layout.addWidget(session_input)
                    layout.addWidget(QLabel("Note:"))
                    layout.addWidget(note_input)

                    save_btn = QPushButton("Save")
                    layout.addWidget(save_btn)

                    if existing_log:
                        date_input.setDate(QDate.fromString(existing_log["date"], "yyyy-MM-dd"))
                        start_input.setTime(QTime.fromString(existing_log["start"], "HH:mm"))
                        end_input.setTime(QTime.fromString(existing_log["end"], "HH:mm"))
                        quality_input.setCurrentText(existing_log.get("quality", "Good"))
                        session_input.setCurrentText(existing_log.get("session", "Nap"))
                        note_input.setText(existing_log.get("note", ""))

                    def save():
                        start_time = start_input.time().toPyTime()
                        end_time = end_input.time().toPyTime()

                        start_dt = datetime.combine(datetime.today(), start_time)
                        end_dt = datetime.combine(datetime.today(), end_time)

                        if end_dt < start_dt:
                            end_dt += timedelta(days=1)

                        duration = str(end_dt - start_dt)[:-3]

                        log = {
                            "start": start_input.time().toString("HH:mm"),
                            "end": end_input.time().toString("HH:mm"),
                            "duration": duration,
                            "quality": quality_input.currentText(),
                            "session": session_input.currentText(),
                            "note": note_input.toPlainText().strip(),
                            "date": date_input.date().toString("yyyy-MM-dd"),
                            "time": datetime.now().strftime("%H:%M")
                        }

                        if existing_log:
                            me["sleep_log"].remove(existing_log)
                        me.setdefault("sleep_log", []).append(log)
                        Path("user_data.json").write_text(json.dumps(users, indent=4))
                        dlg.accept()
                        refresh_sleep_logs()

                    save_btn.clicked.connect(save)
                    QTimer.singleShot(0, lambda: dlg.resize(300, dlg.sizeHint().height()))
                    dlg.exec()

                def delete_sleep_log(log):
                    me["sleep_log"].remove(log)
                    Path("user_data.json").write_text(json.dumps(users, indent=4))
                    refresh_sleep_logs()

                def sleep_log_card(log):
                    def format_time_am_pm(raw_time):
                        qt = QTime.fromString(raw_time, "HH:mm")
                        if qt.isValid():
                            return qt.toString("hh:mm AP")
                        try:
                            parts = list(map(int, raw_time.strip().split(":")))
                            if len(parts) == 2:
                                qt_fallback = QTime(parts[0], parts[1])
                                if qt_fallback.isValid():
                                    return qt_fallback.toString("hh:mm AP")
                        except:
                            pass
                        return raw_time

                    card = QFrame()
                    card.setStyleSheet("background:#37474F;border-radius:10px;")
                    shadow_effect = QGraphicsDropShadowEffect()
                    shadow_effect.setOffset(0, 4)
                    shadow_effect.setBlurRadius(10)
                    semi_transparent_white = QColor(0, 0, 0, int(255 * 0.3))
                    shadow_effect.setColor(semi_transparent_white)
                    card.setGraphicsEffect(shadow_effect)

                    layout = QVBoxLayout(card)
                    layout.setContentsMargins(10, 8, 10, 8)

                    top = QHBoxLayout()
                    session_text = f" • {log['session']}" if log.get("session") else ""

                    start_time_str = format_time_am_pm(log.get('start', ''))
                    end_time_str = format_time_am_pm(log.get('end', ''))

                    duration_raw = log.get("duration", "")
                    try:
                        h, m = map(int, duration_raw.split(":"))
                        duration_fmt = f"{h} hour{'s' if h != 1 else ''}" + (f" {m} minute{'s' if m != 1 else ''}" if m > 0 else "")
                    except:
                        duration_fmt = duration_raw

                    lbl = QLabel(f"{start_time_str}–{end_time_str} ({duration_fmt}) • {log['quality']}{session_text}")
                    lbl.setStyleSheet("color:#00BCD4;font-weight:bold;font-size:12pt;")
                    top.addWidget(lbl)
                    top.addStretch()

                    edit_btn = QPushButton()
                    edit_btn.setIcon(QIcon(resource_path("assets/edit.png")))
                    edit_btn.setFixedSize(28, 28)
                    edit_btn.setIconSize(QSize(20, 20))
                    edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    edit_btn.setStyleSheet("""
                        QPushButton {
                            background: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(255,255,255,0.1);
                            border-radius: 6px;
                        }
                    """)
                    edit_btn.clicked.connect(lambda: open_sleep_log_dialog(log))

                    del_btn = QPushButton()
                    del_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                    del_btn.setFixedSize(28, 28)
                    del_btn.setIconSize(QSize(20, 20))
                    del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    del_btn.setStyleSheet("""
                        QPushButton {
                            background: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(255,255,255,0.1);
                            border-radius: 6px;
                        }
                    """)
                    del_btn.clicked.connect(lambda: delete_sleep_log(log))

                    btn_layout = QHBoxLayout()
                    btn_layout.setSpacing(4)
                    btn_layout.addWidget(edit_btn)
                    btn_layout.addWidget(del_btn)
                    top.addLayout(btn_layout)

                    layout.addLayout(top)

                    sub = QLabel(f"Date: {log.get('date', '')} {log.get('time', '')}")
                    sub.setStyleSheet("color:#aaa; font-size:10pt;")
                    layout.addWidget(sub)

                    if log.get("note"):
                        note = QLabel(f"Note: {log['note']}")
                        note.setStyleSheet("color:#ccc; font-size:9pt;")
                        note.setWordWrap(True)
                        layout.addWidget(note)

                    return card


                def refresh_sleep_logs(filtered=None):
                    logs = filtered if filtered is not None else me.get("sleep_log", [])
                    total = max(1, (len(logs) + logs_per_page - 1) // logs_per_page)
                    current_page[0] = min(current_page[0], total)

                    for i in reversed(range(container_layout.count())):
                        w = container_layout.itemAt(i).widget()
                        if w:
                            w.setParent(None)

                    if not logs:
                        lbl = QLabel("No sleep logs.")
                        lbl.setStyleSheet("color:#888;")
                        container_layout.addWidget(lbl)
                    else:
                        s = (current_page[0] - 1) * logs_per_page
                        e = s + logs_per_page
                        for log in list(reversed(logs))[s:e]:
                            container_layout.addWidget(sleep_log_card(log))

                    page_input.setText(str(current_page[0]))
                    total_pages_lbl.setText(f"/ {total}")
                    prev_btn.setEnabled(current_page[0] > 1)
                    next_btn.setEnabled(current_page[0] < total)

                prev_btn.clicked.connect(lambda: [current_page.__setitem__(0, current_page[0] - 1), refresh_sleep_logs(last_filtered[0] if last_filtered[0] else None)])
                next_btn.clicked.connect(lambda: [current_page.__setitem__(0, current_page[0] + 1), refresh_sleep_logs(last_filtered[0] if last_filtered[0] else None)])
                page_input.returnPressed.connect(lambda: [current_page.__setitem__(0, int(page_input.text()) if page_input.text().isdigit() else 1), refresh_sleep_logs(last_filtered[0] if last_filtered[0] else None)])


                refresh_sleep_logs()

                add_btn = QPushButton("Add Sleep Log")
                add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                add_btn.setFixedSize(200, 40)
                add_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #5C6BC0;
                        color: white;
                        padding: 10px 24px;
                        border-radius: 8px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3F51B5;
                    }
                """)
                add_btn.clicked.connect(lambda: open_sleep_log_dialog())
                layout.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

                return tab

            def create_sleep_summary_tab() -> QWidget:

                def create_weekly_sleep_chart(sleep_logs):
                    today = datetime.now().date()

                    def week_start(date):
                        days_since_sunday = (date.weekday() + 1) % 7
                        return date - timedelta(days=days_since_sunday)

                    current_week_start = week_start(today)
                    current = current_week_start
                    animation_running = False
                    selected_session = ["All"]

                    container = QWidget()
                    container.setStyleSheet("background-color: transparent;border;none")
                    layout = QVBoxLayout(container)
                    layout.setContentsMargins(8, 8, 8, 8)

                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")


                    session_selector = QComboBox()
                    session_selector.addItems(["All", "Nap", "Core Sleep", "Restorative", "Other"])
                    session_selector.setMinimumWidth(80)
                    session_selector.setStyleSheet(f"""
                        QComboBox {{
                            background-color: #444;
                            color: white;
                            border-radius: 6px;
                            padding: 6px 10px;
                            font-weight: bold;
                        }}
                        QComboBox::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 28px;
                            border-left-width: 0px;
                            border-left-color: darkgray;
                            border-left-style: solid;
                            border-top-right-radius: 6px;
                            border-bottom-right-radius: 6px;
                        }}
                        QComboBox::down-arrow {{
                            image: url({down_arrow_path});
                            width: 16px;
                            height: 16px;
                        }}
                        QComboBox QAbstractItemView {{
                            background-color: #333;
                            color: white;
                            selection-background-color: #555;
                            padding: 4px;
                            min-width: 200px;
                        }}
                    """)
                    layout.addWidget(session_selector, alignment=Qt.AlignmentFlag.AlignRight)


                    week_range_label = QLabel("")
                    week_range_label.setStyleSheet("color:#aaa; font-size:9pt;border:none")
                    layout.addWidget(week_range_label, alignment=Qt.AlignmentFlag.AlignCenter)

                    canvas_container = QWidget()
                    canvas_container.setFixedSize(420, 150)

                    h_layout = QHBoxLayout()
                    h_layout.addStretch()
                    h_layout.addWidget(canvas_container)
                    h_layout.addStretch()
                    layout.addLayout(h_layout)


                    fig1 = Figure(figsize=(4.2, 1.5), dpi=100, facecolor="#2E2E3E")
                    ax1 = fig1.add_subplot(111)
                    fig1.subplots_adjust(left=0.15, bottom=0.4)

                    fig2 = Figure(figsize=(4.2, 1.5), dpi=100, facecolor="#2E2E3E")
                    ax2 = fig2.add_subplot(111)
                    fig2.subplots_adjust(left=0.15, bottom=0.4)

                    canvas1 = FigureCanvas(fig1)
                    canvas1.setParent(canvas_container)
                    canvas1.resize(canvas_container.size())
                    canvas1.move(0, 0)

                    canvas2 = FigureCanvas(fig2)
                    canvas2.setParent(canvas_container)
                    canvas2.resize(canvas_container.size())
                    canvas2.move(canvas_container.width(), 0)

                    def parse_duration(dur_str):
                        try:

                            parts = dur_str.strip().split(":")
                            if len(parts) == 2:
                                h = int(parts[0])
                                m = int(parts[1])
                                if m < 0 or m >= 60:

                                    m = max(0, min(m, 59))
                                return h + m / 60
                        except Exception:
                            pass
                        return 0

                    def get_total_duration_for_date(logs, date, session_filter):
                        date_str = date.strftime("%Y-%m-%d")
                        total = 0
                        for log in logs:
                            if log.get("date") == date_str:
                                if session_filter != "All" and log.get("session") != session_filter:
                                    continue
                                total += parse_duration(log.get("duration", "0:0"))
                        return total

                    def update_week_label(start_date):
                        end_date = start_date + timedelta(days=6)
                        week_range_label.setText(f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}")

                    def draw_week(ax, canvas, start_date):
                        ax.clear()
                        ax.set_facecolor("#2E2E3E")
                        ax.grid(True, linestyle="--", alpha=0.2, color="#666")
                        ax.tick_params(colors='white', labelsize=8)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_color('white')
                        ax.spines['left'].set_color('white')

                        dates = [start_date + timedelta(days=i) for i in range(7)]
                        labels = [d.strftime("%a") for d in dates]

                        session_filter = selected_session[0]
                        durations = [get_total_duration_for_date(sleep_logs, d, session_filter) for d in dates]

                        max_val = max(durations) if durations else 1
                        y_max = max_val * 1.3 if max_val > 0 else 1
                        ax.set_ylim(0, y_max)

                        bar_colors = ["#007AFF" if d <= today else "#444444" for d in dates]
                        bars = ax.bar(range(7), durations, color=bar_colors, alpha=0.9, width=0.5)

                        ax.set_xticks(range(7))
                        ax.set_xticklabels(labels, color="white")
                        ax.set_ylabel("Hours slept", color="white")

                        for i, val in enumerate(durations):
                            hours = int(val)
                            minutes = int(round((val - hours) * 60))
                            ax.text(i, val + y_max * 0.03, f"{hours}h {minutes}m", ha='center', color='white', fontsize=7)

                        canvas.draw()

                    draw_week(ax1, canvas1, current)
                    update_week_label(current)

                    def slide_to_week(new_week, direction=1):
                        nonlocal current, animation_running, ax1, ax2, canvas1, canvas2
                        if animation_running:
                            return
                        animation_running = True

                        draw_week(ax2, canvas2, new_week)

                        width = canvas1.width()
                        canvas1.move(0, 0)
                        canvas2.move(direction * width, 0)

                        anim1 = QPropertyAnimation(canvas1, b"pos", container)
                        anim1.setDuration(400)
                        anim1.setStartValue(canvas1.pos())
                        anim1.setEndValue(QPoint(-direction * width, 0))
                        anim1.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        anim2 = QPropertyAnimation(canvas2, b"pos", container)
                        anim2.setDuration(400)
                        anim2.setStartValue(canvas2.pos())
                        anim2.setEndValue(QPoint(0, 0))
                        anim2.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        def on_anim_finished():
                            nonlocal ax1, ax2, canvas1, canvas2, current, animation_running
                            current = new_week
                            update_week_label(current)
                            ax1.clear()
                            canvas1.move(width, 0)
                            ax1, ax2 = ax2, ax1
                            canvas1, canvas2 = canvas2, canvas1
                            animation_running = False

                        anim2.finished.connect(on_anim_finished)

                        anim1.start()
                        anim2.start()

                        container._anim1 = anim1
                        container._anim2 = anim2

                    btn_prev = QPushButton()
                    btn_prev.setIcon(QIcon(resource_path("assets/prev.png")))
                    btn_prev.setIconSize(QSize(24, 24))
                    btn_prev.setFixedSize(32, 32)
                    btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn_prev.setStyleSheet("""
                        QPushButton {
                            border: none;
                            background: transparent;
                            border-radius: 6px;
                        }
                        QPushButton:hover {
                            background-color: rgba(36, 146, 255, 0.5);
                        }
                    """)
                    btn_next = QPushButton()
                    btn_next.setIcon(QIcon(resource_path("assets/next.png")))
                    btn_next.setIconSize(QSize(24, 24))
                    btn_next.setFixedSize(32, 32)
                    btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn_next.setStyleSheet("""
                        QPushButton {
                            border: none;
                            background: transparent;
                            border-radius: 6px;
                        }
                        QPushButton:hover {
                            background-color: rgba(36, 146, 255, 0.5);
                        }
                    """)
                    nav_layout = QHBoxLayout()
                    nav_layout.setContentsMargins(0, 4, 0, 0)
                    nav_layout.addWidget(btn_prev)
                    nav_layout.addStretch()
                    nav_layout.addWidget(btn_next)
                    layout.addLayout(nav_layout)

                    btn_prev.clicked.connect(lambda: slide_to_week(current - timedelta(days=7), direction=-1))
                    btn_next.clicked.connect(lambda: slide_to_week(current + timedelta(days=7), direction=1))

                    session_selector.currentTextChanged.connect(lambda text: [
                        selected_session.__setitem__(0, text),
                        draw_week(ax1, canvas1, current)
                    ])

                    container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    return container

                def create_weekly_sleep_averages(sleep_logs):
                    today = datetime.now().date()

                    def week_start(date):
                        days_since_sunday = (date.weekday() + 1) % 7
                        return date - timedelta(days=days_since_sunday)

                    current = week_start(today)
                    animation_running = False
                    selected_session = ["All"]

                    container = QWidget()
                    container.setStyleSheet("background-color: transparent;border;none")
                    layout = QVBoxLayout(container)
                    layout.setContentsMargins(8, 8, 8, 8)

                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    session_selector = QComboBox()
                    session_selector.addItems(["All", "Nap", "Core Sleep", "Restorative", "Other"])
                    session_selector.setMinimumWidth(80)
                    session_selector.setStyleSheet(f"""
                        QComboBox {{
                            background-color: #444;
                            color: white;
                            border-radius: 6px;
                            padding: 6px 10px;
                            font-weight: bold;
                            border: none;
                        }}
                        QComboBox::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 28px;
                            border-top-right-radius: 6px;
                            border-bottom-right-radius: 6px;
                        }}
                        QComboBox::down-arrow {{
                            image: url({down_arrow_path});
                            width: 16px;
                            height: 16px;
                        }}
                        QComboBox QAbstractItemView {{
                            background-color: #333;
                            color: white;
                            selection-background-color: #555;
                            padding: 4px;
                            min-width: 200px;
                            border: none;
                        }}
                    """)
                    top_row = QHBoxLayout()
                    top_row.setContentsMargins(0, 0, 0, 0)
                    top_row.setSpacing(8)


                    top_row.addStretch()


                    top_row.addSpacing(100)


                    week_range_label = QLabel("")
                    week_range_label.setStyleSheet("color:#aaa; font-size:9pt; padding: 4px; border: none;")
                    week_range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    week_range_label.setFixedWidth(150)
                    top_row.addWidget(week_range_label)


                    top_row.addStretch()


                    top_row.addWidget(session_selector)


                    layout.addLayout(top_row)



                    week_widgets = [QWidget(), QWidget()]
                    for w in week_widgets:
                        w.setFixedSize(420, 180)
                        w.setStyleSheet("background-color: transparent;")
                        w.setLayout(QVBoxLayout())

                    container_stack = QWidget()
                    container_stack.setFixedSize(420, 180)
                    container_stack.setStyleSheet("background-color: transparent;")
                    layout.addWidget(container_stack, alignment=Qt.AlignmentFlag.AlignHCenter)

                    week_widgets[0].setParent(container_stack)
                    week_widgets[1].setParent(container_stack)
                    week_widgets[0].move(0, 0)
                    week_widgets[1].move(container_stack.width(), 0)

                    def parse_time_str(t):
                        try:
                            return QTime.fromString(t, "HH:mm")
                        except:
                            return None

                    def parse_duration(dur_str):
                        try:
                            h, m = map(int, dur_str.split(":"))
                            return h + m / 60
                        except:
                            return 0

                    def format_avg_time(times):
                        if not times:
                            return "--"
                        total_minutes = sum(t.hour() * 60 + t.minute() for t in times)
                        avg_minutes = total_minutes // len(times)
                        return QTime(avg_minutes // 60, avg_minutes % 60).toString("hh:mm AP")

                    def clear_layout(layout):
                        while layout.count():
                            item = layout.takeAt(0)
                            if item.widget():
                                item.widget().deleteLater()
                            elif item.layout():
                                clear_layout(item.layout())

                    def clear_widget(widget):
                        layout = widget.layout()
                        if layout is None:
                            return
                        clear_layout(layout)

                    def update_week_data(widget, start_date):

                        clear_widget(widget)
                        layout = widget.layout()
                        dates = [start_date + timedelta(days=i) for i in range(7)]
                        session_filter = selected_session[0]

                        durations, bedtimes, waketimes, qualities = [], [], [], []

                        for d in dates:
                            date_str = d.strftime("%Y-%m-%d")
                            for log in sleep_logs:
                                if log.get("date") != date_str:
                                    continue
                                if session_filter != "All" and log.get("session") != session_filter:
                                    continue

                                durations.append(parse_duration(log.get("duration", "0:0")))
                                t1 = parse_time_str(log.get("start"))
                                t2 = parse_time_str(log.get("end"))
                                if t1: bedtimes.append(t1)
                                if t2: waketimes.append(t2)
                                if log.get("quality"): qualities.append(log["quality"])

                        avg_dur = sum(durations) / len(durations) if durations else 0
                        avg_quality = max(set(qualities), key=qualities.count) if qualities else "--"
                        avg_bed = format_avg_time(bedtimes)
                        avg_wake = format_avg_time(waketimes)

                        entries = [
                            ("Sleep Duration", f"{int(avg_dur)}h {int((avg_dur % 1) * 60)}m", "clock"),
                            ("Bedtime", avg_bed, "moon"),
                            ("Wake Time", avg_wake, "sun"),
                            ("Quality", avg_quality, "star")
                        ]

                        grid = QGridLayout()
                        grid.setHorizontalSpacing(16)
                        grid.setVerticalSpacing(0)

                        for idx, (label, value, icon) in enumerate(entries):
                            icon_label = QLabel()
                            icon_label.setPixmap(QPixmap(resource_path(f"assets/{icon}.png")))
                            icon_label.setFixedSize(24, 24)
                            icon_label.setScaledContents(True)
                            icon_label.setStyleSheet("border: none;")

                            val_lbl = QLabel(value)
                            val_lbl.setStyleSheet("color:white; font-weight:bold; font-size:11pt; border: none;")
                            val_lbl.setMinimumHeight(26)

                            label_lbl = QLabel(label)
                            label_lbl.setStyleSheet("color:#aaa; font-size:9pt; border: none;")
                            label_lbl.setMinimumHeight(20)

                            col = QVBoxLayout()
                            col.setSpacing(2)
                            col.addWidget(label_lbl)
                            col.addWidget(val_lbl)

                            row_layout = QHBoxLayout()
                            row_layout.addWidget(icon_label)
                            row_layout.addSpacing(8)
                            row_layout.addLayout(col)
                            row_layout.addStretch()

                            container_widget = QWidget()
                            container_widget.setLayout(row_layout)


                            grid.addWidget(container_widget, idx // 2, idx % 2)

                        container_widget = QWidget()
                        container_widget.setLayout(grid)
                        layout.addWidget(container_widget, alignment=Qt.AlignmentFlag.AlignHCenter)



                    update_week_data(week_widgets[0], current)

                    def update_week_label(start):
                        end = start + timedelta(days=6)
                        week_range_label.setText(f"{start.strftime('%b %d')} - {end.strftime('%b %d')}, {start.year}")

                    update_week_label(current)

                    def slide_to_week(new_week, direction=1):
                        nonlocal current, animation_running
                        if animation_running:
                            return
                        animation_running = True

                        update_week_data(week_widgets[1], new_week)

                        width = container_stack.width()
                        week_widgets[0].move(0, 0)
                        week_widgets[1].move(direction * width, 0)

                        anim1 = QPropertyAnimation(week_widgets[0], b"pos", container_stack)
                        anim1.setDuration(400)
                        anim1.setStartValue(week_widgets[0].pos())
                        anim1.setEndValue(QPoint(-direction * width, 0))
                        anim1.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        anim2 = QPropertyAnimation(week_widgets[1], b"pos", container_stack)
                        anim2.setDuration(400)
                        anim2.setStartValue(week_widgets[1].pos())
                        anim2.setEndValue(QPoint(0, 0))
                        anim2.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        def finish():
                            nonlocal animation_running, current
                            current = new_week
                            update_week_label(current)
                            week_widgets[0], week_widgets[1] = week_widgets[1], week_widgets[0]
                            animation_running = False

                        anim2.finished.connect(finish)
                        anim1.start()
                        anim2.start()
                        container_stack._a1 = anim1
                        container_stack._a2 = anim2

                    nav = QHBoxLayout()
                    btn_prev = QPushButton()
                    btn_prev.setIcon(QIcon(resource_path("assets/prev.png")))
                    btn_prev.setIconSize(QSize(24, 24))
                    btn_prev.setFixedSize(32, 32)
                    btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn_prev.setStyleSheet("""
                        QPushButton {
                            border: none;
                            background: transparent;
                            border-radius: 6px;
                        }
                        QPushButton:hover {
                            background-color: rgba(36, 146, 255, 0.5);
                        }
                    """)
                    btn_next = QPushButton()
                    btn_next.setIcon(QIcon(resource_path("assets/next.png")))
                    btn_next.setIconSize(QSize(24, 24))
                    btn_next.setFixedSize(32, 32)
                    btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn_next.setStyleSheet("""
                        QPushButton {
                            border: none;
                            background: transparent;
                            border-radius: 6px;
                        }
                        QPushButton:hover {
                            background-color: rgba(36, 146, 255, 0.5);
                        }
                    """)
                    btn_prev.clicked.connect(lambda: slide_to_week(current - timedelta(days=7), -1))
                    btn_next.clicked.connect(lambda: slide_to_week(current + timedelta(days=7), 1))

                    nav.addWidget(btn_prev)
                    nav.addStretch()
                    nav.addWidget(btn_next)
                    layout.addLayout(nav)

                    session_selector.currentTextChanged.connect(lambda text: [
                        selected_session.__setitem__(0, text),
                        update_week_data(week_widgets[0], current)
                    ])

                    return container


                def create_weekly_quality_chart(sleep_logs):
                    today = datetime.now().date()

                    def week_start(date):
                        days_since_sunday = (date.weekday() + 1) % 7
                        return date - timedelta(days=days_since_sunday)

                    current_week_start = week_start(today)
                    current = current_week_start
                    animation_running = False
                    selected_session = ["All"]

                    container = QWidget()
                    container.setStyleSheet("background-color: transparent; border: none")
                    layout = QVBoxLayout(container)
                    layout.setContentsMargins(8, 8, 8, 8)


                    session_selector = QComboBox()
                    session_selector.addItems(["All", "Nap", "Core Sleep", "Restorative", "Other"])
                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    session_selector.setMinimumWidth(80)
                    session_selector.setStyleSheet(f"""
                        QComboBox {{
                            background-color: #444;
                            color: white;
                            border-radius: 6px;
                            padding: 6px 10px;
                            font-weight: bold;
                            border: none;
                        }}
                        QComboBox::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 28px;
                            border-top-right-radius: 6px;
                            border-bottom-right-radius: 6px;
                        }}
                        QComboBox::down-arrow {{
                            image: url({down_arrow_path});
                            width: 16px;
                            height: 16px;
                        }}
                        QComboBox QAbstractItemView {{
                            background-color: #333;
                            color: white;
                            selection-background-color: #555;
                            padding: 4px;
                            min-width: 200px;
                            border: none;
                        }}
                    """)
                    layout.addWidget(session_selector, alignment=Qt.AlignmentFlag.AlignRight)

                    week_range_label = QLabel("")
                    week_range_label.setStyleSheet("color:#aaa; font-size:9pt; border:none")
                    layout.addWidget(week_range_label, alignment=Qt.AlignmentFlag.AlignCenter)

                    canvas_container = QWidget()
                    canvas_container.setFixedSize(420, 180)
                    canvas_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

                    h_layout = QHBoxLayout()
                    h_layout.setContentsMargins(0, 0, 0, 0)
                    h_layout.addStretch()
                    h_layout.addWidget(canvas_container)
                    h_layout.addStretch()
                    layout.addLayout(h_layout)


                    fig1 = Figure(figsize=(4.2, 1.8), dpi=100, facecolor="#2E2E3E")
                    ax1 = fig1.add_subplot(111)
                    fig1.subplots_adjust(left=0.18, bottom=0.3)

                    fig2 = Figure(figsize=(4.2, 1.8), dpi=100, facecolor="#2E2E3E")
                    ax2 = fig2.add_subplot(111)
                    fig2.subplots_adjust(left=0.18, bottom=0.3)

                    canvas1 = FigureCanvas(fig1)
                    canvas1.setParent(canvas_container)
                    canvas1.resize(canvas_container.size())
                    canvas1.move(0, 0)
                    canvas1.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

                    canvas2 = FigureCanvas(fig2)
                    canvas2.setParent(canvas_container)
                    canvas2.resize(canvas_container.size())
                    canvas2.move(canvas_container.width(), 0)
                    canvas2.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)



                    quality_map = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}
                    color_map = {"Excellent": "#4CAF50", "Good": "#2196F3", "Fair": "#FFC107", "Poor": "#F44336"}

                    def get_quality_score(quality_str):
                        return quality_map.get(quality_str, 0)

                    def get_quality_for_date(logs, date, session_filter):
                        date_str = date.strftime("%Y-%m-%d")

                        qualities = []
                        for log in logs:
                            if log.get("date") == date_str:
                                if session_filter != "All" and log.get("session") != session_filter:
                                    continue
                                q_str = log.get("quality", "Poor")
                                q_score = get_quality_score(q_str)
                                if q_score > 0:
                                    qualities.append(q_score)
                        if qualities:
                            return sum(qualities) / len(qualities)
                        else:
                            return 0

                    def get_quality_label_for_score(score):

                        if score >= 3.5:
                            return "Excellent"
                        elif score >= 2.5:
                            return "Good"
                        elif score >= 1.5:
                            return "Fair"
                        elif score > 0:
                            return "Poor"
                        return "No Data"

                    def update_week_label(start_date):
                        end_date = start_date + timedelta(days=6)
                        week_range_label.setText(f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}")

                    def draw_week(ax, canvas, start_date):
                        ax.clear()
                        ax.set_facecolor("#2E2E3E")
                        ax.grid(True, linestyle="--", alpha=0.2, color="#666")
                        ax.tick_params(colors='white', labelsize=9)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_color('white')
                        ax.spines['left'].set_color('white')

                        dates = [start_date + timedelta(days=i) for i in range(7)]
                        labels = [d.strftime("%a") for d in dates]

                        session_filter = selected_session[0]
                        qualities = [get_quality_for_date(sleep_logs, d, session_filter) for d in dates]

                        max_val = 4
                        y_min, y_max = 0, 4.5
                        ax.set_ylim(y_min, y_max)

                        ax.set_xticks(range(7))
                        ax.set_xticklabels(labels, color="white")


                        x_vals = list(range(7))
                        ax.plot(x_vals, qualities, color="#00BCD4", linewidth=2)


                        ax.fill_between(x_vals, qualities, y_min, color="#00BCD4", alpha=0.25)

                        for i, score in enumerate(qualities):
                            if score == 0:
                                continue
                            label = get_quality_label_for_score(score)
                            rounded_label = get_quality_label_for_score(round(score))
                            dot_color = color_map.get(rounded_label, "#888888")
                            dot = ax.scatter(i, score, color=dot_color, s=50, edgecolors='none', linewidth=0, zorder=5)

                            ax.text(i, score + 0.15, label, color="white", fontsize=8, ha='center')
                        ax.set_yticks([1, 2, 3, 4])
                        ax.set_yticklabels(["Poor", "Fair", "Good", "Excellent"], color="white")

                        canvas.draw()

                    draw_week(ax1, canvas1, current)
                    update_week_label(current)

                    def slide_to_week(new_week, direction=1):
                        nonlocal current, animation_running, ax1, ax2, canvas1, canvas2
                        if animation_running:
                            return
                        animation_running = True

                        draw_week(ax2, canvas2, new_week)

                        width = canvas1.width()
                        canvas1.move(0, 0)
                        canvas2.move(direction * width, 0)

                        anim1 = QPropertyAnimation(canvas1, b"pos", container)
                        anim1.setDuration(400)
                        anim1.setStartValue(canvas1.pos())
                        anim1.setEndValue(QPoint(-direction * width, 0))
                        anim1.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        anim2 = QPropertyAnimation(canvas2, b"pos", container)
                        anim2.setDuration(400)
                        anim2.setStartValue(canvas2.pos())
                        anim2.setEndValue(QPoint(0, 0))
                        anim2.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        def on_anim_finished():
                            nonlocal ax1, ax2, canvas1, canvas2, current, animation_running
                            current = new_week
                            update_week_label(current)
                            ax1.clear()
                            canvas1.move(width, 0)
                            ax1, ax2 = ax2, ax1
                            canvas1, canvas2 = canvas2, canvas1
                            animation_running = False

                        anim2.finished.connect(on_anim_finished)

                        anim1.start()
                        anim2.start()

                        container._anim1 = anim1
                        container._anim2 = anim2

                    btn_prev = QPushButton()
                    btn_prev.setIcon(QIcon(resource_path("assets/prev.png")))
                    btn_prev.setIconSize(QSize(24, 24))
                    btn_prev.setFixedSize(32, 32)
                    btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn_prev.setStyleSheet("""
                        QPushButton {
                            border: none;
                            background: transparent;
                            border-radius: 6px;
                        }
                        QPushButton:hover {
                            background-color: rgba(36, 146, 255, 0.5);
                        }
                    """)

                    btn_next = QPushButton()
                    btn_next.setIcon(QIcon(resource_path("assets/next.png")))
                    btn_next.setIconSize(QSize(24, 24))
                    btn_next.setFixedSize(32, 32)
                    btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn_next.setStyleSheet("""
                        QPushButton {
                            border: none;
                            background: transparent;
                            border-radius: 6px;
                        }
                        QPushButton:hover {
                            background-color: rgba(36, 146, 255, 0.5);
                        }
                    """)

                    nav_layout = QHBoxLayout()
                    nav_layout.setContentsMargins(0, 4, 0, 0)
                    nav_layout.addWidget(btn_prev)
                    nav_layout.addStretch()
                    nav_layout.addWidget(btn_next)
                    layout.addLayout(nav_layout)

                    btn_prev.clicked.connect(lambda: slide_to_week(current - timedelta(days=7), direction=-1))
                    btn_next.clicked.connect(lambda: slide_to_week(current + timedelta(days=7), direction=1))

                    session_selector.currentTextChanged.connect(lambda text: [
                        selected_session.__setitem__(0, text),
                        draw_week(ax1, canvas1, current)
                    ])

                    container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    return container

                def create_sleep_consistency_heatmap(sleep_logs):

                    def parse_duration_to_hours(duration_str):
                        try:
                            h, m = map(int, duration_str.split(":"))
                            return h + m / 60
                        except Exception:
                            return 0

                    def parse_onset_minutes(start_str):
                        try:
                            t = QTime.fromString(start_str, "HH:mm")
                            minutes = t.hour() * 60 + t.minute()
                            if minutes < 720:
                                minutes += 1440
                            return minutes
                        except Exception:
                            return 0


                    container = QWidget()
                    container.setStyleSheet("background-color: transparent;")
                    main_layout = QVBoxLayout(container)
                    main_layout.setContentsMargins(8, 8, 8, 8)
                    main_layout.setSpacing(8)


                    if sleep_logs:
                        dates = sorted([datetime.strptime(log["date"], "%Y-%m-%d").date() for log in sleep_logs])
                        label = QLabel(f"{dates[0].strftime('%b %d, %Y')} – {dates[-1].strftime('%b %d, %Y')}")
                        label.setStyleSheet("color:#aaa; font-size:9pt; border:none;")
                        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        main_layout.addWidget(label)


                    fig = Figure(figsize=(4.1, 2.8), dpi=100, facecolor="#2E2E3E")
                    ax = fig.add_subplot(111)

                    fig.subplots_adjust(left=0.13, right=0.88, top=0.9, bottom=0.3)
                    canvas = FigureCanvas(fig)


                    h_layout = QHBoxLayout()
                    h_layout.addStretch()
                    h_layout.addWidget(canvas)
                    h_layout.addStretch()
                    main_layout.addLayout(h_layout)


                    time_bins = np.arange(720, 2161, 15)
                    dur_bins = np.arange(0, 12.25, 0.25)
                    heatmap = np.zeros((len(dur_bins) - 1, len(time_bins) - 1), dtype=int)

                    for log in sleep_logs:
                        onset = parse_onset_minutes(log.get("start", "00:00"))
                        duration = parse_duration_to_hours(log.get("duration", "0:0"))
                        time_idx = np.digitize(onset, time_bins) - 1
                        dur_idx = np.digitize(duration, dur_bins) - 1
                        if 0 <= time_idx < heatmap.shape[1] and 0 <= dur_idx < heatmap.shape[0]:
                            heatmap[dur_idx, time_idx] += 1


                    cmap = colormaps.get_cmap("turbo")
                    nonzero_max = np.max(heatmap) if np.max(heatmap) > 0 else 1
                    norm = LogNorm(vmin=1, vmax=nonzero_max)

                    im = ax.imshow(
                        heatmap,
                        aspect='auto',
                        cmap=cmap,
                        norm=norm,
                        origin='lower',
                        interpolation='bilinear',
                        extent=[time_bins[0], time_bins[-1], dur_bins[0], dur_bins[-1]]
                    )


                    x_ticks = np.arange(720, 2161, 60)
                    x_labels = [
                        (datetime(2000, 1, 1) + timedelta(minutes=int(m % 1440))).strftime("%I %p").lstrip("0")
                        for m in x_ticks
                    ]
                    x_labels[-1] = ""
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45, ha='right', color='white', fontsize=7)


                    ax.set_yticks(np.arange(0, 13, 2))
                    ax.set_ylabel("Sleep Duration (h)", color='white', fontsize=9)
                    ax.set_title("Sleep Timing Heatmap", color='white', fontsize=10)
                    ax.tick_params(colors='white', labelsize=6)
                    ax.set_facecolor("#2E2E3E")


                    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
                    cbar.ax.tick_params(labelsize=6, color='white', labelcolor='white')
                    cbar.set_label("log₁₀(Frequency)", color='white', fontsize=8)
                    cbar.outline.set_edgecolor('white')

                    canvas.draw()
                    container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    return container

                def create_sleep_pattern_insights_card(sleep_logs):


                    def week_start(date):
                        days_since_sunday = (date.weekday() + 1) % 7
                        return date - timedelta(days=days_since_sunday)

                    def get_logs_by_week(base_date):
                        start = week_start(base_date)
                        end = start + timedelta(days=6)
                        label = f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"
                        logs = [log for log in sleep_logs if start <= datetime.strptime(log["date"], "%Y-%m-%d").date() <= end]
                        return logs, label

                    def create_insight_label(text, icon_path):
                        row = QHBoxLayout()
                        icon = QLabel()
                        if os.path.exists(icon_path):
                            icon.setPixmap(QIcon(resource_path(icon_path)).pixmap(18, 18))
                        icon.setFixedSize(22, 22)
                        label = QLabel(text)
                        label.setStyleSheet("color: white; font-size: 8pt;")
                        row.addWidget(icon)
                        row.addSpacing(8)
                        row.addWidget(label)
                        row.addStretch()
                        return row

                    def build_insights_layout(week_logs):
                        insights_box = QVBoxLayout()
                        insights_box.setSpacing(6)

                        def make_insight(text, icon):
                            insights_box.addLayout(create_insight_label(text, f"assets/{icon}"))

                        durations = []
                        start_times = []
                        for log in week_logs:
                            dur = log.get("duration", "0:0")
                            try:
                                h, m = map(int, dur.split(":"))
                                durations.append(h + m / 60)
                            except:
                                pass
                            start = log.get("start", "00:00")
                            try:
                                h, m = map(int, start.split(":"))
                                start_times.append(h + m / 60)
                            except:
                                pass

                        if durations:
                            avg = sum(durations) / len(durations)
                            if avg >= 8:
                                make_insight(f"Great! Avg sleep duration: {avg:.1f}h", "check.png")
                            elif avg >= 6:
                                make_insight(f"Decent sleep avg: {avg:.1f}h", "neutral.png")
                            else:
                                make_insight(f"Low sleep avg: {avg:.1f}h", "warning.png")
                        if start_times:
                            early = sum(1 for s in start_times if s < 24)
                            late = len(start_times) - early
                            if late > early:
                                make_insight("You tend to sleep late this week", "latesleep.png")
                            else:
                                make_insight("Consistent early bedtime pattern", "consistentsleep.png")
                        if not durations and not start_times:
                            make_insight("No sleep data for this week", "info.png")

                        return insights_box


                    container = QFrame()
                    container.setStyleSheet("background-color: #2E2E3E; border-radius: 12px;")
                    layout = QVBoxLayout(container)
                    layout.setContentsMargins(8, 8, 8, 8)
                    layout.setSpacing(12)


                    title = QLabel("Sleep Pattern Insights")
                    title.setStyleSheet("color: white; font-size: 10pt; font-weight: normal;")
                    layout.addWidget(title)


                    nav_row = QHBoxLayout()
                    week_label = QLabel("")
                    week_label.setStyleSheet("color: #aaa; font-size: 8pt; border:none;")
                    nav_row.addWidget(week_label)
                    nav_row.addStretch()

                    def make_nav_button(icon_path):
                        btn = QPushButton()
                        btn.setIcon(QIcon(icon_path))
                        btn.setIconSize(QSize(24, 24))
                        btn.setFixedSize(32, 32)
                        btn.setCursor(Qt.CursorShape.PointingHandCursor)
                        btn.setStyleSheet("""
                            QPushButton {
                                border: none;
                                background: transparent;
                                border-radius: 6px;
                            }
                            QPushButton:hover {
                                background-color: rgba(36, 146, 255, 0.5);
                            }
                        """)
                        return btn

                    btn_prev = make_nav_button("assets/prev.png")
                    btn_next = make_nav_button("assets/next.png")
                    nav_row.addWidget(btn_prev)
                    nav_row.addWidget(btn_next)
                    layout.addLayout(nav_row)


                    stacked = QStackedWidget()
                    layout.addWidget(stacked)

                    current_week = [datetime.now().date()]


                    fig1 = Figure(figsize=(4.6, 6), dpi=100, facecolor="#2E2E3E")
                    ax1 = fig1.add_subplot(211)
                    ax2 = fig1.add_subplot(212)
                    fig1.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.45, hspace=1.2)
                    ax1.set_facecolor("#2E2E3E")
                    ax2.set_facecolor("#2E2E3E")

                    canvas1 = FigureCanvas(fig1)
                    canvas1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

                    fig2 = Figure(figsize=(4.5, 6), dpi=100, facecolor="#2E2E3E")
                    ax3 = fig2.add_subplot(211)
                    ax4 = fig2.add_subplot(212)
                    fig2.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.45, hspace=1.2)
                    ax3.set_facecolor("#2E2E3E")
                    ax4.set_facecolor("#2E2E3E")

                    canvas2 = FigureCanvas(fig2)
                    canvas2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



                    canvas_container = QWidget()
                    canvas_container.setFixedSize(450, 380)
                    canvas_container_layout = QVBoxLayout(canvas_container)
                    canvas_container_layout.setContentsMargins(0, 0, 0, 0)
                    canvas_container_layout.setSpacing(0)
                    canvas1.setParent(canvas_container)
                    canvas1.move(0, 0)
                    canvas2.setParent(canvas_container)
                    canvas2.move(canvas_container.width(), 0)
                    layout.addWidget(canvas_container, alignment=Qt.AlignmentFlag.AlignHCenter)

                    def build_pie_chart(ax, week_logs):
                        ax.clear()
                        session_counts = {}
                        for log in week_logs:
                            session = log.get("session", "Other")
                            session_counts[session] = session_counts.get(session, 0) + 1
                        if not session_counts:
                            session_counts = {"No Data": 1}

                        sessions, counts = zip(*session_counts.items())
                        cmap = colormaps.get_cmap("Set3")
                        colors = [to_hex(cmap(i / len(sessions))) for i in range(len(sessions))]

                        wedges, texts, autotexts = ax.pie(
                            counts,
                            labels=sessions,
                            colors=colors,
                            autopct='%1.0f%%',
                            startangle=90,
                            textprops={'color': 'white', 'fontsize': 8, 'weight': 'normal'},
                            wedgeprops={'width': 0.6, 'edgecolor': 'black', 'linewidth': 1.2}
                        )
                        ax.set_aspect('equal')
                        ax.set_facecolor("#2E2E3E")


                        ax.legend(wedges, sessions, title="Sessions", loc="upper right",
                                fontsize=8, title_fontsize=9,
                                facecolor='#3A3A4A', edgecolor='#555', labelcolor='white', framealpha=0.9,
                                bbox_to_anchor=(-0.9, 1))


                        ax.figure.subplots_adjust(left=0.07, right=0.93, top=0.95, bottom=0.05)

                    def build_bar_chart(ax, week_logs):
                        ax.clear()
                        dates = []
                        actual = []
                        needed = []
                        seen_dates = set()
                        for log in week_logs:
                            try:
                                d = datetime.strptime(log['date'], '%Y-%m-%d')
                                if d.strftime('%a') in seen_dates:
                                    continue
                                seen_dates.add(d.strftime('%a'))
                                dur = log.get('duration', '0:0')
                                h, m = map(int, dur.split(':'))
                                dates.append(d.strftime('%a'))
                                actual.append(h + m / 60)
                                needed.append(8)
                            except:
                                continue

                        max_val = max(max(actual, default=0), 8)
                        y_max = max_val * 1.3 if max_val > 0 else 10
                        ax.set_ylim(0, y_max)

                        ax.bar(np.arange(len(dates)) - 0.2, needed, width=0.4, color="#555555", label='Need')
                        ax.bar(np.arange(len(dates)) + 0.2, actual, width=0.4, color="#007AFF", label='Actual')

                        ax.set_xticks(np.arange(len(dates)))
                        ax.set_xticklabels(dates, color='white', fontsize=8, fontweight='normal')
                        ax.set_yticks([0, 4, 8])
                        ax.set_yticklabels(['0h', '4h', '8h'], color='white', fontsize=8, fontweight='normal')
                        ax.set_ylabel('Hours slept', color='white', fontsize=8, fontweight='normal')

                        ax.grid(True, linestyle="--", alpha=0.25, color="#666")
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_color('white')
                        ax.spines['left'].set_color('white')
                        ax.tick_params(colors='white', labelsize=8)


                        legend = ax.legend(facecolor='#3A3A4A', labelcolor='white',bbox_to_anchor=(1, 1.6), edgecolor='#555', fontsize=8, loc='upper right', framealpha=0.9)
                        legend.get_frame().set_linewidth(0.8)

                        for i, val in enumerate(actual):
                            hours = int(val)
                            minutes = int(round((val - hours) * 60))
                            ax.text(i, val + y_max * 0.03, f"{hours}h {minutes}m", ha='center', color='white', fontsize=8, weight='normal')

                        ax.figure.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.4)

                    def build_week_widget(week_logs):
                        page = QWidget()
                        vbox = QVBoxLayout(page)
                        vbox.setContentsMargins(0, 0, 0, 0)
                        vbox.setSpacing(50)

                        build_pie_chart(ax1, week_logs)
                        build_bar_chart(ax2, week_logs)
                        canvas1.draw()

                        insights_layout = build_insights_layout(week_logs)
                        vbox.addLayout(insights_layout)
                        vbox.addStretch()

                        return page

                    animation_running = False

                    def update_week_widget():
                        logs, label = get_logs_by_week(current_week[0])
                        week_label.setText(label)
                        page = build_week_widget(logs)

                        build_pie_chart(ax3, logs)
                        build_bar_chart(ax4, logs)
                        canvas2.draw()

                        while stacked.count() > 0:
                            w = stacked.widget(0)
                            stacked.removeWidget(w)
                            w.deleteLater()
                        stacked.addWidget(page)
                        stacked.setCurrentWidget(page)

                    def slide_week(offset):
                        nonlocal animation_running
                        if animation_running:
                            return
                        animation_running = True

                        new_week = current_week[0] + timedelta(days=7 * offset)
                        logs, label = get_logs_by_week(new_week)
                        week_label.setText(label)

                        build_pie_chart(ax3, logs)
                        build_bar_chart(ax4, logs)
                        canvas2.draw()

                        width = canvas1.width()
                        canvas1.move(0, 0)
                        canvas2.move(offset * width, 0)

                        anim1 = QPropertyAnimation(canvas1, b"pos", container)
                        anim1.setDuration(400)
                        anim1.setStartValue(canvas1.pos())
                        anim1.setEndValue(QPoint(-offset * width, 0))
                        anim1.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        anim2 = QPropertyAnimation(canvas2, b"pos", container)
                        anim2.setDuration(400)
                        anim2.setStartValue(canvas2.pos())
                        anim2.setEndValue(QPoint(0, 0))
                        anim2.setEasingCurve(QEasingCurve.Type.InOutQuad)

                        def on_anim_finished():
                            nonlocal animation_running, current_week, ax1, ax2, ax3, ax4, canvas1, canvas2
                            current_week[0] = new_week
                            animation_running = False
                            ax1, ax3 = ax3, ax1
                            ax2, ax4 = ax4, ax2
                            canvas1, canvas2 = canvas2, canvas1
                            canvas1.move(0, 0)
                            canvas2.move(container.width(), 0)

                            update_week_widget()

                        anim2.finished.connect(on_anim_finished)

                        anim1.start()
                        anim2.start()

                        container._anim1 = anim1
                        container._anim2 = anim2

                    btn_prev.clicked.connect(lambda: slide_week(-1))
                    btn_next.clicked.connect(lambda: slide_week(1))

                    update_week_widget()

                    container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    return container



                tab = QWidget()
                tab.setStyleSheet("background-color: #222; border-radius: 12px;")
                main_layout = QVBoxLayout(tab)
                main_layout.setContentsMargins(20, 20, 20, 20)
                main_layout.setSpacing(20)


                title = QLabel("Sleep Summary")
                title.setStyleSheet("""
                    color: #8AB6D6;
                    font-size: 24px;
                    font-weight: bold;
                    border: none;
                """)
                main_layout.addWidget(title)

                content_row = QHBoxLayout()
                content_row.setSpacing(16)

                grid_widget = QWidget()
                grid_layout = QGridLayout(grid_widget)
                grid_layout.setSpacing(16)
                grid_layout.setContentsMargins(0, 0, 0, 0)

                def create_card_with_chart(title_text, chart_widget):
                    card = QFrame()
                    card.setStyleSheet("""
                        QFrame {
                            background-color: #2E2E3E;
                            border-radius: 12px;
                            border: none;
                        }
                    """)
                    layout = QVBoxLayout(card)
                    layout.setContentsMargins(12, 12, 12, 12)
                    layout.setSpacing(8)

                    label = QLabel(title_text)
                    label.setStyleSheet("""
                        color: #fff;
                        font-size: 14pt;
                        font-weight: bold;
                        border: none;
                    """)

                    layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignTop)
                    layout.addWidget(chart_widget)
                    return card

                def create_placeholder_card(title_text):
                    card = QFrame()
                    card.setStyleSheet("""
                        QFrame {
                            background-color: #2E2E3E;
                            border-radius: 12px;
                            border: none;
                        }
                    """)
                    layout = QVBoxLayout(card)
                    layout.setContentsMargins(12, 12, 12, 12)
                    layout.setSpacing(8)

                    label = QLabel(title_text)
                    label.setStyleSheet("""
                        color: #fff;
                        font-size: 14pt;
                        font-weight: bold;
                        border: none;
                    """)

                    placeholder = QLabel("Coming soon...")
                    placeholder.setStyleSheet("color: #aaa; font-size: 10pt; border: none;")

                    layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignTop)
                    layout.addWidget(placeholder)
                    layout.addStretch()
                    return card


                weekly_sleep_chart = create_weekly_sleep_chart(me.get("sleep_log", []))
                weekly_sleep_chart.setMinimumHeight(140)

                weekly_avg_widget = create_weekly_sleep_averages(me.get("sleep_log", []))
                weekly_avg_widget.setMinimumHeight(140)

                quality_trend_widget = create_weekly_quality_chart(me.get("sleep_log", []))
                quality_trend_widget.setMinimumHeight(140)

                sleep_consistency_heatmap = create_sleep_consistency_heatmap(me.get("sleep_log", []))
                sleep_consistency_heatmap.setMinimumHeight(140)


                card1 = create_card_with_chart("Weekly Sleep Duration", weekly_sleep_chart)
                card2 = create_card_with_chart("Weekly Averages", weekly_avg_widget)

                card3 = create_card_with_chart("Sleep Quality Breakdown", quality_trend_widget)
                card4 = create_card_with_chart("Sleep Consistency", sleep_consistency_heatmap)

                for card in [card1, card2, card3, card4]:
                    card.setMinimumSize(220, 150)
                    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

                grid_layout.addWidget(card1, 0, 0)
                grid_layout.addWidget(card2, 0, 1)
                grid_layout.addWidget(card3, 1, 0)
                grid_layout.addWidget(card4, 1, 1)

                card5 = create_sleep_pattern_insights_card(me.get("sleep_log", []))
                card5.setMinimumWidth(160)
                card5.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

                content_row.addWidget(grid_widget, stretch=2)
                content_row.addWidget(card5, stretch=1)

                main_layout.addLayout(content_row)

                return tab



            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(20, 20, 20, 20)

            tab_widget = QTabWidget()
            tab_widget.setStyleSheet("""
                QTabWidget::pane {
                    border: none;
                    background-color: transparent;
                }
                QTabBar::tab {
                    border: none;
                    background: #2e2e3e;
                    color: #ccc;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background: #222;
                    color: #fff;
                }
            """)

            tab_widget.addTab(create_sleep_log_tab(), "Sleep Log")
            tab_widget.addTab(create_sleep_summary_tab(), "Summary")
            layout.addWidget(tab_widget)

            return page

        def create_heart_rate_monitor_tab() -> QWidget:

            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            me.setdefault("heart_rate_log", [])

            def create_heart_beat_log_tab() -> QWidget:
                logs_per_page = 5
                current_page = [1]

                def bluetooth_heartbeat_monitor():
                    return 90
                tab = QWidget()
                tab.setStyleSheet("""
                    background-color: #222;
                    border-top-left-radius: 0px;
                    border-top-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border-bottom-right-radius: 12px;
                """)
                layout = QVBoxLayout(tab)
                layout.setSpacing(12)
                layout.setContentsMargins(20, 20, 20, 20)
                layout.setAlignment(Qt.AlignmentFlag.AlignTop)


                title_bar = QWidget()
                title_layout = QHBoxLayout(title_bar)
                title_layout.setContentsMargins(0, 0, 0, 0)
                title_layout.setSpacing(4)
                title = QLabel("Heartbeat Log")
                title.setStyleSheet("color:#8AB6D6; font-size:20pt; font-weight:bold;")

                filter_btn = QPushButton()
                filter_btn.setIcon(QIcon(resource_path("assets/filter.png")))
                filter_btn.setIconSize(QSize(40, 40))
                filter_btn.setFixedSize(48, 48)
                filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                filter_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,255,255,0.1);
                        border-radius:6px;
                    }
                """)

                clear_filter_btn = QPushButton()
                clear_filter_btn.setIcon(QIcon(resource_path("assets/clear.png")))
                clear_filter_btn.setIconSize(QSize(16, 16))
                clear_filter_btn.setFixedSize(20, 20)
                clear_filter_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                clear_filter_btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: rgba(255,255,255,0.1);
                        border-radius:10px;
                    }
                """)
                clear_filter_btn.setVisible(False)

                filter_wrap = QWidget()
                filter_layout = QGridLayout(filter_wrap)
                filter_layout.setContentsMargins(0, 0, 0, 0)
                filter_layout.setSpacing(0)
                filter_layout.addWidget(filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
                filter_layout.addWidget(clear_filter_btn, 0, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

                title_layout.addWidget(title)
                title_layout.addStretch()
                title_layout.addWidget(filter_wrap)
                layout.addWidget(title_bar)


                quick_row = QHBoxLayout()
                quick_row.setSpacing(6)
                quick_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
                selected_filter = [None]
                last_filtered = [None]
                def highlight_buttons(btn):
                    for i in range(quick_row.count()):
                        w = quick_row.itemAt(i).widget()
                        w.setStyleSheet("background:#444;color:#fff;padding:4px 12px;border-radius:6px;")
                    if btn:
                        btn.setStyleSheet("background:#00BCD4;color:#fff;padding:4px 12px;border-radius:6px;font-weight:bold;")

                for label, days in [("Day",1),("Week",7),("Month",30),("Year",365)]:
                    b = QPushButton(f"Past {label}")
                    b.clicked.connect(lambda _, d=days, btn=b: [apply_date_range(d), highlight_buttons(btn)])
                    quick_row.addWidget(b)

                highlight_buttons(None)
                layout.addLayout(quick_row)

                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setSpacing(8)
                container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
                scroll = wrap_scroll(container)
                layout.addWidget(scroll)

                pagination = QWidget()
                pagination_layout = QHBoxLayout(pagination)
                pagination_layout.setContentsMargins(0, 4, 0, 4)
                pagination_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                prev_btn = QPushButton("←")
                next_btn = QPushButton("→")
                page_input = QLineEdit("1")
                page_input.setFixedWidth(40)
                page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
                page_input.setValidator(QIntValidator(1, 999999999))
                page_input.setStyleSheet("""
                    QLineEdit {
                        color: #fff;
                        background: #222;
                        border-radius: 4px;
                        padding: 2px 4px;
                    }
                """)

                total_pages_lbl = QLabel("/ 1")
                total_pages_lbl.setStyleSheet("color:#ccc;")

                for btn in (prev_btn, next_btn):
                    btn.setFixedSize(32, 32)
                    btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn.setStyleSheet("""
                        QPushButton {
                            background: #5C6BC0;
                            color: #fff;
                            border-radius: 6px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background: #3F51B5;
                        }
                    """)

                pagination_layout.addWidget(prev_btn)
                pagination_layout.addWidget(page_input)
                pagination_layout.addWidget(total_pages_lbl)
                pagination_layout.addWidget(next_btn)

                pagination_widget = QWidget()
                pagination_widget.setLayout(pagination_layout)
                pagination_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
                layout.addWidget(pagination_widget, alignment=Qt.AlignmentFlag.AlignHCenter)


                def open_filter_dialog():
                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")

                    dlg = QDialog()
                    dlg.setWindowTitle("Filter Heartbeat Logs")
                    dlg.setWindowIcon(QIcon(resource_path("assets/filter.png")))
                    dlg.setFixedWidth(300)
                    dlg.setStyleSheet(f"""
                        QDialog {{
                            background-color: #2e2e3e;
                            color: #fff;
                            border-radius: 12px;
                        }}
                        QLabel {{
                            color: #ccc;
                            font-size: 11pt;
                        }}
                        QComboBox, QDateEdit, QLineEdit {{
                            background: #444;
                            color: #fff;
                            border: none;
                            border-radius: 6px;
                            padding: 6px;
                        }}
                        QComboBox::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QComboBox::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}
                        QComboBox QAbstractItemView {{
                            background: #333;
                            color: #fff;
                            selection-background-color: #5C6BC0;
                            border: none;
                            outline: none;
                            padding: 4px;
                        }}
                        QDateEdit::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QDateEdit::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}
                        QPushButton {{
                            background-color: #5C6BC0;
                            color: white;
                            font-weight: bold;
                            padding: 8px;
                            border-radius: 6px;
                        }}
                        QPushButton:hover {{
                            background-color: #3F51B5;
                        }}
                    """)

                    layout = QVBoxLayout(dlg)
                    layout.setContentsMargins(20, 20, 20, 20)
                    layout.setSpacing(12)


                    source_box = QComboBox(); source_box.addItems(["All", "Webcam", "Bluetooth", "Manual"])
                    layout.addWidget(QLabel("Source:")); layout.addWidget(source_box)


                    start = QDateEdit(); end = QDateEdit()
                    start.setCalendarPopup(True); end.setCalendarPopup(True)
                    start.setDate(QDate(2000, 1, 1)); end.setDate(QDate.currentDate())
                    layout.addWidget(QLabel("Start Date:")); layout.addWidget(start)
                    layout.addWidget(QLabel("End Date:")); layout.addWidget(end)


                    comp = QComboBox(); comp.addItems(["Any", ">=", "<=", "="])
                    bpm_val = QLineEdit(); bpm_val.setPlaceholderText("BPM")
                    h = QHBoxLayout(); h.addWidget(comp); h.addWidget(bpm_val)
                    layout.addWidget(QLabel("BPM:")); layout.addLayout(h)


                    cat = QComboBox(); cat.addItems(["All", "Resting", "During Exercise", "After Exercise"])
                    layout.addWidget(QLabel("Category:")); layout.addWidget(cat)


                    btn = QPushButton("Apply Filter")
                    btn.clicked.connect(lambda: [
                        apply_advanced_filter(source_box.currentText(), start.date(), end.date(),
                                            comp.currentText(), bpm_val.text(), cat.currentText()),
                        dlg.accept()
                    ])
                    layout.addWidget(btn)

                    dlg.exec()


                filter_btn.clicked.connect(open_filter_dialog)
                clear_filter_btn.clicked.connect(lambda: [last_filtered.__setitem__(0, None), refresh_logs(), clear_filter_btn.setVisible(False), highlight_buttons(None)])


                def apply_advanced_filter(src, start_dt, end_dt, cmp, bpm_text, category):
                    logs = me.get("heart_rate_log", [])
                    filtered = []
                    for l in logs:
                        d = QDate.fromString(l['date'],"yyyy-MM-dd")
                        if not (start_dt <= d <= end_dt): continue
                        if src != "All" and l['source'] != src: continue
                        if category != "All" and l['category'] != category: continue
                        if cmp != "Any" and bpm_text.isdigit():
                            b = int(bpm_text); lb = l['bpm']
                            if cmp==">=" and not lb>=b: continue
                            if cmp=="<=" and not lb<=b: continue
                            if cmp=="=" and not lb==b: continue
                        filtered.append(l)
                    last_filtered[0] = filtered
                    clear_filter_btn.setVisible(True)
                    current_page[0] = 1
                    refresh_logs(filtered)

                def apply_date_range(days):
                    cutoff = QDate.currentDate().addDays(-days)
                    logs = me.get("heart_rate_log", [])
                    filtered = [l for l in logs if QDate.fromString(l['date'],"yyyy-MM-dd")>=cutoff]
                    last_filtered[0] = filtered
                    clear_filter_btn.setVisible(True)
                    highlight_buttons(None)
                    current_page[0] = 1
                    refresh_logs(filtered)

                def open_log_dialog(source_type="Manual", bpm_value="", existing_log=None):
                    down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")
                    dlg = QDialog(self)
                    dlg.setWindowTitle("Log Heart Rate")
                    dlg.setWindowIcon(QIcon(resource_path("assets/heartratelog.png")))
                    dlg.setFixedWidth(300)
                    dlg.setStyleSheet(f"""
                        QDialog {{
                            background:#2e2e3e;
                            color:#fff;
                        }}
                        QLabel {{
                            background: transparent;
                        }}
                        QComboBox {{
                            background: #444;
                            color: #fff;
                            padding: 6px 28px 6px 6px;
                            border-radius: 6px;
                        }}
                        QComboBox::drop-down {{
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }}
                        QComboBox::down-arrow {{
                            image: url("{down_arrow_path}");
                            width: 14px;
                            height: 14px;
                        }}
                        QPushButton {{
                            background: #5C6BC0;
                            color: #fff;
                            font-weight: bold;
                            padding: 8px;
                            border-radius: 6px;
                        }}
                        QPushButton:hover {{
                            background: #3F51B5;
                        }}
                        QLineEdit {{
                            background:#444;
                            color:#fff;
                            padding:6px;
                            border-radius:6px;
                        }}
                        QTextEdit {{
                            background:#444;
                            color:#fff;
                            border-radius:6px;
                        }}
                    """)

                    layout = QVBoxLayout(dlg)
                    layout.setContentsMargins(16, 16, 16, 16)


                    source_selector = QComboBox()
                    source_selector.addItems(["Webcam", "Bluetooth", "Manual"])
                    source_selector.setCurrentText(source_type)
                    layout.addWidget(QLabel("Source:"))
                    layout.addWidget(source_selector)


                    bpm_widget_container = QWidget()
                    bpm_widget_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
                    bpm_widget_container.setStyleSheet("background: transparent;")
                    bpm_layout = QVBoxLayout(bpm_widget_container)
                    bpm_layout.setContentsMargins(0, 0, 0, 0)
                    layout.addWidget(QLabel("BPM:"))
                    layout.addWidget(bpm_widget_container)



                    cat_box = QComboBox()
                    cat_box.addItems(["Resting", "During Exercise", "After Exercise"])
                    layout.addWidget(QLabel("Category:"))
                    layout.addWidget(cat_box)


                    note_input = QTextEdit()
                    note_input.setPlaceholderText("Optional note")
                    note_input.setFixedHeight(80)
                    layout.addWidget(QLabel("Note:"))
                    layout.addWidget(note_input)


                    save_btn = QPushButton("Save")
                    save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    layout.addWidget(save_btn)

                    bpm_input = None
                    record_btn = None
                    bpm_display_input = None

                    def clear_layout(layout):
                        while layout.count():
                            item = layout.takeAt(0)
                            widget = item.widget()
                            if widget:
                                widget.deleteLater()

                    def setup_manual_mode():
                        nonlocal bpm_input, record_btn, bpm_display_input
                        clear_layout(bpm_layout)
                        bpm_input = QLineEdit(str(bpm_value))
                        bpm_input.setPlaceholderText("Enter BPM")
                        bpm_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;")
                        bpm_layout.addWidget(bpm_input)
                        record_btn = None
                        bpm_display_input = None
                        QTimer.singleShot(0, lambda: dlg.resize(300, dlg.sizeHint().height()))

                    def setup_record_mode(monitor_func):
                        nonlocal bpm_input, record_btn, bpm_display_input
                        clear_layout(bpm_layout)
                        bpm_input = None

                        record_btn = QPushButton("Record BPM")
                        record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                        record_btn.setStyleSheet("""
                            QPushButton {
                                background: #5C6BC0;
                                color: #fff;
                                font-weight: bold;
                                padding: 8px;
                                border-radius: 6px;
                            }
                            QPushButton:hover {
                                background: #3F51B5;
                            }
                        """)
                        bpm_layout.addWidget(record_btn)

                        bpm_display_input = QLineEdit()
                        bpm_display_input.setPlaceholderText("Result")
                        bpm_display_input.setReadOnly(True)
                        bpm_display_input.setStyleSheet("background:#444;color:#fff;padding:6px;border-radius:6px;margin-top:6px;")
                        bpm_layout.addWidget(bpm_display_input)

                        def on_record_clicked():
                            bpm = monitor_func()
                            if bpm == -1:
                                bpm_display_input.setStyleSheet("background:#444; color:red; padding:6px; border-radius:6px; margin-top:6px;")
                                bpm_display_input.setText("Recording failed or canceled")
                            else:
                                bpm_display_input.setStyleSheet("background:#444; color:#fff; padding:6px; border-radius:6px; margin-top:6px;")
                                bpm_display_input.setText(str(bpm))

                        record_btn.clicked.connect(on_record_clicked)
                        QTimer.singleShot(0, lambda: dlg.resize(300, dlg.sizeHint().height()))





                    if source_type == "Manual":
                        setup_manual_mode()
                    elif source_type == "Webcam":
                        setup_record_mode(webcam_heartbeat_monitor)
                    elif source_type == "Bluetooth":
                        setup_record_mode(bluetooth_heartbeat_monitor)

                    def on_source_changed(text):
                        nonlocal bpm_value
                        if text == "Manual":
                            setup_manual_mode()
                        elif text == "Webcam":
                            setup_record_mode(webcam_heartbeat_monitor)
                        elif text == "Bluetooth":
                            setup_record_mode(bluetooth_heartbeat_monitor)

                    source_selector.currentTextChanged.connect(on_source_changed)


                    if existing_log:
                        cat_box.setCurrentText(existing_log["category"])
                        note_input.setText(existing_log.get("note", ""))
                        source_selector.setCurrentText(existing_log.get("source", "Manual"))
                        if existing_log.get("source") == "Manual":
                            bpm_value = str(existing_log["bpm"])

                    def save():
                        try:
                            if source_selector.currentText() == "Manual":
                                bpm = int(bpm_input.text())
                            else:
                                bpm = int(bpm_display_input.text())
                        except Exception:
                            QMessageBox.warning(dlg, "Invalid BPM", "Please enter a valid BPM number or record BPM first.")
                            return

                        log = {
                            "bpm": bpm,
                            "category": cat_box.currentText(),
                            "note": note_input.toPlainText().strip(),
                            "source": source_selector.currentText(),
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "time": datetime.now().strftime("%H:%M"),
                        }

                        if existing_log:
                            me["heart_rate_log"].remove(existing_log)
                        me["heart_rate_log"].append(log)
                        Path("user_data.json").write_text(json.dumps(users, indent=4))
                        dlg.accept()
                        refresh_logs()

                    save_btn.clicked.connect(save)
                    QTimer.singleShot(0, lambda: dlg.resize(300, dlg.sizeHint().height()))

                    dlg.exec()

                def delete_log(log: dict):
                    me["heart_rate_log"].remove(log)
                    Path("user_data.json").write_text(json.dumps(users, indent=4))
                    refresh_logs()

                def log_card(log: dict) -> QWidget:
                    card = QFrame()
                    card.setStyleSheet("background:#37474F;border-radius:10px;")
                    shadow_effect = QGraphicsDropShadowEffect()
                    shadow_effect.setOffset(0, 4)
                    shadow_effect.setBlurRadius(10)
                    semi_transparent_white = QColor(0, 0, 0, int(255 * 0.3))
                    shadow_effect.setColor(semi_transparent_white)
                    card.setGraphicsEffect(shadow_effect)

                    layout = QVBoxLayout(card)
                    layout.setContentsMargins(10, 8, 10, 8)

                    top = QHBoxLayout()
                    lbl = QLabel(f"{log['bpm']} bpm • {log['category']}")
                    lbl.setStyleSheet("color:#00BCD4;font-weight:bold;font-size:12pt;")
                    top.addWidget(lbl)
                    top.addStretch()

                    edit_btn = QPushButton()
                    edit_btn.setIcon(QIcon(resource_path("assets/edit.png")))
                    edit_btn.setFixedSize(28, 28)
                    edit_btn.setIconSize(QSize(20, 20))
                    edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    edit_btn.setStyleSheet("""
                        QPushButton {
                            background: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(255,255,255,0.1);
                            border-radius: 6px;
                        }
                    """)
                    edit_btn.clicked.connect(lambda: open_log_dialog(log["source"], log["bpm"], log))

                    del_btn = QPushButton()
                    del_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                    del_btn.setFixedSize(28, 28)
                    del_btn.setIconSize(QSize(20, 20))
                    del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    del_btn.setStyleSheet("""
                        QPushButton {
                            background: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(255,255,255,0.1);
                            border-radius: 6px;
                        }
                    """)
                    del_btn.clicked.connect(lambda: delete_log(log))

                    btn_layout = QHBoxLayout()
                    btn_layout.setSpacing(4)
                    btn_layout.addWidget(edit_btn)
                    btn_layout.addWidget(del_btn)
                    top.addLayout(btn_layout)

                    layout.addLayout(top)

                    sub = QLabel(f"Source: {log['source']} | {log['date']} {log['time']}")
                    sub.setStyleSheet("color:#aaa; font-size:10pt;")
                    layout.addWidget(sub)

                    if log.get("note"):
                        note = QLabel(f"Note: {log['note']}")
                        note.setStyleSheet("color:#ccc; font-size:9pt;")
                        note.setWordWrap(True)
                        layout.addWidget(note)

                    return card


                def refresh_logs(filtered=None):
                    logs = filtered if filtered is not None else me.get("heart_rate_log", [])
                    total = max(1,(len(logs)+logs_per_page-1)//logs_per_page)
                    current_page[0] = min(current_page[0], total)
                    for i in reversed(range(container_layout.count())):
                        w = container_layout.itemAt(i).widget(); w and w.setParent(None)
                    if not logs:
                        lbl = QLabel("No logs."); lbl.setStyleSheet("color:#888;"); container_layout.addWidget(lbl)
                    else:
                        s=(current_page[0]-1)*logs_per_page; e=s+logs_per_page
                        for l in list(reversed(logs))[s:e]: container_layout.addWidget(log_card(l))
                    page_input.setText(str(current_page[0])); total_pages_lbl.setText(f"/ {total}")
                    prev_btn.setEnabled(current_page[0]>1); next_btn.setEnabled(current_page[0]<total)

                prev_btn.clicked.connect(lambda: [current_page.__setitem__(0,current_page[0]-1), refresh_logs(last_filtered[0] if last_filtered[0] else None)])
                next_btn.clicked.connect(lambda: [current_page.__setitem__(0,current_page[0]+1), refresh_logs(last_filtered[0] if last_filtered[0] else None)])
                page_input.returnPressed.connect(lambda: [current_page.__setitem__(0,int(page_input.text()) if page_input.text().isdigit() else 1), refresh_logs(last_filtered[0] if last_filtered[0] else None)])


                refresh_logs()



                add_btn = QPushButton("Add Heartbeat Log")
                add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                add_btn.setFixedSize(200, 40)
                add_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #5C6BC0;
                        color: white;
                        padding: 10px 24px;
                        border-radius: 8px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3F51B5;
                    }
                """)
                add_btn.clicked.connect(lambda: open_log_dialog("Manual"))
                layout.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

                return tab


            def create_heart_summary_tab() -> QWidget:
                tab = QWidget()
                tab.setStyleSheet("background-color: #222; border-radius: 12px;")
                layout = QVBoxLayout(tab)
                layout.setContentsMargins(20, 20, 20, 20)
                layout.setSpacing(12)

                down_arrow_path = resource_path("assets/down_arrow.png").replace("\\", "/")


                combo_style = f"""
                    QComboBox {{
                        background-color: #333;
                        color: #fff;
                        padding: 6px 28px 6px 12px;
                        border: 1px solid #555;
                        border-radius: 6px;
                        min-width: 50px;
                    }}
                    QComboBox::drop-down {{
                        subcontrol-origin: padding;
                        subcontrol-position: top right;
                        width: 24px;
                        border-left: 1px solid #555;
                    }}
                    QComboBox::down-arrow {{
                        image: url("{down_arrow_path}");
                        width: 14px;
                        height: 14px;
                    }}
                    QComboBox QAbstractItemView {{
                        background-color: #333;
                        color: #fff;
                        selection-background-color: #444;
                        border: none;
                        outline: none;
                        padding: 4px;
                    }}
                """

                def modern_combo():
                    combo = QComboBox()
                    combo.setStyleSheet(combo_style)
                    return combo

                def modern_edit():
                    edit = QLineEdit()
                    edit.setStyleSheet("""
                        QLineEdit {
                            background-color: #333;
                            color: #fff;
                            padding: 6px 12px;
                            border: 1px solid #555;
                            border-radius: 6px;
                            min-width: 60px;
                            max-width: 80px; 
                        }
                    """)
                    return edit


                filter_row = QHBoxLayout()
                filter_row.setSpacing(10)

                source_box = modern_combo()
                source_box.addItems(["All", "Webcam", "Bluetooth", "Manual"])

                bpm_cmp = modern_combo()
                bpm_cmp.addItems(["Any", ">=", "<=", "="])

                bpm_input = modern_edit()
                bpm_input.setPlaceholderText("BPM")
                bpm_input.setValidator(QIntValidator(0, 300))

                logs = me.get("heart_rate_log", [])
                dates = [QDate.fromString(l["date"], "yyyy-MM-dd") for l in logs if "date" in l]

                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                else:
                    min_date = QDate.currentDate().addMonths(-1)
                    max_date = QDate.currentDate()

                start_date = QDateEdit(min_date)
                end_date = QDateEdit(max_date)

                for d in [start_date, end_date]:
                    d.setCalendarPopup(True)
                    d.setStyleSheet("""
                        QDateEdit {
                            background-color: #333;
                            color: #fff;
                            padding: 6px 12px;
                            border: 1px solid #555;
                            border-radius: 6px;
                            min-width: 120px;
                        }
                        QDateEdit::drop-down {
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #555;
                        }
                        QDateEdit::down-arrow {
                            image: url("%s");
                            width: 14px;
                            height: 14px;
                        }
                    """ % down_arrow_path)

                clear_btn = QPushButton("Reset Filters")
                clear_btn.setFixedWidth(100)
                clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                clear_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #444;
                        color: #bbb;
                        padding: 6px 12px;
                        border-radius: 6px;
                    }
                    QPushButton:disabled {
                        background-color: #333;
                        color: #666;
                    }
                """)
                clear_btn.setEnabled(False)

                def label_input_pair(text: str, widget: QWidget) -> QWidget:
                    container = QWidget()
                    layout = QHBoxLayout()
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)

                    label = QLabel(text)
                    label.setStyleSheet("color: #ccc;")
                    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    label.setFixedWidth(45)

                    layout.addWidget(label)
                    layout.addWidget(widget)
                    layout.setStretch(1, 1)

                    container.setLayout(layout)
                    container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    return container

                filter_row.addWidget(label_input_pair("Source:", source_box))
                filter_row.addWidget(label_input_pair("From:", start_date))
                filter_row.addWidget(label_input_pair("To:", end_date))


                bpm_container = QWidget()
                bpm_layout = QHBoxLayout(bpm_container)
                bpm_layout.setContentsMargins(0, 0, 0, 0)
                bpm_layout.setSpacing(4)
                bpm_label = QLabel("BPM:")
                bpm_label.setStyleSheet("color: #ccc;")
                bpm_layout.addWidget(bpm_label)
                bpm_layout.addWidget(bpm_cmp)
                bpm_layout.addWidget(bpm_input)
                bpm_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                filter_row.addWidget(bpm_container)

                filter_row.addWidget(clear_btn)



                layout.addLayout(filter_row)


                category_row = QHBoxLayout()
                category_row.setSpacing(10)
                category_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

                selected_category = ["Resting"]
                category_buttons = {}

                def highlight_category(btn):
                    for b in category_buttons.values():
                        b.setStyleSheet("background:#444;color:#fff;padding:6px 12px;border-radius:8px;")
                    btn.setStyleSheet("background:#4CAF50;color:#fff;padding:6px 12px;border-radius:8px;font-weight:bold;")

                for cat in ["Resting", "During Exercise", "After Exercise"]:
                    b = QPushButton(cat)
                    b.setCursor(Qt.CursorShape.PointingHandCursor)
                    b.setStyleSheet("background:#444;color:#fff;padding:6px 12px;border-radius:8px;")
                    category_buttons[cat] = b
                    b.clicked.connect(lambda _, c=cat, btn=b: [
                        selected_category.__setitem__(0, c),
                        highlight_category(btn),
                        update_chart()
                    ])
                    category_row.addWidget(b)
                layout.addLayout(category_row)
                highlight_category(category_buttons["Resting"])

                canvas = FigureCanvas(Figure(figsize=(6, 3), facecolor="#222"))
                ax = canvas.figure.add_subplot(111)
                canvas.setStyleSheet("background-color: #222;")
                layout.addWidget(canvas)




                def update_chart():
                    logs = me.get("heart_rate_log", [])
                    cat = selected_category[0]
                    filtered = []

                    for l in logs:
                        if l["category"] != cat:
                            continue
                        d = QDate.fromString(l["date"], "yyyy-MM-dd")
                        if d < start_date.date() or d > end_date.date():
                            continue
                        if source_box.currentText() != "All" and l["source"] != source_box.currentText():
                            continue
                        if bpm_cmp.currentText() != "Any" and bpm_input.text().isdigit():
                            bpm = int(bpm_input.text())
                            val = l["bpm"]
                            if bpm_cmp.currentText() == ">=" and not val >= bpm: continue
                            if bpm_cmp.currentText() == "<=" and not val <= bpm: continue
                            if bpm_cmp.currentText() == "=" and not val == bpm: continue
                        filtered.append(l)

                    clear_btn.setEnabled(
                        source_box.currentIndex() != 0 or
                        bpm_cmp.currentIndex() != 0 or
                        bpm_input.text().strip() != "" or
                        start_date.date() != QDate.currentDate().addMonths(-1) or
                        end_date.date() != QDate.currentDate()
                    )


                    daily_avg = {}
                    for l in filtered:
                        day = l["date"]
                        daily_avg.setdefault(day, []).append(l["bpm"])

                    dates = sorted(daily_avg)
                    x = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
                    y = [sum(v) / len(v) for v in [daily_avg[d] for d in dates]]

                    ax.clear()
                    ax.set_facecolor("#222")
                    ax.plot(x, y, color="#4CAF50", linewidth=3)
                    ax.fill_between(x, y, color="#4CAF50", alpha=0.3)
                    ax.tick_params(colors="#ccc")
                    ax.set_title("Average BPM", color="#ccc")
                    ax.set_ylabel("BPM", color="#ccc")
                    ax.set_xlabel("Date", color="#ccc")
                    ax.grid(True, alpha=0.3)
                    canvas.draw()


                source_box.currentIndexChanged.connect(update_chart)
                bpm_cmp.currentIndexChanged.connect(update_chart)
                bpm_input.textChanged.connect(update_chart)
                start_date.dateChanged.connect(update_chart)
                end_date.dateChanged.connect(update_chart)

                clear_btn.clicked.connect(lambda: [
                    source_box.setCurrentIndex(0),
                    bpm_cmp.setCurrentIndex(0),
                    bpm_input.clear(),
                    start_date.setDate(min_date),
                    end_date.setDate(max_date),
                    update_chart()
                ])


                QTimer.singleShot(0, update_chart)

                return tab


            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(20, 20, 20, 20)

            tab_widget = QTabWidget()
            tab_widget.setStyleSheet("""
                QTabWidget::pane {
                    border: none;
                    background-color: transparent;
                }
                QTabBar::tab {
                    border: none;
                    background: #2e2e3e;
                    color: #ccc;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background: #222;
                    color: #fff;
                    border: none; 
                }
            """)


            tab_widget.addTab(create_heart_beat_log_tab(), "Heartbeat Log")
            tab_widget.addTab(create_heart_summary_tab(), "Summary")
            layout.addWidget(tab_widget)

            return page

        def create_settings_tab() -> QWidget:
            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            if me is None:
                me = {}

            tab = QWidget()
            layout = QHBoxLayout(tab)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)


            sidebar = QListWidget()
            sidebar.setFixedWidth(280)
            sidebar.setStyleSheet("""
                QListWidget {
                    background-color: #121212;
                    border: none;
                    padding: 16px;
                    outline: none;
                }
                QListWidget::item {
                    padding: 16px 20px;
                    margin-bottom: 12px;
                    font-size: 18px;
                    border-radius: 14px;
                    color: #ccc;
                    background-color: #1e1e1e;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
                QListWidget::item:selected {
                    background-color: #005792;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
                QListWidget::item:focus {
                    outline: none;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
                QListWidget::item:selected:!active {
                    outline: none;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
            """)

            sidebar_items = [
                ("Profile", "assets/profile.png"),
                ("Security", "assets/security.png"),
                ("Support", "assets/support.png"),
            ]

            for name, icon in sidebar_items:
                item = QListWidgetItem(QIcon(resource_path(icon)), f"  {name}")
                sidebar.addItem(item)

            def delete_account():
                confirm = QMessageBox()
                confirm.setIcon(QMessageBox.Icon.Warning)
                confirm.setWindowIcon(QIcon(resource_path("assets/delete.png")))
                confirm.setWindowTitle("Confirm Deletion")
                confirm.setText("Are you sure you want to delete your account? This action cannot be undone.")
                confirm.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)

                if confirm.exec() == QMessageBox.StandardButton.Ok:
                    try:
                        users = json.loads(Path("user_data.json").read_text())
                        updated_users = [u for u in users if u.get("username") != self.user_data.get("username")]
                        Path("user_data.json").write_text(json.dumps(updated_users, indent=2))


                        self.user_data.clear()
                        self.parent().login_page.reset_fields()
                        self.parent().setCurrentIndex(0)
                    except Exception as e:
                        error_dialog = QMessageBox()
                        error_dialog.setIcon(QMessageBox.Icon.Critical)
                        error_dialog.setWindowTitle("Error")
                        error_dialog.setText(f"Failed to delete account.\n\n{str(e)}")
                        error_dialog.exec()


            def logout():
                self.parent().login_page.reset_fields()
                self.parent().setCurrentIndex(0)


            logout_btn = QWidget()
            logout_btn.setFixedSize(260, 50)
            logout_btn.setStyleSheet("""
                background-color: #d32f2f;
                border-radius: 12px;
            """)
            logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            logout_layout = QHBoxLayout(logout_btn)
            logout_layout.setContentsMargins(16, 0, 16, 0)
            logout_layout.setSpacing(12)

            label = QLabel("Log Out")
            label.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            icon_label = QLabel()
            icon_label.setPixmap(QIcon(resource_path("assets/logout.png")).pixmap(24, 24))
            icon_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            logout_layout.addWidget(label)
            logout_layout.addStretch()
            logout_layout.addWidget(icon_label)


            logout_btn.mousePressEvent = lambda e: logout()



            sidebar_layout = QVBoxLayout()
            sidebar_layout.setContentsMargins(12, 12, 12, 12)
            sidebar_layout.setSpacing(16)
            sidebar_layout.addWidget(sidebar)
            sidebar_layout.addStretch()
            sidebar_layout.addWidget(logout_btn)

            sidebar_wrap = QFrame()
            sidebar_wrap.setLayout(sidebar_layout)
            sidebar_wrap.setStyleSheet("background-color: #121212;border: none; ")


            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.VLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            separator.setStyleSheet("color: white;")


            content_stack = QStackedWidget()
            content_stack.setStyleSheet("background-color: #121212;")

            def create_profile_tab():
                page = QWidget()
                layout = QVBoxLayout(page)
                layout.setContentsMargins(24, 24, 24, 24)
                layout.setSpacing(24)
                page.setStyleSheet("background-color: #121212;")


                profile_card = QFrame()
                profile_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 20px;
                        border: none;
                    }
                """)
                profile_card_layout = QVBoxLayout(profile_card)
                profile_card_layout.setContentsMargins(20, 20, 20, 20)
                profile_card_layout.setSpacing(12)


                username = me.get("username", "default_user")
                profile_pic_widget = ProfilePicWithEdit(username, me)
                profile_card_layout.addWidget(profile_pic_widget, alignment=Qt.AlignmentFlag.AlignHCenter)



                full_name = f"{me.get('first_name', '')} {me.get('last_name', '')}".strip() or "Unnamed"
                name_editable = EditableLabel(full_name, me)
                profile_card_layout.addWidget(name_editable, alignment=Qt.AlignmentFlag.AlignHCenter)

                layout.addWidget(profile_card)


                contact_card = QFrame()
                contact_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 20px;
                        border: none;
                    }
                """)
                contact_layout = QVBoxLayout(contact_card)
                contact_layout.setContentsMargins(20, 16, 20, 16)
                contact_layout.setSpacing(24)

                email = me.get("email", "Not set")
                phone = me.get("phone", "Not provided")
                contact_layout.addWidget(EditableContactRow("assets/email.png", "Email:", email, me, "email"))
                contact_layout.addWidget(EditableContactRow("assets/phone.png", "Phone:", phone, me, "phone"))

                layout.addWidget(contact_card)


                layout.addStretch()


                delete_row = QHBoxLayout()
                delete_row.addStretch()
                delete_btn = QPushButton("Delete Account")
                delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                delete_btn.setIcon(QIcon(resource_path("assets/trash.png")))
                delete_btn.setIconSize(QSize(16, 16))
                delete_btn.setFixedHeight(36)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #d32f2f;
                        color: white;
                        font-size: 12pt;
                        border-radius: 8px;
                        padding: 6px 16px;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: #b71c1c;
                    }
                """)
                delete_row.addWidget(delete_btn)
                delete_btn.clicked.connect(delete_account)
                layout.addLayout(delete_row)

                return page


            def create_security_tab():
                page = QWidget()
                layout = QVBoxLayout(page)
                layout.setContentsMargins(24, 24, 24, 24)
                layout.setSpacing(24)
                page.setStyleSheet("background-color: #121212;")


                title = QLabel("Security Settings")
                title.setStyleSheet("color: white; font-size: 20pt; font-weight: bold;")
                layout.addWidget(title)


                if not me.get("verified", False):
                    verify_card = QFrame()
                    verify_card.setStyleSheet("""
                        QFrame {
                            background-color: #1e1e1e;
                            border-radius: 16px;
                        }
                    """)
                    verify_layout = QVBoxLayout(verify_card)
                    verify_layout.setContentsMargins(20, 20, 20, 20)
                    verify_layout.setSpacing(12)

                    info = QLabel("Verify your email to enable 2FA and improve your account security.")
                    info.setWordWrap(True)
                    info.setStyleSheet("color: white; font-size: 12pt;")
                    verify_layout.addWidget(info)

                    verify_btn = QPushButton("Verify Email")
                    verify_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    verify_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #00897b;
                            color: white;
                            font-size: 13pt;
                            padding: 10px;
                            border-radius: 10px;
                        }
                        QPushButton:hover {
                            background-color: #00695c;
                        }
                    """)
                    verify_layout.addWidget(verify_btn, alignment=Qt.AlignmentFlag.AlignRight)

                    def verify_email():
                        success = send_verification_email(me["email"], me["username"])
                        if success:
                            QMessageBox.information(page, "Verification Sent",
                                "Verification email sent! Please check your inbox and click the link to verify your account.")
                        else:
                            QMessageBox.critical(page, "Error", "Failed to send verification email. Please try again.")


                    verify_btn.clicked.connect(verify_email)
                    layout.addWidget(verify_card)


                pw_card = QFrame()
                pw_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 16px;
                    }
                """)
                pw_layout = QVBoxLayout(pw_card)
                pw_layout.setContentsMargins(20, 20, 20, 20)
                pw_layout.setSpacing(12)

                pw_label = QLabel("Change your password")
                pw_label.setStyleSheet("color: white; font-size: 14pt;")
                pw_layout.addWidget(pw_label)

                change_pw_btn = QPushButton("Change Password")
                change_pw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                change_pw_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #005792;
                        color: white;
                        font-size: 13pt;
                        padding: 10px;
                        border-radius: 10px;
                    }
                    QPushButton:hover {
                        background-color: #0077b6;
                    }
                """)
                pw_layout.addWidget(change_pw_btn, alignment=Qt.AlignmentFlag.AlignRight)

                layout.addWidget(pw_card)


                twofa_card = QFrame()
                twofa_card.setVisible(me.get("verified", False))
                twofa_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 16px;
                    }
                """)
                twofa_layout = QVBoxLayout(twofa_card)
                twofa_layout.setContentsMargins(20, 20, 20, 20)
                twofa_layout.setSpacing(12)

                info = QLabel("Enable Two-Factor Authentication for enhanced security.")
                info.setWordWrap(True)
                info.setStyleSheet("color: white; font-size: 12pt;")
                twofa_layout.addWidget(info)

                twofa_enabled = me.get("2fa", False)
                enable_btn = QPushButton("Disable 2FA" if twofa_enabled else "Enable 2FA")
                enable_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                def update_2fa_button_style():
                    if me.get("2fa", False):
                        enable_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #c62828;
                                color: white;
                                font-size: 13pt;
                                padding: 10px;
                                border-radius: 10px;
                            }
                            QPushButton:hover {
                                background-color: #b71c1c;
                            }
                        """)
                    else:
                        enable_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #00796b;
                                color: white;
                                font-size: 13pt;
                                padding: 10px;
                                border-radius: 10px;
                            }
                            QPushButton:hover {
                                background-color: #004d40;
                            }
                        """)

                def toggle_2fa():
                    me["2fa"] = not me.get("2fa", False)
                    for user in users:
                        if user["username"] == me["username"]:
                            user["2fa"] = me["2fa"]
                            break
                    Path("user_data.json").write_text(json.dumps(users, indent=2))
                    enable_btn.setText("Disable 2FA" if me["2fa"] else "Enable 2FA")
                    update_2fa_button_style()

                enable_btn.clicked.connect(toggle_2fa)
                update_2fa_button_style()
                twofa_layout.addWidget(enable_btn, alignment=Qt.AlignmentFlag.AlignRight)


                layout.addWidget(twofa_card)
                layout.addStretch()


                def open_password_dialog():
                    dialog = QDialog(page)
                    dialog.setWindowTitle("Change Password")
                    dialog.setFixedWidth(400)
                    dialog.setStyleSheet("background-color: #1e1e1e; color: white;")

                    dlg_layout = QVBoxLayout(dialog)
                    dlg_layout.setContentsMargins(20, 20, 20, 20)
                    dlg_layout.setSpacing(16)

                    def labeled_input(label_text, placeholder):
                        container = QVBoxLayout()
                        label = QLabel(label_text)
                        label.setStyleSheet("color: #ccc; font-size: 11pt;")
                        field = QLineEdit()
                        field.setPlaceholderText(placeholder)
                        field.setEchoMode(QLineEdit.EchoMode.Password)
                        field.setStyleSheet("background-color: #2c2c2c; padding: 8px; border-radius: 8px;")
                        container.addWidget(label)
                        container.addWidget(field)
                        return container, field

                    curr_layout, current = labeled_input("Current Password", "Enter current password")
                    new_layout, new = labeled_input("New Password", "Enter new password")
                    conf_layout, confirm = labeled_input("Confirm New Password", "Re-enter new password")

                    dlg_layout.addLayout(curr_layout)
                    dlg_layout.addLayout(new_layout)
                    dlg_layout.addLayout(conf_layout)


                    toggle_row = QHBoxLayout()
                    toggle_btn = QPushButton()
                    toggle_btn.setIcon(QIcon(resource_path("assets/closedeye.png")))
                    toggle_btn.setFixedSize(24, 24)
                    toggle_btn.setStyleSheet("border: none;")
                    toggle_label = QLabel("Show Password")
                    toggle_label.setStyleSheet("color: white; font-size: 10pt;")
                    toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                    def toggle_visibility():
                        is_hidden = current.echoMode() == QLineEdit.EchoMode.Password
                        for field in (current, new, confirm):
                            field.setEchoMode(QLineEdit.EchoMode.Normal if is_hidden else QLineEdit.EchoMode.Password)
                        toggle_btn.setIcon(QIcon(resource_path("assets/openeye.png") if is_hidden else resource_path("assets/closedeye.png")))
                        toggle_label.setText("Hide Password" if is_hidden else "Show Password")

                    toggle_btn.clicked.connect(toggle_visibility)

                    toggle_row.addWidget(toggle_btn)
                    toggle_row.addSpacing(8)
                    toggle_row.addWidget(toggle_label)
                    toggle_row.addStretch()
                    dlg_layout.addLayout(toggle_row)


                    save_btn = QPushButton("Save")
                    save_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #00aaff;
                            padding: 10px;
                            border-radius: 8px;
                            font-weight: bold;
                            color: white;
                        }
                        QPushButton:hover {
                            background-color: #007acc;
                        }
                    """)

                    def submit_change():
                        if not current.text() or not new.text() or not confirm.text():
                            QMessageBox.warning(dialog, "Missing", "Please fill all fields.")
                            return
                        if new.text() != confirm.text():
                            QMessageBox.warning(dialog, "Mismatch", "New passwords do not match.")
                            return
                        is_strong, message = check_password_strength(new.text())
                        if not is_strong:
                            QMessageBox.warning(dialog, "Weak Password", message)
                            return
                        if current.text() != me.get("password"):
                            QMessageBox.critical(dialog, "Wrong", "Current password is incorrect.")
                            return

                        me["password"] = new.text()
                        for user in users:
                            if user["username"] == me["username"]:
                                user["password"] = new.text()
                                break
                        Path("user_data.json").write_text(json.dumps(users, indent=2))
                        QMessageBox.information(dialog, "Success", "Password updated.")
                        dialog.accept()


                    save_btn.clicked.connect(submit_change)
                    dlg_layout.addWidget(save_btn)

                    dialog.exec()

                change_pw_btn.clicked.connect(open_password_dialog)

                def check_verification_status():
                    try:
                        with open("user_data.json") as f:
                            users = json.load(f)
                        for user in users:
                            if user["username"] == me["username"]:
                                if user.get("verified", False) and not me.get("verified", False):
                                    me["verified"] = True

                                    if verify_card.isVisible():
                                        page.layout().removeWidget(verify_card)
                                        verify_card.deleteLater()
                                    twofa_card.setVisible(True)
                                    timer.stop()
                                break
                    except Exception as e:
                        print("Error checking verification status:", e)

                timer = QTimer(page)
                timer.timeout.connect(check_verification_status)
                timer.start(5000)

                return page


            def create_support_tab():
                page = QWidget()
                outer_layout = QVBoxLayout(page)
                outer_layout.setContentsMargins(0, 0, 0, 0)
                outer_layout.setSpacing(0)

                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_area.setStyleSheet("""
                    QScrollBar:vertical, QScrollBar:horizontal {
                        background: transparent;
                        width: 8px; height: 8px;
                    }
                    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                        background: #555;
                        border-radius: 4px;
                    }
                    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
                        background: none;
                        height: 0;
                        width: 0;
                    }
                """)

                scroll_content = QWidget()
                layout = QVBoxLayout(scroll_content)
                layout.setContentsMargins(24, 24, 24, 24)
                layout.setSpacing(24)
                scroll_content.setStyleSheet("background-color: #121212; color: white;")

                title = QLabel("Support & Help")
                title.setStyleSheet("font-size: 20pt; font-weight: bold;")
                layout.addWidget(title)

                contact_card = QFrame()
                contact_card.setStyleSheet("background-color: #1e1e1e; border-radius: 16px;")
                contact_layout = QVBoxLayout(contact_card)
                contact_layout.setContentsMargins(20, 20, 20, 20)
                contact_layout.setSpacing(12)

                email_input = QLineEdit()
                email_input.setPlaceholderText("Your email (optional)")
                email_input.setStyleSheet("background-color: #2c2c2c; padding: 8px; border-radius: 8px; color: white;")

                message_container = QFrame()
                message_container.setStyleSheet("""
                    background-color: #2c2c2c;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: 0px;
                    border:none;
                """)
                message_container_layout = QVBoxLayout(message_container)
                message_container_layout.setContentsMargins(8, 8, 8, 8)
                message_container_layout.setSpacing(4)

                attachments_tray = QFrame()
                attachments_tray.setStyleSheet("background-color: transparent; border-radius: 12px;")
                attachments_layout = QHBoxLayout(attachments_tray)
                attachments_layout.setContentsMargins(8, 8, 8, 8)
                attachments_layout.setSpacing(8)
                attachments_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                attachments_tray.setVisible(False)
                message_container_layout.addWidget(attachments_tray)

                message_input = QTextEdit()
                message_input.setPlaceholderText("Describe your issue or question here...")
                message_input.setMinimumHeight(120)
                message_input.setStyleSheet("""
                    QTextEdit {
                        background-color: #2c2c2c;
                        padding: 8px;
                        border-top-left-radius: 12px;
                        border-top-right-radius: 12px;
                        border-bottom-right-radius: 0px;
                        border-bottom-left-radius: 0px;
                        color: white;
                        border:none;
                    }
                    QScrollBar:vertical {
                        background: transparent;
                        width: 8px;
                    }
                    QScrollBar::handle:vertical {
                        background: #555;
                        border-radius: 4px;
                    }
                """)
                message_container_layout.addWidget(message_input)



                attach_card = QFrame()
                attach_card.setStyleSheet("""
                    background-color: #2c2c2c;
                    border-top-left-radius: 0px;
                    border-top-right-radius: 0px;
                    border-bottom-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border:none;
                """)

                attach_card_layout = QHBoxLayout(attach_card)
                attach_card_layout.setContentsMargins(8, 0, 8, 8)
                attach_card_layout.setSpacing(0)

                attach_btn = QPushButton()
                attach_btn.setIcon(QIcon(resource_path("assets/attachment.png")))
                attach_btn.setIconSize(QSize(28, 28))
                attach_btn.setFixedSize(36, 36)
                attach_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: #444;
                        border-radius: 6px;
                    }
                """)
                attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                attach_card_layout.addWidget(attach_btn,alignment=Qt.AlignmentFlag.AlignLeft)



                attachment_widgets = []

                def update_tray_state():
                    attachments_tray.setVisible(bool(attachment_widgets))

                def add_attachment():
                    path, _ = QFileDialog.getOpenFileName(page, "Select Attachment")
                    if not path:
                        return

                    ext = os.path.splitext(path)[1].lower()
                    icon_path = resource_path("assets/pdf.png") if ext == ".pdf" else path
                    icon = QIcon(icon_path)

                    container = QFrame()
                    container.setFixedSize(48, 48)
                    container.setStyleSheet("background-color: rgba(255, 255, 255, 0.6); border-radius: 8px;")

                    thumb = QLabel(container)
                    thumb.setPixmap(icon.pixmap(36, 36))
                    thumb.setFixedSize(36, 36)
                    thumb.move(6, 6)

                    remove_btn = QPushButton(container)
                    remove_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                    remove_btn.setIconSize(QSize(14, 14))
                    remove_btn.setFixedSize(18, 18)
                    remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    remove_btn.setStyleSheet("""
                        QPushButton {
                            background: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(255, 0, 0, 0.6);
                            border-radius: 9px;
                        }
                    """)
                    remove_btn.move(container.width() - 18, container.height() - 18)

                    attachments_layout.addWidget(container)
                    attachment_widgets.append((container, path))
                    update_tray_state()

                    def remove_attachment():
                        attachments_layout.removeWidget(container)
                        container.deleteLater()
                        attachment_widgets.remove((container, path))
                        update_tray_state()

                    remove_btn.clicked.connect(remove_attachment)

                attach_btn.clicked.connect(add_attachment)

                send_btn = QPushButton("Send via Gmail")
                send_btn.setIcon(QIcon(resource_path("assets/send.png")))
                send_btn.setIconSize(QSize(16, 16))
                send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                send_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #0077b6;
                        color: white;
                        font-size: 8pt;
                        padding: 6px 12px;
                        border-radius: 6px;
                        text-align: left;
                    }
                    QPushButton:hover {
                        background-color: #005792;
                    }
                """)


                def send_support_message():
                    message = message_input.toPlainText().strip()
                    if not message:
                        QMessageBox.warning(page, "Empty", "Please enter a message.")
                        return

                    sender_email = email_input.text().strip() or me.get("email", "Anonymous")
                    body = f"From: {sender_email}\n\n{message}"

                    if attachment_widgets:
                        files_list = "\n".join([f"- {p}" for _, p in attachment_widgets])
                        body += f"\n\nAttachments:\n{files_list}"

                    url = (
                        "https://mail.google.com/mail/?view=cm&fs=1"
                        f"&to=fittrackerpy@gmail.com"
                        f"&su=FitnessTracker Support Request"
                        f"&body={urllib.parse.quote(body)}"
                    )
                    webbrowser.open(url)

                send_btn.clicked.connect(send_support_message)
                attach_card_layout.addWidget(send_btn, alignment=Qt.AlignmentFlag.AlignRight)

                contact_layout.addWidget(QLabel("Need help? Send us a message:"))
                contact_layout.addWidget(email_input)

                message_block = QWidget()
                message_block_layout = QVBoxLayout(message_block)
                message_block_layout.setContentsMargins(0, 0, 0, 0)
                message_block_layout.setSpacing(0)

                message_block_layout.addWidget(message_container)
                message_block_layout.addWidget(attach_card)

                contact_layout.addWidget(message_block)


                layout.addWidget(contact_card)


                faq_card = QFrame()
                faq_card.setFixedHeight(220)
                faq_card.setStyleSheet("background-color: #1e1e1e; border-radius: 16px;")
                faq_layout = QVBoxLayout(faq_card)
                faq_layout.setContentsMargins(20, 20, 20, 20)
                faq_layout.setSpacing(8)

                faq_label = QLabel("Frequently Asked Questions")
                faq_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
                faq_layout.addWidget(faq_label)

                faq_scroll = QScrollArea()
                faq_scroll.setWidgetResizable(True)
                faq_scroll.setStyleSheet("""
                    QScrollBar:vertical {
                        background: transparent;
                        width: 8px;
                    }
                    QScrollBar::handle:vertical {
                        background: #555;
                        border-radius: 4px;
                    }
                """)

                faq_content = QWidget()
                faq_content_layout = QVBoxLayout(faq_content)
                faq_content_layout.setContentsMargins(0, 0, 0, 0)
                faq_content_layout.setSpacing(8)

                faqs = [
                    ("How do I verify my email?", "Go to the Security tab and click 'Verify Email'. A link will be sent to your inbox."),
                    ("How can I reset my password?", "Use the 'Change Password' option under the Security tab."),
                    ("Why am I not receiving emails?", "Check your spam folder. Also make sure your email is correct."),
                    ("Can I attach screenshots?", "Yes, you can add image attachments to support requests."),
                    ("Where can I manage 2FA?", "Under the Security tab after verifying your email."),
                ]

                for question, answer in faqs:
                    q_label = QLabel(f"Q: {question}")
                    q_label.setStyleSheet("font-weight: bold;")
                    a_label = QLabel(f"A: {answer}")
                    a_label.setWordWrap(True)
                    faq_content_layout.addWidget(q_label)
                    faq_content_layout.addWidget(a_label)

                faq_scroll.setWidget(faq_content)
                faq_layout.addWidget(faq_scroll)
                layout.addWidget(faq_card)
                layout.addStretch()

                scroll_area.setWidget(scroll_content)
                outer_layout.addWidget(scroll_area)


                help_card = QFrame()
                help_card.setStyleSheet("""
                    background-color: #1e1e1e;
                    border-radius: 16px;
                    padding: 0px;
                """)
                help_layout = QHBoxLayout(help_card)
                help_layout.setContentsMargins(16, 8, 16, 8)
                help_layout.setSpacing(6)

                help_icon_label = QLabel()
                help_icon_label.setPixmap(QIcon(resource_path("assets/help.png")).pixmap(32, 32))
                help_icon_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

                help_text_btn = QPushButton("Open Help Documentation")
                help_text_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                help_text_btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        color: white;
                        font-size: 12pt;
                        border: none;
                        text-align: left;
                    }
                    QPushButton:hover {
                        text-decoration: underline;
                        color: #00aaff;
                    }
                """)

                def open_help():
                    help_file = resource_path("assets/help.html").replace("\\", "/")
                    url = QUrl.fromLocalFile(help_file)
                    QDesktopServices.openUrl(url)

                help_text_btn.clicked.connect(open_help)

                help_layout.addWidget(help_icon_label)
                help_layout.addWidget(help_text_btn)
                help_layout.addStretch()

                layout.addWidget(help_card)

                return page

            content_stack.addWidget(create_profile_tab())
            content_stack.addWidget(create_security_tab())
            content_stack.addWidget(create_support_tab())

            sidebar.currentRowChanged.connect(content_stack.setCurrentIndex)
            sidebar.setCurrentRow(0)

            layout.addWidget(sidebar_wrap)
            layout.addWidget(separator)
            layout.addWidget(content_stack, stretch=1)

            return tab


        self.clear_buttons(); self.clear_content_area(); self.pages = []

        self.pages.append(create_my_progress_tab())
        self.pages.append(create_trainer_tab())
        self.pages.append(create_workout_log_tab())
        self.pages.append(create_workout_schedule_tab())
        self.pages.append(create_nutrition_log_tab())
        self.pages.append(create_fitness_goals_tab())
        self.pages.append(create_sleep_tracker_tab())
        self.pages.append(create_heart_rate_monitor_tab())
        self.pages.append(create_settings_tab())


        for page in self.pages:
            self.content_area.addWidget(page)

        palette = [
            ("#245126", "My Progress"),
            ("#005662", "My Trainer"),
            ("#6C6F1A", "Workout Log"),
            ("#7A4F00", "Workout Schedule"),
            ("#732C13", "Nutrition Log"),
            ("#521A6D", "Fitness Goals"),
            ("#10416F", "Sleep Tracker"),
            ("#701031", "Heart Rate Monitor"),
            ("#2C3A40", "Settings"),
        ]





        icon_paths = [resource_path("assets/progress.png"),resource_path("assets/workoutlog.png"),
                      resource_path("assets/trainer.png"),resource_path("assets/schedule.png"),
                      resource_path("assets/nutrition.png"),resource_path("assets/goals.png"),
                      resource_path("assets/sleep.png"),resource_path("assets/heart.png"),
                      resource_path("assets/settings.png")
        ]

        for i, ((clr, txt), icon_path) in enumerate(zip(palette, icon_paths)):
            b = QPushButton(txt, self.button_row)
            b.setFont(QFont(sf_family, 10))
            b.setIcon(QIcon(icon_path))
            b.setIconSize(QSize(24, 24))
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {clr};
                    color: #fff;
                    border: none;
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                }}
                QPushButton:hover {{
                    border: 3px solid white;
                }}
            """)
            b.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
            b.setContentsMargins(0, 0, 0, 0)
            self.button_layout.addWidget(b)
            self.button_list.append(b)



            def make_handler(index, button):
                return lambda: (
                    self.animate_tab_indicator(button),
                    self.slide_to_page(index),
                    [
                        QTimer.singleShot(0, lambda: self.trainer_tab_switcher(self.last_trainer_subtab_index))
                    ] if index == 0 and hasattr(self, 'trainer_tab_switcher') else None
                )


            b.clicked.connect(make_handler(i, b))



        first = self.button_list[0]
        self.content_area.setCurrentIndex(0)
        self.current_index = 0
        first.setFocus()

        QTimer.singleShot(0, lambda: self.tab_indicator.setGeometry(
            first.x()+8, self.top_container.height() - 4, first.width(), 4
        ))


    def setup_trainer_dashboard(self):
        self.clear_buttons()
        self.clear_content_area()
        self.pages = []

        self.button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        def circular(path: str, d: int = 44) -> QPixmap:
            pm = QPixmap(path).scaled(d, d, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                    Qt.TransformationMode.SmoothTransformation)
            out = QPixmap(d, d); out.fill(Qt.GlobalColor.transparent)
            p = QPainter(out); p.setRenderHint(QPainter.RenderHint.Antialiasing)
            circle = QPainterPath(); circle.addEllipse(0, 0, d, d)
            p.setClipPath(circle); p.drawPixmap(0, 0, pm); p.end()
            return out

        def flexible_block(cards: list[QWidget]) -> QWidget:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            for card in cards:
                layout.addWidget(card)
            layout.addStretch()
            return container

        def wrap_scroll(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(widget)
            scroll.setStyleSheet("""
                QScrollArea { background-color: transparent; }
                QScrollBar:vertical { background:#1e1e2f;width:8px;border-radius:4px; }
                QScrollBar::handle:vertical { background:#555;border-radius:4px;min-height:40px; }
                QScrollBar::add-line,QScrollBar::sub-line{height:0;}
            """)
            return scroll

        def create_my_clients_tab() -> QWidget:
            class SignalEmitter(QObject):
                 message_signal = pyqtSignal(str, str, object, str)
                 typing_signal = pyqtSignal(bool)
                 trainer_list_changed_signal = pyqtSignal()
            emitter = SignalEmitter()
            previous_clients_hash = None
            previous_tobechosenclients_hash = None
            def client_card(cl: dict, pending: bool, on_add=None, on_remove=None) -> QWidget:
                f = QFrame()
                f.setFixedHeight(60)

                shadow_effect = QGraphicsDropShadowEffect()
                shadow_effect.setOffset(0, 4)
                shadow_effect.setBlurRadius(10)
                shadow_effect.setColor(Qt.GlobalColor.black)
                f.setGraphicsEffect(shadow_effect)

                f.setStyleSheet("""
                    QFrame {
                        background: rgba(76, 175, 80, 0.3);
                        border-radius: 8px;
                    }
                """)

                h = QHBoxLayout(f)
                h.setContentsMargins(12, 0, 12, 0)
                h.setSpacing(8)

                avatar = QLabel()
                avatar_path = cl.get("profile_picture", "assets/defaultprofile.png")
                avatar_full_path = get_image_path(avatar_path)
                avatar.setPixmap(QIcon(avatar_full_path).pixmap(44, 44))
                avatar.setFixedSize(44, 44)
                avatar.setStyleSheet("background: transparent;")
                h.addWidget(avatar)

                name = QLabel(f"{cl['first_name']} {cl['last_name']}")
                name.setStyleSheet("color:#fff;font-weight:600;font-size:13pt;background: transparent;")
                h.addWidget(name, 1)

                if pending:
                    tag = QLabel("Pending")
                    tag.setStyleSheet("color:#ffc107;font-weight:600;background: transparent;")
                    h.addWidget(tag)


                if pending and on_add:
                    add_btn = QPushButton()
                    add_btn.setIcon(QIcon(resource_path("assets/add.png")))
                    add_btn.setIconSize(QSize(20, 20))
                    add_btn.setFixedSize(32, 32)
                    add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    add_btn.setStyleSheet("""
                        QPushButton {
                            background-color: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(76, 175, 80, 0.3);  /* Green hover */
                            border-radius: 4px;
                        }
                    """)
                    add_btn.clicked.connect(lambda: on_add(cl))
                    h.addWidget(add_btn)


                if on_remove:
                    remove_btn = QPushButton()
                    remove_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                    remove_btn.setIconSize(QSize(20, 20))
                    remove_btn.setFixedSize(32, 32)
                    remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    remove_btn.setStyleSheet("""
                        QPushButton {
                            background-color: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(255, 82, 82, 0.3);  /* Red hover */
                            border-radius: 4px;
                        }
                    """)
                    remove_btn.clicked.connect(lambda: on_remove(cl))
                    h.addWidget(remove_btn)


                chat_btn = QPushButton("Chat")
                chat_btn.setIcon(QIcon(resource_path("assets/chat.png")))
                chat_btn.setIconSize(QSize(20, 20))
                chat_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        color: #fff;
                        padding: 10px 15px;  /* Add padding */
                        font-size: 14px;      /* Increase font size */
                        border-radius: 8px;   /* Rounded corners */
                        text-align: left;
                    }
                    QPushButton:hover {
                        background-color: rgba(0, 188, 212, 0.2);
                    }
                    QPushButton:pressed {
                        background-color: rgba(0, 188, 212, 0.4);
                    }
                """)
                chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                chat_btn.clicked.connect(lambda: open_chat_dialog(cl))
                h.addWidget(chat_btn)


                return f
            class MessageBubble(QWidget):

                long_pressed = pyqtSignal(str, str, datetime, str, QPoint)

                def __init__(self, message: str, is_my_message: bool, timestamp=None, status=None, message_id=None, edited=False):
                    super().__init__()
                    self.message = message
                    self.is_my_message = is_my_message
                    self.timestamp = timestamp
                    self.status = status
                    self.message_id = message_id
                    self.edited = edited

                    self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
                    self.opacity_effect = QGraphicsOpacityEffect(self)
                    self.opacity_effect.setOpacity(1.0)
                    self.setGraphicsEffect(self.opacity_effect)

                    self.label = QLabel(message, self)
                    self.label.setWordWrap(True)
                    self.label.setStyleSheet(f"""
                        QLabel {{
                            color: {'#FFFFFF' if is_my_message else '#D1D1D6'};
                            padding: 0px 4px;
                        }}
                    """)
                    self.label.setMaximumWidth(280)
                    self.label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


                    self.time_label = QLabel(self)
                    self.time_label.setStyleSheet("color: #ccc; font-size: 9pt;")
                    self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

                    layout = QVBoxLayout(self)
                    layout.setContentsMargins(14, 8, 14, 20)
                    layout.addWidget(self.label)
                    layout.addWidget(self.time_label, alignment=Qt.AlignmentFlag.AlignRight)


                    self.status_icon_label = None
                    self.time_text_label = None


                    if self.timestamp:
                        self.time_str = self.timestamp.strftime("%I:%M %p").lstrip("0")

                        time_container = QWidget()
                        time_layout = QHBoxLayout(time_container)
                        time_layout.setContentsMargins(0, 0, 0, 0)
                        time_layout.setSpacing(4)

                        self.time_text_label = QLabel(self.time_str)
                        self.time_text_label.setStyleSheet("color: #ccc; font-size: 9pt;")

                        time_layout.addWidget(self.time_text_label)

                        if self.is_my_message and status:
                            icon_path = "assets/seen.png" if status == "seen" else "assets/delivered.png"
                            icon = QIcon(resource_path(icon_path))
                            pixmap = icon.pixmap(16, 16)
                            self.status_icon_label = QLabel()
                            self.status_icon_label.setPixmap(pixmap)
                            time_layout.addWidget(self.status_icon_label)

                        layout.removeWidget(self.time_label)
                        self.time_label.deleteLater()
                        layout.addWidget(time_container, alignment=Qt.AlignmentFlag.AlignRight)


                    if self.edited:

                        self.edited_label = QLabel("Edited", self)
                        self.edited_label.setStyleSheet("""
                            color: #B0B0B0;
                            font-style: italic;
                            font-size: 8pt;
                        """)
                        self.edited_label.setAlignment(Qt.AlignmentFlag.AlignRight)


                        self.edited_label.setContentsMargins(0, 0, 10, 0)


                        layout.addWidget(self.edited_label)


                    self._long_press_timer = QTimer(self)
                    self._long_press_timer.setInterval(700)
                    self._long_press_timer.setSingleShot(True)
                    self._long_press_timer.timeout.connect(self._emit_long_press)
                    self._mouse_press_pos = QPoint()

                def paintEvent(self, event):
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                    rect = self.rect().adjusted(4, 0, -4, -12)

                    if self.is_my_message:
                        gradient = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomLeft()))
                        gradient.setColorAt(0, QColor("#3B82F6"))
                        gradient.setColorAt(1, QColor("#0A84FF"))
                        painter.setBrush(QBrush(gradient))
                    else:
                        gradient = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomLeft()))
                        gradient.setColorAt(0, QColor("#646464"))
                        gradient.setColorAt(1, QColor("#494949"))
                        painter.setBrush(QBrush(gradient))

                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(rect, 12, 12)

                    tail = QPolygonF()
                    if self.is_my_message:
                        base = rect.right() - 20
                        tail.append(QPointF(base, rect.bottom()))
                        tail.append(QPointF(base + 10, rect.bottom()))
                        tail.append(QPointF(base + 5, rect.bottom() + 10))
                    else:
                        base = rect.left() + 20
                        tail.append(QPointF(base, rect.bottom()))
                        tail.append(QPointF(base - 10, rect.bottom()))
                        tail.append(QPointF(base - 5, rect.bottom() + 10))

                    painter.drawPolygon(tail)

                def set_status(self, status):
                    self.status = status
                    if self.is_my_message and self.status_icon_label:
                        icon_path = "assets/seen.png" if status == "seen" else "assets/delivered.png"
                        icon = QIcon(resource_path(icon_path))
                        pixmap = icon.pixmap(16, 16)
                        self.status_icon_label.setPixmap(pixmap)

                def mousePressEvent(self, event):
                    if event.button() == Qt.MouseButton.LeftButton:

                        self.opacity_effect.setOpacity(0.85)
                        self._mouse_press_pos = event.globalPosition()
                        self._long_press_timer.start()
                    super().mousePressEvent(event)

                def mouseReleaseEvent(self, event):
                    if event.button() == Qt.MouseButton.LeftButton:

                        self.opacity_effect.setOpacity(1.0)
                        self._long_press_timer.stop()
                    super().mouseReleaseEvent(event)

                def mouseMoveEvent(self, event):

                    if self._long_press_timer.isActive() and \
                            (event.globalPosition() - self._mouse_press_pos).manhattanLength() > 10:
                        self._long_press_timer.stop()
                    super().mouseMoveEvent(event)

                def _emit_long_press(self):

                    sender = "me" if self.is_my_message else "other"
                    self.long_pressed.emit(self.message, sender, self.timestamp, self.status, self.mapToGlobal(self.rect().center()))
                    self.opacity_effect.setOpacity(1.0)

                def update_message_text(self, new_text):
                    self.message = new_text
                    self.label.setText(new_text)
                    self.label.adjustSize()


                    if self.edited:

                        if not hasattr(self, 'edited_label'):
                            self.edited_label = QLabel("Edited", self)
                            self.edited_label.setStyleSheet("""
                                color: #B0B0B0;
                                font-style: italic;
                                font-size: 8pt;
                            """)
                            self.edited_label.setAlignment(Qt.AlignmentFlag.AlignRight)


                            self.edited_label.setContentsMargins(0, 0, 10, 0)
                            self.layout().addWidget(self.edited_label)
                    else:

                        if hasattr(self, 'edited_label'):
                            self.edited_label.deleteLater()
                            del self.edited_label



                    self.parentWidget().layout().invalidate()
                    self.update()

            def open_chat_dialog(receiver):
                emitter = SignalEmitter()
                last_message_date = None

                message_widgets = {}
                message_data_map = {}

                me_username = self.user_data.get("username")
                receiver_username = receiver["username"]
                chat_id = "_".join(sorted([me_username, receiver_username]))
                messages_ref = db.collection("chats").document(chat_id).collection("messages")

                dlg = QDialog(self)
                dlg.setWindowTitle(f"Chat with {receiver['first_name']} {receiver['last_name']}")
                dlg.setFixedSize(400, 600)
                dlg.setStyleSheet("background:#2e2e3e; color:#fff; border:none;")
                dlg.setWindowIcon(QIcon(resource_path("assets/chat.png")))

                v = QVBoxLayout(dlg)
                v.setContentsMargins(12, 12, 12, 12)
                v.setSpacing(10)

                message_box = QScrollArea()
                message_box.setWidgetResizable(True)
                message_box.setStyleSheet("""
                    QScrollArea {
                        border: none;
                        background: transparent;
                    }
                    QScrollBar:vertical {
                        background: transparent;
                        width: 8px;
                        margin: 0px 0px 0px 0px;
                    }
                    QScrollBar::handle:vertical {
                        background: #888;
                        min-height: 20px;
                        border-radius: 4px;
                    }
                    QScrollBar::handle:vertical:hover {
                        background: #555;
                    }
                    QScrollBar::add-line, QScrollBar::sub-line {
                        height: 0px;
                    }
                    QScrollBar::add-page, QScrollBar::sub-page {
                        background: none;
                    }
                """)

                message_area = QWidget()
                message_layout = QVBoxLayout(message_area)
                message_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
                message_area.setStyleSheet("background: transparent; border: none;")
                message_box.setWidget(message_area)

                v.addWidget(message_box)
                typing_label = QLabel("")
                typing_label.setStyleSheet("color: #aaa; font-size: 12px; margin-left: 6px;")
                typing_label.setVisible(False)
                v.addWidget(typing_label,alignment=Qt.AlignmentFlag.AlignLeft)
                typing_status_ref = db.collection("chats").document(chat_id).collection("status").document(me_username)
                typing_timer = QTimer()
                typing_timer.setInterval(1500)
                typing_timer.setSingleShot(True)
                wave_timer = QTimer()
                wave_timer.setInterval(300)
                wave_frame = 0

                dot_wave_frames = [
                    ". . .",
                    "• . .",
                    ". • .",
                    ". . •",
                    ". • .",
                    "• . .",
                ]

                message_input = QLineEdit()
                message_input.setPlaceholderText("Type your message...")
                message_input.setStyleSheet("""
                    QLineEdit {
                        background: #1e1e2f;
                        border: none;
                        border-radius: 6px;
                        padding: 8px 12px;
                        color: #fff;
                        font-size: 14px;
                    }
                """)

                send_btn = QPushButton()
                send_btn.setIcon(QIcon(resource_path("assets/send4.png")))
                send_btn.setIconSize(QSize(32, 32))
                send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                send_btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        border: none;
                        padding: 3px;
                    }
                    QPushButton:hover {
                        background: rgba(255, 255, 255, 0.08);
                        border-radius: 12px;
                    }
                """)


                local_messages = []

                def hyphenate_text(text, font, max_width):
                    metrics = QFontMetrics(font)
                    words = text.split()
                    result = []
                    padding_correction = 12 * 2

                    for word in words:
                        if metrics.horizontalAdvance(word) + padding_correction <= max_width:
                            result.append(word)
                        else:
                            split_word = ""
                            current = ""
                            for char in word:
                                if metrics.horizontalAdvance(current + char + '-') + padding_correction > max_width:
                                    split_word += current + "-\n"
                                    current = char
                                else:
                                    current += char
                            split_word += current
                            result.append(split_word)
                    return " ".join(result)

                class DateLabel(QLabel):
                    def __init__(self, text, parent=None):
                        super().__init__(text, parent)
                        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.setFont(QFont(sf_family, 10, QFont.Weight.Medium))
                        self.setStyleSheet("color: #fff; padding: 4px 12px;")

                        fm = QFontMetrics(self.font())
                        text_width = fm.horizontalAdvance(text)
                        self.setFixedSize(text_width + 24, fm.height() + 12)

                    def paintEvent(self, event):
                        painter = QPainter(self)
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                        rect = self.rect().adjusted(0, 0, 0, 0)

                        gradient = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomLeft()))
                        gradient.setColorAt(0, QColor("#5c5c5c"))
                        gradient.setColorAt(1, QColor("#3a3a3a"))

                        painter.setBrush(QBrush(gradient))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawRoundedRect(rect, 10, 10)

                        super().paintEvent(event)


                def create_date_label(text):
                    label = DateLabel(text)
                    container = QWidget()
                    layout = QHBoxLayout(container)
                    layout.setContentsMargins(0, 10, 0, 10)
                    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(label)
                    return container

                def handle_message_long_press(message_text, is_my_message, message_timestamp, current_status, global_pos):

                    if not is_my_message:
                        return

                    menu = QMenu()


                    edit_icon = QIcon(resource_path("assets/edit4.png"))
                    delete_icon = QIcon(resource_path("assets/delete.png"))


                    edit_action = menu.addAction(edit_icon, "Edit Message")
                    delete_action = menu.addAction(delete_icon, "Delete Message")


                    action = menu.exec(global_pos)

                    if action == edit_action:
                        edit_message(message_text, message_timestamp, me_username, chat_id)
                    elif action == delete_action:
                        delete_message(message_text, message_timestamp, me_username, chat_id)

                def create_message_bubble(message: str, is_my_message: bool, timestamp: datetime, status=None, message_id=None, edited=False):
                    font = QFont()
                    display_message = hyphenate_text(message, font, 280)

                    bubble = MessageBubble(display_message, is_my_message, timestamp, status, message_id, edited)


                    bubble.long_pressed.connect(handle_message_long_press)

                    container = QWidget()
                    layout = QHBoxLayout(container)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(0)

                    if is_my_message:
                        layout.addStretch()
                        layout.addWidget(bubble)
                    else:
                        layout.addWidget(bubble)
                        layout.addStretch()

                    return container, bubble



                def update_ui(text, sender, timestamp, status="delivered", message_id=None, edited=False):
                    nonlocal last_message_date, message_widgets, message_data_map

                    if message_layout is None or message_layout.parent() is None:
                        return


                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)

                    local_tz = timezone(timedelta(hours=4))
                    local_timestamp = timestamp.astimezone(local_tz)

                    msg_date = local_timestamp.date()
                    label_text = format_date_label(local_timestamp)

                    if message_id is None:
                        actual_key = (local_timestamp.isoformat(), sender, text)
                    else:
                        actual_key = message_id

                    if actual_key in message_widgets:
                        bubble = message_widgets[actual_key]

                        if bubble.status != status:
                            bubble.set_status(status)


                        if bubble.message != text:
                            bubble.edited = edited
                            bubble.update_message_text(text)
                            if actual_key in message_data_map:
                                message_data_map[actual_key]["text"] = text
                                message_data_map[actual_key]["edited"] = edited
                        if actual_key in message_data_map:
                            message_data_map[actual_key]["status"] = status
                    else:
                        if last_message_date != msg_date:
                            date_label = create_date_label(label_text)
                            message_layout.addWidget(date_label)
                            last_message_date = msg_date

                        is_me = sender == me_username
                        print(edited)
                        msg_container, bubble = create_message_bubble(text, is_me, local_timestamp, status, message_id, edited)

                        message_widgets[actual_key] = bubble
                        message_data_map[actual_key] = {
                            "text": text,
                            "sender": sender,
                            "timestamp": local_timestamp.isoformat(),
                            "status": status,
                            "message_id": message_id,
                            "edited": edited
                        }

                        message_layout.addWidget(msg_container)

                    QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))




                local_messages = load_local_chat(me_username, receiver_username)
                for msg in local_messages:
                    try:
                        ts = datetime.fromisoformat(msg["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except Exception:
                        ts = datetime.now(timezone.utc)

                    update_ui(msg["text"], msg["sender"], ts, msg.get("status", "delivered"), msg.get("message_id"),msg.get("edited"))

                QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))


                last_local_ts = None
                if local_messages:
                    try:

                        last_local_ts = max(datetime.fromisoformat(m["timestamp"]) for m in local_messages)
                    except Exception:
                        last_local_ts = None

                def load_missing_messages():
                    query = messages_ref.order_by("timestamp")
                    if last_local_ts:
                        query = query.start_after([last_local_ts])

                    docs = query.get()
                    for doc in docs:
                        doc_dict = doc.to_dict()
                        message_id = doc.id

                        text = doc_dict.get("text", "")
                        sender = doc_dict.get("sender", "")
                        status = doc_dict.get("status", "delivered")
                        timestamp = doc_dict.get("timestamp")
                        edited = doc_dict.get("edited", False)

                        if hasattr(timestamp, "ToDatetime"):
                            timestamp = timestamp.ToDatetime()
                        elif isinstance(timestamp, dict) and "_seconds" in timestamp:
                            timestamp = datetime.fromtimestamp(timestamp["_seconds"], tz=timezone.utc)
                        elif timestamp is None:
                            timestamp = datetime.now(timezone.utc)

                        iso_ts = timestamp.isoformat()
                        key = (iso_ts, sender, text)


                        exists_locally = any(
                            m.get("timestamp") == iso_ts and m.get("text") == text and m.get("sender") == sender
                            for m in local_messages
                        )
                        if not exists_locally:
                            new_msg = {
                                "text": text,
                                "sender": sender,
                                "timestamp": iso_ts,
                                "status": status,
                                "message_id": message_id,
                                "edited": edited
                            }
                            local_messages.append(new_msg)
                            save_local_chat(me_username, receiver_username, local_messages)
                            update_ui(text, sender, timestamp, status, message_id, edited)

                    QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))

                load_missing_messages()

                def mark_seen_for_incoming_messages():
                    updated = False
                    for msg in local_messages:
                        if msg["sender"] != me_username and msg.get("status") != "seen":
                            msg["status"] = "seen"
                            updated = True


                            try:

                                if "message_id" in msg and msg["message_id"]:
                                    messages_ref.document(msg["message_id"]).update({"status": "seen"})
                                else:

                                    query = messages_ref \
                                        .where(filter=FieldFilter("timestamp", "==", datetime.fromisoformat(msg["timestamp"]))) \
                                        .where(filter=FieldFilter("sender", "==", msg["sender"])) \
                                        .get()

                                    for doc in query:
                                        try:
                                            messages_ref.document(doc.id).update({"status": "seen"})
                                        except Exception as e:
                                            print("Failed to update status for doc:", doc.id, "Error:", e)

                            except Exception as outer_e:
                                print("Failed to query/update messages for status update:", outer_e)

                    if updated:
                        save_local_chat(me_username, receiver_username, local_messages)

                mark_seen_for_incoming_messages()


                def send_message():
                    message = message_input.text().strip()
                    if message:
                        utc_now = datetime.now(timezone.utc)


                        try:
                            doc_ref = messages_ref.add({
                                "text": message,
                                "sender": me_username,
                                "timestamp": utc_now,
                                "status": "delivered"
                            })
                            message_id = doc_ref[1].id
                        except Exception as e:
                            print(f"Error adding message to Firebase: {e}")
                            message_id = None

                        msg_data = {
                            "text": message,
                            "sender": me_username,
                            "timestamp": utc_now.isoformat(),
                            "status": "delivered",
                            "message_id": message_id,
                            "edited": False
                        }

                        local_messages.append(msg_data)
                        save_local_chat(me_username, receiver_username, local_messages)


                        update_ui(message, me_username, utc_now, "delivered", message_id,False)
                        QTimer.singleShot(50, lambda: message_box.verticalScrollBar().setValue(message_box.verticalScrollBar().maximum()))

                        message_input.clear()

                def send_typing_status():
                    try:
                        typing_status_ref.set({"typing": True}, merge=True)
                        typing_timer.start()
                    except Exception as e:
                        print("Failed to send typing status:", e)

                def stop_typing_status():
                    try:
                        typing_status_ref.set({"typing": False}, merge=True)
                    except Exception as e:
                        print("Failed to stop typing status:", e)

                typing_timer.timeout.connect(stop_typing_status)

                def handle_user_typing():
                    if not typing_timer.isActive():
                        send_typing_status()
                    typing_timer.start()

                message_input.textChanged.connect(handle_user_typing)

                send_btn.clicked.connect(send_message)

                input_layout = QHBoxLayout()
                input_layout.setContentsMargins(0, 0, 0, 0)
                input_layout.setSpacing(6)
                input_layout.addWidget(message_input)
                input_layout.addWidget(send_btn)

                v.addLayout(input_layout)


                listener_unsubscribe = None
                typing_unsubscribe = None

                def handle_typing(is_typing):
                    nonlocal wave_frame
                    if is_typing:
                        typing_label.setVisible(True)
                        def update_wave():
                            nonlocal wave_frame
                            dots = dot_wave_frames[wave_frame % len(dot_wave_frames)]
                            typing_label.setText(f"{receiver['first_name']} is typing {dots}")
                            wave_frame += 1

                        try:
                            wave_timer.timeout.disconnect()
                        except TypeError:
                            pass
                        wave_timer.timeout.connect(update_wave)
                        wave_timer.start()
                    else:
                        wave_timer.stop()
                        typing_label.setText("")
                        typing_label.setVisible(False)

                def listen_to_messages():
                    def on_snapshot(col_snapshot, changes, read_time):
                        for change in changes:

                            doc = change.document.to_dict()
                            message_id = change.document.id

                            if not doc:
                                continue

                            text = doc.get("text", "")
                            sender = doc.get("sender", "")
                            status = doc.get("status", "delivered")
                            timestamp = doc.get("timestamp")
                            edited = doc.get("edited", False)


                            if hasattr(timestamp, "ToDatetime"):
                                timestamp = timestamp.ToDatetime()
                            elif isinstance(timestamp, dict) and "_seconds" in timestamp:
                                timestamp = datetime.fromtimestamp(timestamp["_seconds"], tz=timezone.utc)
                            elif timestamp is None:
                                timestamp = datetime.now(timezone.utc)

                            iso_ts = timestamp.isoformat()
                            key = (iso_ts, sender, text)


                            if 'type' in dir(change):
                                if change.type == 'ADDED':

                                    exists_locally = any(
                                        m.get("timestamp") == iso_ts and m.get("sender") == sender and m.get("text") == text
                                        for m in local_messages
                                    )
                                    if not exists_locally:
                                        local_messages.append({
                                            "text": text,
                                            "sender": sender,
                                            "timestamp": iso_ts,
                                            "status": status,
                                            "message_id": message_id,
                                            "edited": edited
                                        })
                                        save_local_chat(me_username, receiver_username, local_messages)
                                        emitter.message_signal.emit(text, sender, timestamp, status, message_id, edited)
                                        if sender != me_username:
                                            QTimer.singleShot(0, mark_seen_for_incoming_messages)

                                elif change.type == 'MODIFIED':

                                    for i, msg in enumerate(local_messages):
                                        if msg.get("message_id") == message_id:
                                            local_messages[i]["text"] = text
                                            local_messages[i]["status"] = status
                                            save_local_chat(me_username, receiver_username, local_messages)

                                            emitter.message_signal.emit(text, sender, timestamp, status, message_id, edited)
                                            break

                                    for ui_key, bubble_widget in message_widgets.items():
                                        if ui_key[0] == iso_ts and ui_key[1] == sender and ui_key[2] == bubble_widget.message:
                                            bubble_widget.update_message_text(text)
                                            bubble_widget.set_status(status)
                                            break

                                elif change.type == 'REMOVED':
                                    message_id = change.document.id


                                    local_messages[:] = [msg for msg in local_messages if msg.get("message_id") != message_id]
                                    save_local_chat(me_username, receiver_username, local_messages)


                                    if message_id in message_data_map:
                                        del message_data_map[message_id]

                                    if message_id in message_widgets:
                                        bubble_widget = message_widgets.pop(message_id)
                                        if bubble_widget:
                                            container_widget = bubble_widget.parentWidget().parentWidget()
                                            if container_widget:
                                                message_layout.removeWidget(container_widget)
                                                container_widget.setParent(None)
                                                container_widget.deleteLater()


                            if sender == receiver_username:
                                QTimer.singleShot(0, lambda: wave_timer.stop())
                                QTimer.singleShot(0, lambda: typing_label.setText(""))
                                QTimer.singleShot(0, lambda: typing_label.setVisible(False))

                    return messages_ref.order_by("timestamp").on_snapshot(on_snapshot)

                def listen_to_typing_status():
                    other_user_status_ref = db.collection("chats").document(chat_id).collection("status").document(receiver_username)

                    def on_status_snapshot(doc_snapshot, changes, read_time):
                        for doc in doc_snapshot:
                            data = doc.to_dict()
                            if data and data.get("typing", False):
                                emitter.typing_signal.emit(True)
                            else:
                                emitter.typing_signal.emit(False)

                    return other_user_status_ref.on_snapshot(on_status_snapshot)






                def edit_message(original_text, timestamp, sender, chat_id):
                    current_msg_data = None
                    message_id = None


                    for i, msg in enumerate(local_messages):
                        if msg.get("text") == original_text and msg.get("sender") == sender:
                            current_msg_data = msg
                            message_id = msg.get("message_id")
                            break

                    if not current_msg_data:
                        QMessageBox.warning(dlg, "Error", "Could not find message to edit locally.")
                        return


                    new_text, ok = QInputDialog.getText(dlg, "Edit Message", "Enter new message:",
                                                        QLineEdit.EchoMode.Normal, original_text)

                    if ok and new_text and new_text != original_text:
                        if current_msg_data:

                            current_msg_data["text"] = new_text
                            current_msg_data["edited"] = True

                            message_data_map[message_id]["text"] = new_text


                            print("setting edited to true")
                            message_data_map[message_id]["edited"] = True


                            update_ui(new_text, sender, timestamp, current_msg_data["status"], message_id,edited = True)


                            save_local_chat(me_username, receiver_username, local_messages)


                        try:
                            if message_id:

                                messages_ref.document(message_id).update({
                                    "text": new_text,
                                    "edited": True
                                })
                            else:

                                query = messages_ref \
                                    .where(filter=FieldFilter("timestamp", "==", timestamp)) \
                                    .where(filter=FieldFilter("sender", "==", sender)) \
                                    .get()
                                for doc in query:
                                    messages_ref.document(doc.id).update({
                                        "text": new_text,
                                        "edited": True
                                    })
                                    break
                        except Exception as e:
                            QMessageBox.critical(dlg, "Firebase Error", f"Failed to edit message in Firebase: {e}")




                def delete_message(message_text, timestamp, sender, chat_id):
                    reply = QMessageBox.question(
                        dlg,
                        "Delete Message",
                        "Are you sure you want to delete this message?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )

                    if reply == QMessageBox.StandardButton.Yes:
                        message_id_to_delete = None


                        for msg in local_messages:

                            if msg.get("text") == message_text and msg.get("sender") == sender:
                                message_id_to_delete = msg.get("message_id")
                                break

                        if not message_id_to_delete:
                            print("Error: Could not find message_id to delete.")
                            return




                        if message_id_to_delete in message_data_map:
                            del message_data_map[message_id_to_delete]

                        if message_id_to_delete in message_widgets:
                            bubble_widget = message_widgets.pop(message_id_to_delete)
                            if bubble_widget:

                                container_widget = bubble_widget.parentWidget().parentWidget()
                                if container_widget:
                                    message_layout.removeWidget(container_widget)
                                    container_widget.setParent(None)
                                    container_widget.deleteLater()


                        local_messages[:] = [msg for msg in local_messages if msg.get("message_id") != message_id_to_delete]
                        save_local_chat(me_username, receiver_username, local_messages)


                        try:
                            messages_ref.document(message_id_to_delete).delete()
                        except Exception as e:
                            QMessageBox.critical(dlg, "Firebase Error", f"Failed to delete message from Firebase: {e}")




                emitter.message_signal.connect(update_ui)
                emitter.typing_signal.connect(handle_typing)
                listener_unsubscribe = listen_to_messages()
                typing_unsubscribe = listen_to_typing_status()

                def cleanup():
                    emitter.message_signal.disconnect(update_ui)
                    wave_timer.stop()
                    if listener_unsubscribe:
                        listener_unsubscribe.unsubscribe()
                    if typing_unsubscribe:
                        typing_unsubscribe.unsubscribe()
                    try:
                        typing_status_ref.set({"typing": False}, merge=True)
                    except:
                        pass

                dlg.finished.connect(cleanup)
                dlg.exec()



            def top_aligned_label(text: str) -> QWidget:
                lbl = QLabel(text)
                lbl.setStyleSheet("color:gray;font-size:13pt;")
                lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
                wrapper = QWidget()
                v = QVBoxLayout(wrapper)
                v.setContentsMargins(10, 10, 10, 10)
                v.setAlignment(Qt.AlignmentFlag.AlignTop)
                v.addWidget(lbl)
                return wrapper

            page = QWidget()
            root = QVBoxLayout(page)
            root.setContentsMargins(20, 20, 20, 20)
            root.setSpacing(16)
            root.setAlignment(Qt.AlignmentFlag.AlignTop)
            page.setStyleSheet("background:#2e2e3e;color:#fff;")

            title = QLabel("My Clients")
            title.setFont(QFont(sf_family, 20, QFont.Weight.Bold))
            root.addWidget(title)

            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.setSpacing(0)

            tab_buttons = QWidget()
            tab_buttons.setStyleSheet("""
                border-bottom: 1px solid #444;
            """)
            tab_btn_layout = QHBoxLayout(tab_buttons)
            tab_btn_layout.setContentsMargins(0, 0, 0, 0)
            tab_btn_layout.setSpacing(0)
            tab_btn_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

            pending_btn = QPushButton("Pending Clients")
            accepted_btn = QPushButton("Accepted Clients")
            for btn in (pending_btn, accepted_btn):
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        color: #ccc;
                        font-weight: 600;
                        padding: 8px 14px;
                        border: none;
                    }
                    QPushButton:hover {
                        color: #fff;
                    }
                """)
            tab_btn_layout.addWidget(pending_btn)
            tab_btn_layout.addWidget(accepted_btn)


            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            clients = [u for u in users if u.get("role", "").lower() == "user"]

            def compute_hash(lst):
                """Helper function to compute hash of a list."""
                return hashlib.sha256(str(lst).encode('utf-8')).hexdigest()

            def detect_client_list_changes():
                nonlocal previous_clients_hash, previous_tobechosenclients_hash


                current_clients = me.get("clients", [])
                current_tobechosenclients = me.get("tobechosenclients", [])


                current_clients_hash = compute_hash(current_clients)
                current_tobechosenclients_hash = compute_hash(current_tobechosenclients)


                if current_clients_hash != previous_clients_hash or current_tobechosenclients_hash != previous_tobechosenclients_hash:

                    emitter.trainer_list_changed_signal.emit()


                    previous_clients_hash = current_clients_hash
                    previous_tobechosenclients_hash = current_tobechosenclients_hash


            def setup_change_detection_timer():
                timer = QTimer()
                timer.setInterval(500)
                timer.timeout.connect(detect_client_list_changes)
                timer.start()
            def add_client(cl):
                if cl["username"] not in me.get("clients", []):
                    me.setdefault("clients", []).append(cl["username"])
                if cl["username"] in me.get("tobechosenclients", []):
                    me["tobechosenclients"].remove(cl["username"])

                cl.setdefault("trainers", [])
                if me["username"] not in cl["trainers"]:
                    cl["trainers"].append(me["username"])

                if me["username"] in cl.get("chosentrainers", []):
                    cl["chosentrainers"].remove(me["username"])

                save_user_data()
                refresh_blocks(stay_on_index=0)

            def remove_pending_client(cl):
                if cl["username"] in me.get("tobechosenclients", []):
                    me["tobechosenclients"].remove(cl["username"])
                if me["username"] in cl.get("chosentrainers", []):
                    cl["chosentrainers"].remove(me["username"])
                save_user_data()
                refresh_blocks(stay_on_index=0)

            def remove_accepted_client(cl):
                if cl["username"] in me.get("clients", []):
                    me["clients"].remove(cl["username"])
                if me["username"] in cl.get("trainers", []):
                    cl["trainers"].remove(me["username"])
                save_user_data()
                refresh_blocks(stay_on_index=1)

            def get_client_blocks():
                pending_usernames = [u for u in me.get("tobechosenclients", []) if u not in me.get("clients", [])]
                accepted_usernames = me.get("clients", [])

                pending = [c for c in clients if c["username"] in pending_usernames]
                accepted = [c for c in clients if c["username"] in accepted_usernames]

                pb = wrap_scroll(flexible_block([
                    client_card(c, True, on_add=add_client, on_remove=remove_pending_client) for c in pending
                ])) if pending else top_aligned_label("No pending clients.")

                ab = wrap_scroll(flexible_block([
                    client_card(c, False, on_remove=remove_accepted_client) for c in accepted
                ])) if accepted else top_aligned_label("No accepted clients.")

                return pb, ab

            def save_user_data():
                try:
                    Path("user_data.json").write_text(json.dumps(users, indent=4))
                except Exception as e:
                    print("Failed saving user data:", e)

            def refresh_blocks(stay_on_index):
                nonlocal pending_block, accepted_block
                new_pending, new_accepted = get_client_blocks()
                stack.removeWidget(pending_block); pending_block.deleteLater()
                stack.removeWidget(accepted_block); accepted_block.deleteLater()
                stack.insertWidget(0, new_pending)
                stack.insertWidget(1, new_accepted)
                pending_block, accepted_block = new_pending, new_accepted
                stack.setCurrentIndex(stay_on_index)
                switch_tab(stay_on_index)

            pending_block, accepted_block = get_client_blocks()

            stack_container = QWidget()
            stack = QStackedLayout(stack_container)
            stack.setContentsMargins(0, 0, 0, 0)
            stack.addWidget(pending_block)
            stack.addWidget(accepted_block)

            tab_layout.addWidget(tab_buttons)
            tab_layout.addSpacing(12)
            tab_layout.addWidget(stack_container)



            def switch_tab(index: int):
                stack.setCurrentIndex(index)
                self.last_client_subtab_index = index

                for i, btn in enumerate((pending_btn, accepted_btn)):
                    if not isinstance(btn, QPushButton):
                        continue
                    if i == index:
                        btn.setStyleSheet("""
                            QPushButton {
                                background: transparent;
                                color: white;
                                font-weight: 600;
                                font-size: 14pt;
                                padding: 10px 18px;
                                border: none;
                                border-bottom: 5px solid #ffffff;
                            }
                        """)
                    else:
                        btn.setStyleSheet("""
                            QPushButton {
                                background: transparent;
                                color: #aaa;
                                font-weight: 500;
                                font-size: 13pt;
                                padding: 10px 18px;
                                border: none;
                            }
                            QPushButton:hover {
                                color: white;
                            }
                        """)


            pending_btn.clicked.connect(lambda: switch_tab(0))
            accepted_btn.clicked.connect(lambda: switch_tab(1))

            self.last_client_subtab_index = 0
            self.client_tab_switcher = switch_tab

            panel = QWidget()
            plo = QVBoxLayout(panel)
            plo.setContentsMargins(16, 0, 16, 16)
            plo.setSpacing(6)
            plo.setAlignment(Qt.AlignmentFlag.AlignTop)
            plo.addWidget(tab_widget)
            root.addWidget(panel)

            switch_tab(self.last_client_subtab_index)
            emitter.trainer_list_changed_signal.connect(refresh_blocks)
            setup_change_detection_timer()
            return page


        def create_settings_tab() -> QWidget:
            try:
                users = json.loads(Path("user_data.json").read_text())
            except Exception as e:
                print("user_data.json:", e)
                users = []

            me = next((u for u in users if u.get("username") == self.user_data.get("username")), None)
            if me is None:
                me = {}

            tab = QWidget()
            layout = QHBoxLayout(tab)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)


            sidebar = QListWidget()
            sidebar.setFixedWidth(280)
            sidebar.setStyleSheet("""
                QListWidget {
                    background-color: #121212;
                    border: none;
                    padding: 16px;
                    outline: none;
                }
                QListWidget::item {
                    padding: 16px 20px;
                    margin-bottom: 12px;
                    font-size: 18px;
                    border-radius: 14px;
                    color: #ccc;
                    background-color: #1e1e1e;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
                QListWidget::item:selected {
                    background-color: #005792;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
                QListWidget::item:focus {
                    outline: none;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
                QListWidget::item:selected:!active {
                    outline: none;
                    border: none;
                    border-bottom: 1px solid #ffffff22;
                }
            """)

            sidebar_items = [
                ("Profile", "assets/profile.png"),
                ("Security", "assets/security.png"),
                ("Support", "assets/support.png"),
            ]

            for name, icon in sidebar_items:
                item = QListWidgetItem(QIcon(resource_path(icon)), f"  {name}")
                sidebar.addItem(item)

            def delete_account():
                confirm = QMessageBox()
                confirm.setIcon(QMessageBox.Icon.Warning)
                confirm.setWindowIcon(QIcon(resource_path("assets/delete.png")))
                confirm.setWindowTitle("Confirm Deletion")
                confirm.setText("Are you sure you want to delete your account? This action cannot be undone.")
                confirm.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)

                if confirm.exec() == QMessageBox.StandardButton.Ok:
                    try:
                        users = json.loads(Path("user_data.json").read_text())
                        updated_users = [u for u in users if u.get("username") != self.user_data.get("username")]
                        Path("user_data.json").write_text(json.dumps(updated_users, indent=2))


                        self.user_data.clear()
                        self.parent().login_page.reset_fields()
                        self.parent().setCurrentIndex(0)
                    except Exception as e:
                        error_dialog = QMessageBox()
                        error_dialog.setIcon(QMessageBox.Icon.Critical)
                        error_dialog.setWindowTitle("Error")
                        error_dialog.setText(f"Failed to delete account.\n\n{str(e)}")
                        error_dialog.exec()


            def logout():
                self.parent().login_page.reset_fields()
                self.parent().setCurrentIndex(0)


            logout_btn = QWidget()
            logout_btn.setFixedSize(260, 50)
            logout_btn.setStyleSheet("""
                background-color: #d32f2f;
                border-radius: 12px;
            """)
            logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            logout_layout = QHBoxLayout(logout_btn)
            logout_layout.setContentsMargins(16, 0, 16, 0)
            logout_layout.setSpacing(12)

            label = QLabel("Log Out")
            label.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            icon_label = QLabel()
            icon_label.setPixmap(QIcon(resource_path("assets/logout.png")).pixmap(24, 24))
            icon_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            logout_layout.addWidget(label)
            logout_layout.addStretch()
            logout_layout.addWidget(icon_label)


            logout_btn.mousePressEvent = lambda e: logout()



            sidebar_layout = QVBoxLayout()
            sidebar_layout.setContentsMargins(12, 12, 12, 12)
            sidebar_layout.setSpacing(16)
            sidebar_layout.addWidget(sidebar)
            sidebar_layout.addStretch()
            sidebar_layout.addWidget(logout_btn)

            sidebar_wrap = QFrame()
            sidebar_wrap.setLayout(sidebar_layout)
            sidebar_wrap.setStyleSheet("background-color: #121212;border: none; ")


            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.VLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            separator.setStyleSheet("color: white;")


            content_stack = QStackedWidget()
            content_stack.setStyleSheet("background-color: #121212;")

            def create_profile_tab():
                page = QWidget()
                layout = QVBoxLayout(page)
                layout.setContentsMargins(24, 24, 24, 24)
                layout.setSpacing(24)
                page.setStyleSheet("background-color: #121212;")


                profile_card = QFrame()
                profile_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 20px;
                        border: none;
                    }
                """)
                profile_card_layout = QVBoxLayout(profile_card)
                profile_card_layout.setContentsMargins(20, 20, 20, 20)
                profile_card_layout.setSpacing(12)


                username = me.get("username", "default_user")
                profile_pic_widget = ProfilePicWithEdit(username, me)
                profile_card_layout.addWidget(profile_pic_widget, alignment=Qt.AlignmentFlag.AlignHCenter)



                full_name = f"{me.get('first_name', '')} {me.get('last_name', '')}".strip() or "Unnamed"
                name_editable = EditableLabel(full_name, me)
                profile_card_layout.addWidget(name_editable, alignment=Qt.AlignmentFlag.AlignHCenter)

                layout.addWidget(profile_card)


                contact_card = QFrame()
                contact_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 20px;
                        border: none;
                    }
                """)
                contact_layout = QVBoxLayout(contact_card)
                contact_layout.setContentsMargins(20, 16, 20, 16)
                contact_layout.setSpacing(24)

                email = me.get("email", "Not set")
                phone = me.get("phone", "Not provided")
                contact_layout.addWidget(EditableContactRow("assets/email.png", "Email:", email, me, "email"))
                contact_layout.addWidget(EditableContactRow("assets/phone.png", "Phone:", phone, me, "phone"))

                layout.addWidget(contact_card)


                layout.addStretch()


                delete_row = QHBoxLayout()
                delete_row.addStretch()
                delete_btn = QPushButton("Delete Account")
                delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                delete_btn.setIcon(QIcon(resource_path("assets/trash.png")))
                delete_btn.setIconSize(QSize(16, 16))
                delete_btn.setFixedHeight(36)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #d32f2f;
                        color: white;
                        font-size: 12pt;
                        border-radius: 8px;
                        padding: 6px 16px;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: #b71c1c;
                    }
                """)
                delete_row.addWidget(delete_btn)
                delete_btn.clicked.connect(delete_account)
                layout.addLayout(delete_row)

                return page


            def create_security_tab():
                page = QWidget()
                layout = QVBoxLayout(page)
                layout.setContentsMargins(24, 24, 24, 24)
                layout.setSpacing(24)
                page.setStyleSheet("background-color: #121212;")


                title = QLabel("Security Settings")
                title.setStyleSheet("color: white; font-size: 20pt; font-weight: bold;")
                layout.addWidget(title)


                if not me.get("verified", False):
                    verify_card = QFrame()
                    verify_card.setStyleSheet("""
                        QFrame {
                            background-color: #1e1e1e;
                            border-radius: 16px;
                        }
                    """)
                    verify_layout = QVBoxLayout(verify_card)
                    verify_layout.setContentsMargins(20, 20, 20, 20)
                    verify_layout.setSpacing(12)

                    info = QLabel("Verify your email to enable 2FA and improve your account security.")
                    info.setWordWrap(True)
                    info.setStyleSheet("color: white; font-size: 12pt;")
                    verify_layout.addWidget(info)

                    verify_btn = QPushButton("Verify Email")
                    verify_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    verify_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #00897b;
                            color: white;
                            font-size: 13pt;
                            padding: 10px;
                            border-radius: 10px;
                        }
                        QPushButton:hover {
                            background-color: #00695c;
                        }
                    """)
                    verify_layout.addWidget(verify_btn, alignment=Qt.AlignmentFlag.AlignRight)

                    def verify_email():
                        success = send_verification_email(me["email"], me["username"])
                        if success:
                            QMessageBox.information(page, "Verification Sent",
                                "Verification email sent! Please check your inbox and click the link to verify your account.")
                        else:
                            QMessageBox.critical(page, "Error", "Failed to send verification email. Please try again.")


                    verify_btn.clicked.connect(verify_email)
                    layout.addWidget(verify_card)


                pw_card = QFrame()
                pw_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 16px;
                    }
                """)
                pw_layout = QVBoxLayout(pw_card)
                pw_layout.setContentsMargins(20, 20, 20, 20)
                pw_layout.setSpacing(12)

                pw_label = QLabel("Change your password")
                pw_label.setStyleSheet("color: white; font-size: 14pt;")
                pw_layout.addWidget(pw_label)

                change_pw_btn = QPushButton("Change Password")
                change_pw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                change_pw_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #005792;
                        color: white;
                        font-size: 13pt;
                        padding: 10px;
                        border-radius: 10px;
                    }
                    QPushButton:hover {
                        background-color: #0077b6;
                    }
                """)
                pw_layout.addWidget(change_pw_btn, alignment=Qt.AlignmentFlag.AlignRight)

                layout.addWidget(pw_card)


                twofa_card = QFrame()
                twofa_card.setVisible(me.get("verified", False))
                twofa_card.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e1e;
                        border-radius: 16px;
                    }
                """)
                twofa_layout = QVBoxLayout(twofa_card)
                twofa_layout.setContentsMargins(20, 20, 20, 20)
                twofa_layout.setSpacing(12)

                info = QLabel("Enable Two-Factor Authentication for enhanced security.")
                info.setWordWrap(True)
                info.setStyleSheet("color: white; font-size: 12pt;")
                twofa_layout.addWidget(info)

                twofa_enabled = me.get("2fa", False)
                enable_btn = QPushButton("Disable 2FA" if twofa_enabled else "Enable 2FA")
                enable_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                def update_2fa_button_style():
                    if me.get("2fa", False):
                        enable_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #c62828;
                                color: white;
                                font-size: 13pt;
                                padding: 10px;
                                border-radius: 10px;
                            }
                            QPushButton:hover {
                                background-color: #b71c1c;
                            }
                        """)
                    else:
                        enable_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #00796b;
                                color: white;
                                font-size: 13pt;
                                padding: 10px;
                                border-radius: 10px;
                            }
                            QPushButton:hover {
                                background-color: #004d40;
                            }
                        """)

                def toggle_2fa():
                    me["2fa"] = not me.get("2fa", False)
                    for user in users:
                        if user["username"] == me["username"]:
                            user["2fa"] = me["2fa"]
                            break
                    Path("user_data.json").write_text(json.dumps(users, indent=2))
                    enable_btn.setText("Disable 2FA" if me["2fa"] else "Enable 2FA")
                    update_2fa_button_style()

                enable_btn.clicked.connect(toggle_2fa)
                update_2fa_button_style()
                twofa_layout.addWidget(enable_btn, alignment=Qt.AlignmentFlag.AlignRight)


                layout.addWidget(twofa_card)
                layout.addStretch()


                def open_password_dialog():
                    dialog = QDialog(page)
                    dialog.setWindowTitle("Change Password")
                    dialog.setFixedWidth(400)
                    dialog.setStyleSheet("background-color: #1e1e1e; color: white;")

                    dlg_layout = QVBoxLayout(dialog)
                    dlg_layout.setContentsMargins(20, 20, 20, 20)
                    dlg_layout.setSpacing(16)

                    def labeled_input(label_text, placeholder):
                        container = QVBoxLayout()
                        label = QLabel(label_text)
                        label.setStyleSheet("color: #ccc; font-size: 11pt;")
                        field = QLineEdit()
                        field.setPlaceholderText(placeholder)
                        field.setEchoMode(QLineEdit.EchoMode.Password)
                        field.setStyleSheet("background-color: #2c2c2c; padding: 8px; border-radius: 8px;")
                        container.addWidget(label)
                        container.addWidget(field)
                        return container, field

                    curr_layout, current = labeled_input("Current Password", "Enter current password")
                    new_layout, new = labeled_input("New Password", "Enter new password")
                    conf_layout, confirm = labeled_input("Confirm New Password", "Re-enter new password")

                    dlg_layout.addLayout(curr_layout)
                    dlg_layout.addLayout(new_layout)
                    dlg_layout.addLayout(conf_layout)


                    toggle_row = QHBoxLayout()
                    toggle_btn = QPushButton()
                    toggle_btn.setIcon(QIcon(resource_path("assets/closedeye.png")))
                    toggle_btn.setFixedSize(24, 24)
                    toggle_btn.setStyleSheet("border: none;")
                    toggle_label = QLabel("Show Password")
                    toggle_label.setStyleSheet("color: white; font-size: 10pt;")
                    toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                    def toggle_visibility():
                        is_hidden = current.echoMode() == QLineEdit.EchoMode.Password
                        for field in (current, new, confirm):
                            field.setEchoMode(QLineEdit.EchoMode.Normal if is_hidden else QLineEdit.EchoMode.Password)
                        toggle_btn.setIcon(QIcon(resource_path("assets/openeye.png") if is_hidden else resource_path("assets/closedeye.png")))
                        toggle_label.setText("Hide Password" if is_hidden else "Show Password")

                    toggle_btn.clicked.connect(toggle_visibility)

                    toggle_row.addWidget(toggle_btn)
                    toggle_row.addSpacing(8)
                    toggle_row.addWidget(toggle_label)
                    toggle_row.addStretch()
                    dlg_layout.addLayout(toggle_row)


                    save_btn = QPushButton("Save")
                    save_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #00aaff;
                            padding: 10px;
                            border-radius: 8px;
                            font-weight: bold;
                            color: white;
                        }
                        QPushButton:hover {
                            background-color: #007acc;
                        }
                    """)

                    def submit_change():
                        if not current.text() or not new.text() or not confirm.text():
                            QMessageBox.warning(dialog, "Missing", "Please fill all fields.")
                            return
                        if new.text() != confirm.text():
                            QMessageBox.warning(dialog, "Mismatch", "New passwords do not match.")
                            return
                        is_strong, message = check_password_strength(new.text())
                        if not is_strong:
                            QMessageBox.warning(dialog, "Weak Password", message)
                            return
                        if current.text() != me.get("password"):
                            QMessageBox.critical(dialog, "Wrong", "Current password is incorrect.")
                            return

                        me["password"] = new.text()
                        for user in users:
                            if user["username"] == me["username"]:
                                user["password"] = new.text()
                                break
                        Path("user_data.json").write_text(json.dumps(users, indent=2))
                        QMessageBox.information(dialog, "Success", "Password updated.")
                        dialog.accept()


                    save_btn.clicked.connect(submit_change)
                    dlg_layout.addWidget(save_btn)

                    dialog.exec()

                change_pw_btn.clicked.connect(open_password_dialog)

                def check_verification_status():
                    try:
                        with open("user_data.json") as f:
                            users = json.load(f)
                        for user in users:
                            if user["username"] == me["username"]:
                                if user.get("verified", False) and not me.get("verified", False):
                                    me["verified"] = True

                                    if verify_card.isVisible():
                                        page.layout().removeWidget(verify_card)
                                        verify_card.deleteLater()
                                    twofa_card.setVisible(True)
                                    timer.stop()
                                break
                    except Exception as e:
                        print("Error checking verification status:", e)

                timer = QTimer(page)
                timer.timeout.connect(check_verification_status)
                timer.start(5000)

                return page


            def create_support_tab():
                page = QWidget()
                outer_layout = QVBoxLayout(page)
                outer_layout.setContentsMargins(0, 0, 0, 0)
                outer_layout.setSpacing(0)

                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_area.setStyleSheet("""
                    QScrollBar:vertical, QScrollBar:horizontal {
                        background: transparent;
                        width: 8px; height: 8px;
                    }
                    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                        background: #555;
                        border-radius: 4px;
                    }
                    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
                        background: none;
                        height: 0;
                        width: 0;
                    }
                """)

                scroll_content = QWidget()
                layout = QVBoxLayout(scroll_content)
                layout.setContentsMargins(24, 24, 24, 24)
                layout.setSpacing(24)
                scroll_content.setStyleSheet("background-color: #121212; color: white;")

                title = QLabel("Support & Help")
                title.setStyleSheet("font-size: 20pt; font-weight: bold;")
                layout.addWidget(title)

                contact_card = QFrame()
                contact_card.setStyleSheet("background-color: #1e1e1e; border-radius: 16px;")
                contact_layout = QVBoxLayout(contact_card)
                contact_layout.setContentsMargins(20, 20, 20, 20)
                contact_layout.setSpacing(12)

                email_input = QLineEdit()
                email_input.setPlaceholderText("Your email (optional)")
                email_input.setStyleSheet("background-color: #2c2c2c; padding: 8px; border-radius: 8px; color: white;")

                message_container = QFrame()
                message_container.setStyleSheet("""
                    background-color: #2c2c2c;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: 0px;
                    border:none;
                """)
                message_container_layout = QVBoxLayout(message_container)
                message_container_layout.setContentsMargins(8, 8, 8, 8)
                message_container_layout.setSpacing(4)

                attachments_tray = QFrame()
                attachments_tray.setStyleSheet("background-color: transparent; border-radius: 12px;")
                attachments_layout = QHBoxLayout(attachments_tray)
                attachments_layout.setContentsMargins(8, 8, 8, 8)
                attachments_layout.setSpacing(8)
                attachments_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                attachments_tray.setVisible(False)
                message_container_layout.addWidget(attachments_tray)

                message_input = QTextEdit()
                message_input.setPlaceholderText("Describe your issue or question here...")
                message_input.setMinimumHeight(120)
                message_input.setStyleSheet("""
                    QTextEdit {
                        background-color: #2c2c2c;
                        padding: 8px;
                        border-top-left-radius: 12px;
                        border-top-right-radius: 12px;
                        border-bottom-right-radius: 0px;
                        border-bottom-left-radius: 0px;
                        color: white;
                        border:none;
                    }
                    QScrollBar:vertical {
                        background: transparent;
                        width: 8px;
                    }
                    QScrollBar::handle:vertical {
                        background: #555;
                        border-radius: 4px;
                    }
                """)
                message_container_layout.addWidget(message_input)



                attach_card = QFrame()
                attach_card.setStyleSheet("""
                    background-color: #2c2c2c;
                    border-top-left-radius: 0px;
                    border-top-right-radius: 0px;
                    border-bottom-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border:none;
                """)

                attach_card_layout = QHBoxLayout(attach_card)
                attach_card_layout.setContentsMargins(8, 0, 8, 8)
                attach_card_layout.setSpacing(0)

                attach_btn = QPushButton()
                attach_btn.setIcon(QIcon(resource_path("assets/attachment.png")))
                attach_btn.setIconSize(QSize(28, 28))
                attach_btn.setFixedSize(36, 36)
                attach_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                    }
                    QPushButton:hover {
                        background-color: #444;
                        border-radius: 6px;
                    }
                """)
                attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)

                attach_card_layout.addWidget(attach_btn,alignment=Qt.AlignmentFlag.AlignLeft)



                attachment_widgets = []

                def update_tray_state():
                    attachments_tray.setVisible(bool(attachment_widgets))

                def add_attachment():
                    path, _ = QFileDialog.getOpenFileName(page, "Select Attachment")
                    if not path:
                        return

                    ext = os.path.splitext(path)[1].lower()
                    icon_path = resource_path("assets/pdf.png") if ext == ".pdf" else path
                    icon = QIcon(icon_path)

                    container = QFrame()
                    container.setFixedSize(48, 48)
                    container.setStyleSheet("background-color: rgba(255, 255, 255, 0.6); border-radius: 8px;")

                    thumb = QLabel(container)
                    thumb.setPixmap(icon.pixmap(36, 36))
                    thumb.setFixedSize(36, 36)
                    thumb.move(6, 6)

                    remove_btn = QPushButton(container)
                    remove_btn.setIcon(QIcon(resource_path("assets/remove.png")))
                    remove_btn.setIconSize(QSize(14, 14))
                    remove_btn.setFixedSize(18, 18)
                    remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    remove_btn.setStyleSheet("""
                        QPushButton {
                            background: transparent;
                            border: none;
                        }
                        QPushButton:hover {
                            background-color: rgba(255, 0, 0, 0.6);
                            border-radius: 9px;
                        }
                    """)
                    remove_btn.move(container.width() - 18, container.height() - 18)

                    attachments_layout.addWidget(container)
                    attachment_widgets.append((container, path))
                    update_tray_state()

                    def remove_attachment():
                        attachments_layout.removeWidget(container)
                        container.deleteLater()
                        attachment_widgets.remove((container, path))
                        update_tray_state()

                    remove_btn.clicked.connect(remove_attachment)

                attach_btn.clicked.connect(add_attachment)

                send_btn = QPushButton("Send via Gmail")
                send_btn.setIcon(QIcon(resource_path("assets/send.png")))
                send_btn.setIconSize(QSize(32, 32))
                send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                send_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #0077b6;
                        color: white;
                        font-size: 8pt;
                        padding: 6px 12px;
                        border-radius: 6px;
                        text-align: left;
                    }
                    QPushButton:hover {
                        background-color: #005792;
                    }
                """)


                def send_support_message():
                    message = message_input.toPlainText().strip()
                    if not message:
                        QMessageBox.warning(page, "Empty", "Please enter a message.")
                        return

                    sender_email = email_input.text().strip() or me.get("email", "Anonymous")
                    body = f"From: {sender_email}\n\n{message}"

                    if attachment_widgets:
                        files_list = "\n".join([f"- {p}" for _, p in attachment_widgets])
                        body += f"\n\nAttachments:\n{files_list}"

                    url = (
                        "https://mail.google.com/mail/?view=cm&fs=1"
                        f"&to=fittrackerpy@gmail.com"
                        f"&su=FitnessTracker Support Request"
                        f"&body={urllib.parse.quote(body)}"
                    )
                    webbrowser.open(url)

                send_btn.clicked.connect(send_support_message)
                attach_card_layout.addWidget(send_btn, alignment=Qt.AlignmentFlag.AlignRight)

                contact_layout.addWidget(QLabel("Need help? Send us a message:"))
                contact_layout.addWidget(email_input)

                message_block = QWidget()
                message_block_layout = QVBoxLayout(message_block)
                message_block_layout.setContentsMargins(0, 0, 0, 0)
                message_block_layout.setSpacing(0)

                message_block_layout.addWidget(message_container)
                message_block_layout.addWidget(attach_card)

                contact_layout.addWidget(message_block)


                layout.addWidget(contact_card)


                faq_card = QFrame()
                faq_card.setFixedHeight(220)
                faq_card.setStyleSheet("background-color: #1e1e1e; border-radius: 16px;")
                faq_layout = QVBoxLayout(faq_card)
                faq_layout.setContentsMargins(20, 20, 20, 20)
                faq_layout.setSpacing(8)

                faq_label = QLabel("Frequently Asked Questions")
                faq_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
                faq_layout.addWidget(faq_label)

                faq_scroll = QScrollArea()
                faq_scroll.setWidgetResizable(True)
                faq_scroll.setStyleSheet("""
                    QScrollBar:vertical {
                        background: transparent;
                        width: 8px;
                    }
                    QScrollBar::handle:vertical {
                        background: #555;
                        border-radius: 4px;
                    }
                """)

                faq_content = QWidget()
                faq_content_layout = QVBoxLayout(faq_content)
                faq_content_layout.setContentsMargins(0, 0, 0, 0)
                faq_content_layout.setSpacing(8)

                faqs = [
                    ("How do I verify my email?", "Go to the Security tab and click 'Verify Email'. A link will be sent to your inbox."),
                    ("How can I reset my password?", "Use the 'Change Password' option under the Security tab."),
                    ("Why am I not receiving emails?", "Check your spam folder. Also make sure your email is correct."),
                    ("Can I attach screenshots?", "Yes, you can add image attachments to support requests."),
                    ("Where can I manage 2FA?", "Under the Security tab after verifying your email."),
                ]

                for question, answer in faqs:
                    q_label = QLabel(f"Q: {question}")
                    q_label.setStyleSheet("font-weight: bold;")
                    a_label = QLabel(f"A: {answer}")
                    a_label.setWordWrap(True)
                    faq_content_layout.addWidget(q_label)
                    faq_content_layout.addWidget(a_label)

                faq_scroll.setWidget(faq_content)
                faq_layout.addWidget(faq_scroll)
                layout.addWidget(faq_card)
                layout.addStretch()

                scroll_area.setWidget(scroll_content)
                outer_layout.addWidget(scroll_area)


                help_card = QFrame()
                help_card.setStyleSheet("""
                    background-color: #1e1e1e;
                    border-radius: 16px;
                    padding: 0px;
                """)
                help_layout = QHBoxLayout(help_card)
                help_layout.setContentsMargins(16, 8, 16, 8)
                help_layout.setSpacing(6)

                help_icon_label = QLabel()
                help_icon_label.setPixmap(QIcon(resource_path("assets/help.png")).pixmap(32, 32))
                help_icon_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

                help_text_btn = QPushButton("Open Help Documentation")
                help_text_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                help_text_btn.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        color: white;
                        font-size: 12pt;
                        border: none;
                        text-align: left;
                    }
                    QPushButton:hover {
                        text-decoration: underline;
                        color: #00aaff;
                    }
                """)

                def open_help():
                    help_file = resource_path("assets/help.html").replace("\\", "/")
                    url = QUrl.fromLocalFile(help_file)
                    QDesktopServices.openUrl(url)

                help_text_btn.clicked.connect(open_help)

                help_layout.addWidget(help_icon_label)
                help_layout.addWidget(help_text_btn)
                help_layout.addStretch()

                layout.addWidget(help_card)

                return page

            content_stack.addWidget(create_profile_tab())
            content_stack.addWidget(create_security_tab())
            content_stack.addWidget(create_support_tab())

            sidebar.currentRowChanged.connect(content_stack.setCurrentIndex)
            sidebar.setCurrentRow(0)

            layout.addWidget(sidebar_wrap)
            layout.addWidget(separator)
            layout.addWidget(content_stack, stretch=1)

            return tab



        trainer_pages_info = [
            ("My Clients", "#009688", create_my_clients_tab, resource_path("assets/clients.png")),
            ("Settings", "#607D8B", create_settings_tab, resource_path("assets/settings.png")),
        ]

        for i, (label, color, page_creator, icon_path) in enumerate(trainer_pages_info):
            btn = QPushButton(label, self.button_row)
            btn.setFont(QFont(sf_family, 12))
            btn.setMinimumHeight(50)
            btn.setFixedWidth(200)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 15px;
                }}
                QPushButton:hover {{
                    background-color: #444;
                }}
            """)
            if icon_path:
                icon = QIcon(icon_path)
                btn.setIcon(icon)
                btn.setIconSize(QSize(24, 24))
                btn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

            self.button_layout.addWidget(btn)
            self.button_list.append(btn)

            page = page_creator()
            self.pages.append(page)
            self.content_area.addWidget(page)

            def on_click(checked=False, index=i, button=btn):
                self.animate_tab_indicator(button)
                self.slide_to_page(index)

            btn.clicked.connect(on_click)


        if self.button_list:
            first_btn = self.button_list[0]

            def setup_indicator():
                self.tab_indicator.setGeometry(first_btn.x() + 8, self.top_container.height() - 4, first_btn.width(), 4)

            QTimer.singleShot(0, setup_indicator)

            self.content_area.setCurrentIndex(0)
            self.current_index = 0
            first_btn.setFocus()

    def animate_tab_indicator(self, button):
        local_pos = button.mapTo(self.top_container, QPoint(0, 0))
        new_rect = QRect(
            local_pos.x(),
            self.top_container.height() - 4,
            button.width(),
            4
        )
        self.tab_indicator.raise_()
        anim = QPropertyAnimation(self.tab_indicator, b"geometry", self)
        anim.setDuration(300)
        anim.setStartValue(self.tab_indicator.geometry())
        anim.setEndValue(new_rect)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.start()
        self.tab_animation = anim

    def slide_to_page(self, new_index):
        if new_index == self.current_index:
            return

        current_widget = self.content_area.widget(self.current_index)
        next_widget = self.content_area.widget(new_index)

        direction = 1 if new_index > self.current_index else -1
        offset = direction * self.content_area.width()

        next_widget.setGeometry(offset, 0, self.content_area.width(), self.content_area.height())
        next_widget.show()

        anim_out = QPropertyAnimation(current_widget, b"geometry")
        anim_out.setDuration(300)
        anim_out.setStartValue(current_widget.geometry())
        anim_out.setEndValue(QRect(-offset, 0, self.content_area.width(), self.content_area.height()))
        anim_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        anim_in = QPropertyAnimation(next_widget, b"geometry")
        anim_in.setDuration(300)
        anim_in.setStartValue(QRect(offset, 0, self.content_area.width(), self.content_area.height()))
        anim_in.setEndValue(QRect(0, 0, self.content_area.width(), self.content_area.height()))
        anim_in.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def finish():
            self.content_area.setCurrentIndex(new_index)
            next_widget.setGeometry(0, 0, self.content_area.width(), self.content_area.height())
            self.current_index = new_index

        anim_out.finished.connect(finish)

        anim_out.start()
        anim_in.start()


        self.anim_out = anim_out
        self.anim_in = anim_in

    def add_content_page(self, text):
        label = QLabel(text)
        label.setFont(QFont(sf_family, 24))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: white;")
        self.content_area.addWidget(label)
        self.pages.append(label)
        self.content_area.setCurrentWidget(label)
        self.current_index = self.content_area.currentIndex()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    sf_font = QFont(sf_family,10,QFont.Weight.DemiBold)
    QApplication.setFont(sf_font)
    palette = app.palette()
    gradient = QLinearGradient(0, 0, 0, 700)
    gradient.setColorAt(0.0, QColor("#1f4037"))
    gradient.setColorAt(1.0, QColor("#99f2c8"))
    palette.setBrush(QPalette.ColorRole.Window, QBrush(gradient))
    app.setPalette(palette)
    app.setWindowIcon(QIcon(resource_path("assets/fitnessicon.ico")))



    window = MainWindow()
    window.show()

    sys.exit(app.exec())

def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def show_internet_error_dialog():
    winsound.MessageBeep(winsound.MB_ICONHAND)

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Internet Connection Required")
    msg_box.setText(
        "An active internet connection is required to continue.\n"
        "Please connect to the internet and click 'Retry' to proceed.\n"
        "Click 'Cancel' to exit the application."
    )
    msg_box.setStandardButtons(QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel)
    msg_box.setWindowIcon(QIcon(resource_path("assets/nointernet.png")))
    msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowType.Dialog)

    return msg_box.exec()

if __name__ == "__main__":
    if "--verify-server" in sys.argv:
        from flask import Flask
        run_verification_server()

        import time
        while True:
            time.sleep(1)
    else:

        while not is_connected():
            result = show_internet_error_dialog()

            if result == QMessageBox.StandardButton.Cancel:
                sys.exit(1)

        run_verification_server()

        main()
