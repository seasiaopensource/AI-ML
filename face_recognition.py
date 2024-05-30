from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from apps.home.models import *
from datetime import datetime, timedelta ,date
from django.http import HttpResponseNotFound
from django.db.models import Count, Min, Max, Q, Value
from django.db.models.functions import Coalesce
from .models import registration, attendance
from django.views.decorators import gzip
from django.shortcuts import render
from datetime import datetime, timezone
from apps.home.architecture import *
from sklearn.preprocessing import Normalizer
from django.utils import timezone
from django.db.models.functions import TruncMonth
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import mediapipe as mp
import pytz, os, cv2, pyttsx3, time, shutil, base64, json, threading, mtcnn, pickle, decimal
from calendar import monthrange
from django.core.exceptions import ObjectDoesNotExist
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import multiprocessing
from keras.models import load_model
from scipy.spatial.distance import cosine
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404, redirect
from apps.home.form import AttendanceForm
from django.core.paginator import Paginator, EmptyPage
from pytz import timezone as tz
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator


global dic
dic={
    'id':None,
    'name':None,
    'department':None,
    'img_string':None
}



def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


@login_required(login_url="/login/")
def index(request):
    return render(request,"home/index.html")


@login_required(login_url="/login/")
def user_register(request):
    ignored_id = 1
    employees = registration.objects.all().order_by('-id').exclude(id=ignored_id)

    paginator = Paginator(employees, 10)  # You can adjust the page size as needed
    page_num = request.GET.get('page')

    try:
        paginated_data = paginator.get_page(page_num)
    except EmptyPage:
        # Handle the case where the requested page is out of range
        paginated_data = paginator.get_page(1)

    context = {'data': []}
    for item in paginated_data:
        image_data = item.image  
        image_data_base64 = base64.b64encode(image_data).decode('utf-8')
        context['data'].append({'item': item, 'image_data_base64': image_data_base64})

    context['paginated_data'] = paginated_data  # Add paginated_data to context for pagination links
    return render(request, "home/user_registration.html", context)


def user_register_attendance(request, id):
    uid = int(id)
    current_date = datetime.now().strftime('%Y-%m-%d')
    user_attendance = attendance.objects.filter(emp_id=uid, date_time__contains=current_date)
    context = {'data': []}
    if user_attendance.exists():
        status ="Present"
        for item in user_attendance:
            image_data = item.image 
            image_data_base64 = base64.b64encode(image_data).decode('utf-8')
            context['data'].append({'item': item, 'image_data_base64': image_data_base64, "status":status })
    else:
        user_data=registration.objects.get(id=uid)
        image_data_base64 = base64.b64encode(user_data.image).decode('utf-8')
        context = {'data': []}
        status="Absent"
        context['data'].append({
            "item": user_data, "image_data_base64":image_data_base64, "status":status
        })
    return render(request, "home/user_registration_attendance.html", context)


def get_id(request):
    if request.method == "POST":
        data = json.loads(request.body)
        id = data.get('inputid')
        name = data.get('inputname')
        dept = data.get('inputdepartment')
        dic['id'], dic['name'], dic['dept']=id, name, dept
        if dic['id'] == 'None':
            pass
        else: 
            print(dic['id'])
            print(dic['name'])
            print(dic['dept'])
    return JsonResponse({"status": "success"})


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
    
def speech_thread(text):
    thread = threading.Thread(target=text_to_speech, args=(text,))
    thread.start()
    
    
def capture_video():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/dev/video0')
    # cap = cv2.VideoCapture('/dev/video2')
    # cap = cv2.VideoCapture("rtsp://admin:admin123@10.8.21.48:554/cam/realmonitor?channel=1&subtype=1")
    face_id = dic['id']
    face_position = "Center"
    output_directory = 'emp_dataset/dataset_color/'
    if os.path.isdir(output_directory):
            shutil.rmtree(output_directory)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    frame1 = None
    image_count = 0
    capture_interval = 3 
    image_counts = {"Left": 0, "Right": 0, "Up": 0, "Down": 0, "Center": 0}
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame2 = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]), \
                            int(bbox.width * frame.shape[1]), int(bbox.height * frame.shape[0])
                x_offset = int(-0.1 * w) 
                y_offset = int(-0.2 * h)  
                x, y, w, h = x + x_offset, y + y_offset, w - 2 * x_offset, h - y_offset
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_landmarks = face_mesh.process(frame_rgb)
                if face_landmarks.multi_face_landmarks:
                    for face_landmark in face_landmarks.multi_face_landmarks:
                        nose_x = int(face_landmark.landmark[4].x * frame.shape[1])
                        nose_y = int(face_landmark.landmark[4].y * frame.shape[0])
                        if nose_x < x + 0.4 * w:
                            face_position = "Left"
                        elif nose_x > x + 0.6 * w:
                            face_position = "Right"
                        elif nose_y < y + 0.5 * h:
                            face_position = "Up"
                        elif nose_y > y + 0.6 * h:
                            face_position = "Down"
                        else:
                            face_position = "Center"
                        mp_drawing.draw_landmarks(frame, face_landmark, mp_face_mesh.FACEMESH_TESSELATION,
                                                landmark_drawing_spec=None,
                                                connection_drawing_spec=mp_drawing.DrawingSpec(
                                                    color=(255, 255, 255), thickness=1, circle_radius=1))
                else:
                    face_position = "Please show the face properly"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, face_position, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if frame1 is None:
            frame1 = frame.copy()
        if face_position != "Please show the face properly":
            image_counts[face_position] += 1
            if image_counts[face_position] <= 10:
                if frame1 is not None:
                    image_count += 1
                    image_filename = os.path.join(output_directory, f"image_{image_count}.jpg")
                    face_image = frame2[y:y+h, x:x+w]
                    face_image = cv2.resize(face_image, (160, 160))
                    cv2.imwrite(image_filename, frame2)
                    print(f"Saved image: {image_filename}")
                    if image_count==10:
                            img_str = cv2.imencode('.jpeg', frame2)[1].tostring()
                            dic['img_string'] = img_str
                            nparr = np.fromstring(img_str, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            cv2.imwrite(f'emp_{face_id}.jpeg', frame2)
                    frame1 = None
                    if image_counts[face_position] == 10:
                        additional_text = f"Captured 10 images for {face_position}"
                        cv2.putText(frame, additional_text, (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 0, 255), 2)
                        speech_thread(additional_text)
                        if image_count == 50:
                            time.sleep(3)
                            text = "all images captured thanks for your patience"
                            speech_thread(text)
                            time.sleep(3)
                            break
            else:
                pass
        ret, frame1 = cv2.imencode('.jpg', frame)
        response_bytes = (b'--frame1\r\n'
                        b'Content-Type: image/jpg\r\n\r\n' + frame1.tobytes() + b'\r\n')
        yield response_bytes
    cap.release()

