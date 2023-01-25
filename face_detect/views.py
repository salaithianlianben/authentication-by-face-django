from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from .models import *
import cv2,time
import numpy as np
from os import listdir
from os.path import join, isfile

# Create your views here.
def Login(request):
    error=""
    if request.method=="POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u,password=p)
        if user.is_staff:
            data_path = 'C:/Users/DELL/Downloads/FaceDetectionSystem/FaceDetectionSystem/face_detect/images/'
            onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

            Training_Data, Labels = [], []

            for i, files in enumerate(onlyfiles):
                image_path = data_path + onlyfiles[i]
                images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                Training_Data.append(np.asarray(images, dtype=np.uint8))
                Labels.append(i)

            Labels = np.asarray(Labels, dtype=np.int32)

            model = cv2.face.LBPHFaceRecognizer_create()

            model.train(np.asarray(Training_Data), np.asarray(Labels))

            print("Model training Complete !!!!!")

            face_classifier = cv2.CascadeClassifier(
                'C:/Users/DELL/Downloads/FaceDetectionSystem/FaceDetectionSystem/face_detect/haarcascade_frontalface_default.xml')

            def face_detector(img, size=0.5):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)

                if faces == ():
                    return img, []

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    roi = img[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (200, 200))

                return img, roi

            cap = cv2.VideoCapture(0)
            while True:

                ret, frame = cap.read()

                image, face = face_detector(frame)

                try:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face)

                    if result[1] < 500:
                        confidence = int(100 * (1 - (result[1]) / 300))

                    if confidence > 85:
                        login(request,user)
                        cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face Cropper', image)
                        time.sleep(5)
                        error="yes"
                        break


                    else:
                        cv2.putText(image, "Can't Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face Cropper', image)
                        time.sleep(5)
                        error="no"
                        break


                except:
                    cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', image)
                    time.sleep(5)
                    error="noface"
                    break

            cap.release()

            cv2.destroyAllWindows()
    d={'error':error}
    return render(request,'login1.html',d)

def home(request):
    return render(request,'home.html')

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')


def signup(request):
    error = False
    if request.method=="POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        a = request.POST['add']
        m = request.POST['mobile']
        i = request.FILES['image']
        user = User.objects.create_superuser(username=u,password=p,email=e,first_name=f,last_name=l)
        Profile.objects.create(user=user,mobile=m,add=a,image=i)
        face_classifier = cv2.CascadeClassifier(r"C:\Users\DELL\Downloads\FaceDetectionSystem\FaceDetectionSystem\face_detect\haarcascade_frontalface_default.xml")

        def face_extractor(img):

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if faces == ():
                return None

            for (x, y, w, h) in faces:
                cropped_faces = img[y:y + h, x:x + w]

            return cropped_faces

        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (400, 400))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = 'C:/Users/DELL/Downloads/FaceDetectionSystem/FaceDetectionSystem/face_detect/images/user' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)

            else:
                print("Face Not Found")
                pass
            if cv2.waitKey(1) == 13 or count == 100:
                break
        cap.release()
        cv2.destroyAllWindows()
        error = True

    d = {'error':error}
    return render(request,'signup.html',d)

def Logout(request):
    logout(request)
    return redirect('home')