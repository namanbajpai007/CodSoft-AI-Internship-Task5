import cv2 , numpy , os

haar_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
datasets='dataset'
print('Training...')
(images, labels, names, id) = ([], [], {},0)

for (subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)
(width , height) = (130,100)

model = cv2.face.LBPHFaceRecognizer_create()
#OR model = cv2.face.FisherFaceRecognizer_create()
model.train(images , labels)

webcam = cv2.VideoCapture(0)
cnt=0

while True:
    (_,im) = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h), (0,255,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face, (width,height))

        prediction= model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h), (0,255,0),2)
        if prediction[1]<800:
            cv2.putText(im , '%s-%.0f' % (names[prediction[0]],prediction[1]),(x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0))
            print (names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(im,'Unknown', (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255))
            if (cnt>100):
                        print("Unknown Person")
                        cv2.imwrite("Unknwon.jpg",im)
                        cnt=0
    cv2.imshow("FaceRecognition",im)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.closeAllWindows()
