from __future__ import absolute_import
from __future__  import division
from __future__ import print_function

import face_recognition
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO
import time
from DBConnection import Database
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def mamon_videoFightModel2(tf,wight='mamonbest947oscombo-drive.hdfs'):
    layers = tf.keras.layers
    models = tf.keras.models
    losses = tf.keras.losses
    optimizers = tf.keras.optimizers
    metrics = tf.keras.metrics
    num_classes = 2
    cnn = models.Sequential()
    #cnn.add(base_model)

    input_shapes=(160,160,3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.vgg19.VGG19
    base_model = vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
    # Freeze the layers except the last 4 layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())
    model = models.Sequential()

    model.add(layers.TimeDistributed(cnn,  input_shape=(30, 160, 160, 3)))
    model.add(layers.LSTM(30 , return_sequences= True))

    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="sigmoid"))

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wight)
    rms = optimizers.RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model

import numpy as np
from skimage.transform import resize
np.random.seed(1234)
model22 = mamon_videoFightModel2(tf)


def video_mamonreader(cv2,filename):
    db = Database()
    qryq = "SELECT * FROM `myapp_criminals` "
    resq = db.select(qryq)

    print(resq)

    knownimage = []
    knownids = []

    for i in resq:
        s = i["photo1"]
        # s = s.replace("/media/", "")
        pth = "C:\\Users\\sidharth\\Documents\\GitHub\\Aicctv\\" + s
        picture_of_me = face_recognition.load_image_file(pth)
        print(pth)
        my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
        # print(my_face_encoding)
        knownimage.append(my_face_encoding)
        knownids.append(i['id'])

    frames = np.zeros((30, 160, 160, 3), dtype=np.cfloat)
    i=0
    # print(frames.shape)
    vc = cv2.VideoCapture(filename)


    count=0


    lkframes=[]

    while True:

        try:
            if count < 30:
                rval, frame = vc.read()

                lkframes.append(frame)
                # cv2.imshow("test", frame)
                cv2.waitKey(1)
                frm = resize(frame,(160,160,3))
                frm = np.expand_dims(frm,axis=0)
                if(np.max(frm)>1):
                    frm = frm/255.0
                frames[i][:] = frm
                i +=1
                count= count+1

                img = frame
                # Convert into grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.imshow("test", img)
                cv2.waitKey(1)

            else:
                datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
                datav[0][:][:] = frm
                millis = int(round(time.time() * 1000))
                print(millis)
                f, precent = pred_fight(model22, datav, acuracy=0.65)
                millis2 = int(round(time.time() * 1000))
                print(millis2)
                res_mamon = {'fight': f, 'precentegeoffight': str(precent)}
                print(res_mamon)
                count=0
                i=0


                if f== True:

                    # db=Database()
                    # id=db.insert("INSERT INTO `myapp_violence` (`date`,`time`) VALUES (CURDATE(),CURTIME())")
                    #
                    # mediapath="C:\\Users\\Acer\\PycharmProjects\\AI_CCTV_ideal\\media\\"
                    #
                    # for i in lkframes:
                    #
                    #     from datetime import datetime
                    #     filename= datetime.now().strftime("%Y%m%d%H%M%S")+".jpg"
                    #
                    #     ms= mediapath+ filename
                    #
                    #     cv2.imwrite(ms,i)
                    #
                    #     qry="INSERT INTO `myapp_violencesub` (`image`,`VIOLENCE_id`) VALUES ('"+"/media/"+ filename +"','"+str(id)+"')"
                    #
                    #     db.insert(qry)
                    mediapath = "C:\\Users\\sidha\\OneDrive\\Documents\\Git\\PROJECT\\Aicctv\\media\\detection\\"
                    from datetime import datetime
                    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"

                    ms = mediapath + filename
                    db = Database()
                    id = db.insert("INSERT INTO `myapp_detect` (`date`,`time`,`image`) VALUES (CURDATE(),CURTIME(),'" + "/media/" + filename + "')")


                    for i in lkframes:
                        cv2.imwrite(ms, i)
                        picture_of_others = face_recognition.load_image_file(ms)
                        # print(pth)
                        others_face_encoding = face_recognition.face_encodings(picture_of_others)

                        totface = len(others_face_encoding)
                        print("aaaaa", totface)
                        for i in range(0, totface):
                            print("inside check")
                            res = face_recognition.compare_faces(knownimage, others_face_encoding[i], tolerance=0.45)
                            print(res, "helllo")
                            l = 0
                            for j in res:
                                if j == True:
                                    print(knownids[l], "found")
                                    qryChk = "SELECT * FROM `myapp_detect_sub` where `CRIMINAL_id`='"+str(knownids[i])+"' and `DETECT_id`='"+str(id)+"'"
                                    resChk = db.select(qryChk)
                                    print(resChk)
                                    print(len(resChk), 'leno')
                                    if len(resChk)>0:
                                        continue
                                    else:
                                        qry = "INSERT INTO `myapp_detect_sub` (`CRIMINAL_id`,`DETECT_id`) VALUES ('" + str(knownids[i]) + "','" + str(id) + "')"
                                        # db.insert(qry)
                                        # qry = "INSERT INTO `myapp_violencesub` (`STUDENT_id`,`VIOLENCE_id`) VALUES ('" + str(knownids[i]) + "')"
                                        db.insert(qry)

                                l = l + 1

                lkframes=[]
                lkframes.append(frame)

                img = frame
                # Convert into grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.imshow("test", img)
                cv2.waitKey(1)

                rval, frame = vc.read()
                frm = resize(frame, (160, 160, 3))
                frm = np.expand_dims(frm, axis=0)
                if (np.max(frm) > 1):
                    frm = frm / 255.0
                frames[i][:] = frm
                i += 1
                count = count + 1
        # return frames

        except:
            break


    vc.release()
    cv2.destroyAllWindows()

def pred_fight(model,video,acuracy=0.9):
    pred_test = model.predict(video)
    if pred_test[0][1] >=acuracy:
        return True , pred_test[0][1]
    else:
        return False , pred_test[0][1]


def main_fight(vidoss):
    video_mamonreader(cv2,vidoss)
    # datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
    # datav[0][:][:] = vid
    # millis = int(round(time.time() * 1000))
    # print(millis)
    # f , precent = pred_fight(model22,datav,acuracy=0.65)
    # millis2 = int(round(time.time() * 1000))
    # print(millis2)
    # res_mamon = {'fight':f , 'precentegeoffight':str(precent)}
    # res_mamon['processing_time'] =  str(millis2-millis)
    # return res_mamon

#

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image




# res = main_fight('hdfight.mp4')
# print(res)
#
#
# res = main_fight('golsss.mp4')
# print(res)


res = main_fight('conflict.mp4')
print(res)


res = main_fight('cpl.mp4')
print(res)


res = main_fight('cpl2.mp4')
print(res)

res = main_fight('bully.mp4')
print(res)

res = main_fight('fight12.mp4')
print(res)











