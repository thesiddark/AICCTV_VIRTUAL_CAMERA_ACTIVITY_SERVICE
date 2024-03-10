import cv2
import face_recognition

from dat import Database

db = Database()
# Create your views here.

qry = "SELECT * FROM `myapp_criminals` "
res = db.select(qry)

print(res)

knownimage = []
knownids = []
knownsems = []
types = []
knownname =[]

for i in res:
    s = i["photo"]
    s = s.replace("/media/", "")
    pth = "C:\\Users\\sidharth\\Documents\\GitHub\\Aicctv\\media\\" + s
    picture_of_me = face_recognition.load_image_file(pth)
    print(pth)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
    print(my_face_encoding)
    knownimage.append(my_face_encoding)
    knownids.append(i['id'])
    knownname.append(i['name'])
    types.append("criminal")


userid="4"

qry2="SELECT * FROM `myapp_family` WHERE `USER_id`='28'"
res2=db.selectOne(qry2)


for j in res:
    s = j["photo"]
    s = s.replace("/media/", "")
    pth = "C:\\Users\\sidharth\\Documents\\GitHub\\Aicctv\\media\\" + s
    picture_of_me = face_recognition.load_image_file(pth)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
    knownimage.append(my_face_encoding)
    types.append("family")
    knownname.append(j['name'])





# define a video capture object
vid = cv2.VideoCapture(0)

firsthour = (9, 10)
second = (10, 11)
third = (11, 12)
forth = (13, 14)
fifth = (14, 15)

while (True):

    ret, frame = vid.read()

    import datetime

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    cv2.imwrite("C:\\Users\\sidharth\\Documents\\GitHub\\Aicctv\\media\\det\\"+date+".jpg", frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(20)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    picture_of_others = face_recognition.load_image_file("C:\\Users\\sidharth\\Documents\\GitHub\\Aicctv\\media\\det\\"+date+".jpg")

    # print(pth)
    path="/media/det/"+date+".jpg"
    others_face_encoding = face_recognition.face_encodings(picture_of_others)

    totface = len(others_face_encoding)

    print("wait.", totface)

    from datetime import datetime

    curh = float(str(datetime.now().time().hour) + "." + str(datetime.now().time().minute))

    print(curh, "...")

    k=0

    for i in range(0, totface):

        print("inside check")
        res = face_recognition.compare_faces(knownimage, others_face_encoding[i], tolerance=0.45)
        print(res, "verifiyng")
        l = 0
        for j in res:
            if j == True:

                # print(knownids[l], "detected")
                if types[l]=='criminal':

                    # qry = "INSERT INTO `myapp_detection` (`date`,`time`,`CRIMINAL_id`) VALUES (CURDATE(),CURTIME(),'" + str(knownids[l]) + "')"

                    qry="INSERT INTO `myapp_detection`(`date`,`time`,`did`,`name`,`type`,`photo`,`USER_id`) VALUES (CURDATE(),CURTIME(),'" + str(knownids[l]) + "','"+knownname[l]+"','criminal','"+path+"','"+userid+"')"
                    db.insert(qry)

                    k=k+1
                elif types[l] == 'familiy':
                    # qry = "INSERT INTO `myapp_detection` (`date`,`time`,`CRIMINAL_id`) VALUES (CURDATE(),CURTIME(),'" + str(knownids[l]) + "')"

                    qry = "INSERT INTO `myapp_detection`(`date`,`time`,`did`,`name`,`type`,`photo`,`USER_id`) VALUES (CURDATE(),CURTIME(),'" + str(knownids[l]) + "','"+knownname[l]+"','criminal','"+path+"','"+userid+"')"
                    db.insert(qry)
                    k = k + 1
            l = l + 1


    if k==0:
        qry = "INSERT INTO `myapp_detection`(`date`,`time`,`did`,`name`,`type`,`photo`,`USER_id`) VALUES (CURDATE(),CURTIME(),'" + str("0") + "','unknown detected','unknown','"+path+"','"+userid+"')"
        db.insert(qry)





vid.release()
# Destroy all the windows
cv2.destroyAllWindows()