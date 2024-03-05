import cv2
import face_recognition

from DBConnection import Database


db=Database()
# Create your views here.

qry="SELECT * FROM `myapp_criminals`"
res= db.select(qry)


print(res)



knownimage=[]
knownids=[]
knownsems=[]


for i in res:
    s=i["photo"]
    s=s.replace("/media/","")
    pth="C:\\Users\\sidha\\OneDrive\\Documents\\Git\\PROJECT\\Aicctv\\media\\"+ s
    picture_of_me = face_recognition.load_image_file(pth)
    print(pth)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
    print(my_face_encoding)
    knownimage.append(my_face_encoding)
    knownids.append(i['id'])








# define a video capture object
vid = cv2.VideoCapture(0)



firsthour= (8.50,9.30)
lasthour= (18.30,19.15)





while(True):

    ret, frame = vid.read()

    cv2.imwrite(r"C:\\Users\\sidha\\OneDrive\\Documents\\Git\\PROJECT\\Aicctv\\media\\a.jpg",frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    picture_of_others = face_recognition.load_image_file(r"C:\\Users\\sidha\\OneDrive\\Documents\\Git\\PROJECT\\Aicctv\\media\\a.jpg")
    # print(pth)
    others_face_encoding = face_recognition.face_encodings(picture_of_others)


    totface=len(others_face_encoding)


    print("aaaaa", totface)

    from datetime import datetime

    curh = float(str(datetime.now().time().hour) + "." + str(datetime.now().time().minute))

    print(curh, "hgfhhgfgfghfghfgh")
    period=0
    if firsthour[0]<  curh < firsthour[1]:
        period=1
    # elif secondhour[0]< curh < secondhour[1]:
    #     period=2
    # elif thirdhour[0]< curh < thirdhour[1]:
    #     period=3
    # elif fourthhour[0]< curh < fourthhour[1]:
    #     period=4
    elif lasthour[0]< curh < lasthour[1]:
        period=5


    for i in range(0,totface):


        print("inside check")
        res=face_recognition.compare_faces(knownimage,others_face_encoding[i],tolerance=0.45)
        print(res,"helllo")
        l=0
        # for j in res:
        #     if j==True:
        #
        #         qry = "SELECT * FROM `myapp_attendance` WHERE `date`=CURDATE() and checkin_checkout='Check in' and STUDENT_id='" + str(
        #             knownids[l]) + "'"
        #         resa = db.selectOne(qry)
        #
        #         if resa is not None:
        #
        #             qry = "update myapp_attendance set  checkin_checkout='Check out' where STUDENT_id='" + str(
        #                 knownids[l]) + "'"
        #             db.update(qry)
        #         else:
        #
        #             qry = "INSERT INTO `myapp_attendance` (`checkin_checkout`,`date`,`STUDENT_id`,`time`,`photo`) VALUES ('Check in',CURDATE(),'" + str(
        #                 knownids[l]) + "',CURTIME(),'')"
        #             db.insert(qry)
        #     l=l+1
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()