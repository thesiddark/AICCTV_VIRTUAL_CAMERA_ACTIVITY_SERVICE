from deepface import DeepFace
resp = DeepFace.verify(img1_path = r"C:\Users\shahana kp\PycharmProjects\college_violence\media\tests\elon.jpg",
                       img2_path = r"C:\Users\shahana kp\PycharmProjects\college_violence\media\tests\elon.jpg")
print(resp["verified"])