import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path="datasets"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)] # load thư mục face detect đang chứa ảnh khuôn mặt người dùng (datasets) add vào mảng face cùng id. Face là ảnh còn Id là người dùng
    faces=[]
    ids=[]
    for imagePaths in imagePath: #focus vào đường dẫn imagePaths
        faceImage = Image.open(imagePaths).convert('L')#convert ảnh
        faceNP = np.array(faceImage) #convert ảnh xám  thành mảng numpy 
        Id= (os.path.split(imagePaths)[-1].split(".")[1]) #tách id từ file ảnh
        Id=int(Id) #chuyển đổi id từ chuỗi sang số nguyên
        faces.append(faceNP) #trả về mảng faces
        ids.append(Id) #trả về mảng ids
        cv2.imshow("Training",faceNP)
        cv2.waitKey(1)
    return ids, faces

IDs, facedata = getImageID(path) #gọi hàm với đường dẫn chưa data đã training và danh sách ids và facedata
recognizer.train(facedata, np.array(IDs)) #training và lưu file
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed............")