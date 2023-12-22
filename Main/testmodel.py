import cv2 


video=cv2.VideoCapture(0)


facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #thư viện nhận diện khuôn mặt của OpenCV

recognizer = cv2.face.LBPHFaceRecognizer_create() #tạo bộ nhận diện khuôn mặt sử dụng thuật toán Local Binary Histogram có sẵn trong thư viện của OpenCV
recognizer.read("Trainer.yml")

name_list = ["", "Tay", "MinhMan", "Obama", "Tei"]

#imgBackground = cv2.imread("background.png")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5) #bộ phân loại khuôn mặt với scale chuẩn là 1,3,5 tăng hiệu suất nhận diện khuôn mặt
    for (x,y,w,h) in faces: #duyệt qua khuôn mặt được phát hiện
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w]) #tập trung đối tượng nhận diện khuôn mặt để dự đoán nhãn serial trong ảnh 
        if conf<50:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1) #vẽ hình chữ nhật lên khung ảnh với các kênh màu RBG 0,0,255 là màu đỏ vì xanh và lục là 0
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2) #tông màu đỏ nhẹ, độ dày khung là 2
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, name_list[serial], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) #thêm văn bản vào khung hình frame với tên từ name_list với font chữ hershey_simplex, tỷ lệ 0.8. Màu chữ màu trắng 
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, "Unknown", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    frame=cv2.resize(frame, (640, 480))
    #imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Face Recognition",frame)
    
    if cv2.waitKey(1)==ord('q'): #ord là mã ASCII của ký tự q. ấn q để dừng và thoát vòng lặp 
        break;

video.release()
cv2.destroyAllWindows()
