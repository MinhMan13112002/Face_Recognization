
import cv2 

video=cv2.VideoCapture(0) #mở camera

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #lây thư viện haarcascade bao gồm nhận diện được khuôn mặt, mắt và miệng

id = input("Enter Your ID: ") #nhập id người dùng
# id = int(id)
count=0

while True:
    ret,frame=video.read() #đọc khung hình từ video 
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = facedetect.detectMultiScale(gray, 1.3, 5) #bộ phân loại khuôn mặt với scale chuẩn là 1,3,5 tăng hiệu suất nhận diện khuôn mặt
    for (x,y,w,h) in faces: #duyệt qua khuôn mặt được phát hiện
        count=count+1 #đếm số lượng khuôn mặt được ghi lại từ camera
        cv2.imwrite('datasets/User.'+str(id)+"."+str(count)+".jpg", gray[y:y+h, x:x+w]) #lưu ảnh xám vào tệp datasets với tên theo định dạng User.id.count.jpg vd: User.1.5
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1) #Vẽ một hình chữ nhật xung quanh khuôn mặt trên khung hình gốc (frame). Màu của hình chữ nhật là đỏ (50, 50, 255) và độ dày là 1 pixel.

    cv2.imshow("Face Detection",frame) # hiển thị cửa sổ

    k=cv2.waitKey(1)

    if count>20: #khi số lượng hình lớn hơn 20 thì dừng capture
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done..................")