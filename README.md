# Face_Recognization
Main is folder that are use in Window, Main_pi is folder that are use in Raspberry Pi 4
# Step excutive in window
1. download library python(pip install python)
2. download library Open Cv3(pip install openCV-python3)
3. download library numpy
4. download library Image
5. pip install opencv-contrib-python
6. take the picture
7. train the picture
8. test
9. translet to Raspberry Pi langues
# Thực hiện trên board Rasp:
1. cài đặt hệ điều hành cho Rasp bằng phần mềm Raspberry Pi Image thông qua thẻ nhớ.
2. Tiến hành ping giữa máy tính với lại Rasp để có đc địa chỉ Ip của Rasp:
ping pi, ping -4 pi, ping địa chỉ ip của pi.
3.Sau khi có địa chỉ Ip của Pi thì kết nối với máy tính thông qua Putty để tiến hành config Rasp:
mở SSH, I2C cho Rasp
4. kết nối Pi vs máy tính qua VNC bằng địa chỉ IP của Rasp
5. ktr camera: libcamera-hello -t0
6. Tiến hành thực thi các code để tiến hành train AI, dataset sẽ đc lưu vào mục đã định sẳn trong Pi  
