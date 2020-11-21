import cv2
import sys
import gc
from face_recongnition_system.face_cnn import Model
import face_recongnition_system.face_cnn as face_cnn
#
# if __name__ == '__main__':
#     if len(sys.argv) != 1:
#         print("Usage:%s camera_id\r\n" % (sys.argv[0]))
#         sys.exit(0)

#捕获指定摄像头的实时视频流
capture=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#加载模型
model = Model()
model.load_model(file_path = face_cnn.model_save_path )
font=cv2.FONT_HERSHEY_SIMPLEX
#框住人脸的矩形边框颜色
color = (0, 255, 0)
#人脸识别分类器本地存储路径
haar= cv2.CascadeClassifier(r"E:\Anaconda3\envs\py36\Lib\site-packages\cv2\data/haarcascade_frontalface_default.xml")
status=False
face = " "
#循环检测识别人脸
if capture.isOpened():
    print("相机信号良好，准备就绪")
    status=True
else:
    print("相机故障，请检查后重试")
    status=False
    sys.exit(0)
while status:
    ret, frame = capture.read()   #读取一帧视频
    if ret is True:
        cv2.imshow("cammer",frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # 灰度
        #faces = haar.detectMultiScale(gray,1.2,5,0,(100,100))

        #利用分类器识别出哪个区域为人脸
        faceRects = haar.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))

        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)
                print("face_id:{0}".format(faceID))
                faceID=int(faceID)
                print(type(faceID))

                #如果是“我”
                if faceID == 0:
                    face='wangpeng'
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame, face, (x+5, y-5), font, 1, (0, 0, 255), 1)
                    print(face)
                    #文字提示是谁
                    # cv2.putText(frame,'wangpeng',
                    #             (x + 30, y + 30),                      #坐标
                    #             cv2.FONT_HERSHEY_SIMPLEX,             #字体
                    #             1,                                     #字号
                    #             (255,0,255),                           #颜色
                    #             2)                                     #字的线宽
                    # print("wangpeng is running")
                elif faceID == 1:
                    face2="fulin"
                    print(face)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame, face, (x+5, y-5), font, 1, (0, 0, 255), 1)
                    #文字提示是谁

                    # cv2.putText(frame,'fulin',
                    #             (x + 30, y + 30),                      #坐标
                    #             cv2.FONT_HERSHEY_SIMPLEX,              #字体
                    #             1,                                     #字号
                    #             (255,0,255),                           #颜色
                    #             2)
                elif faceID == 2:
                    print("xuxin")
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    cv2.putText(frame, face, (x+5, y-5), font, 1, (0, 0, 255), 1)
                    #print(face)
                else:
                    continue
                cv2.putText(frame, face, (x+5, y-5), font, 1, (0, 0, 255), 1)
        # cv2.imshow("识别", frame)
        #如果输入q则退出循环
        if (cv2.waitKey(20) & 0xFF==ord("q")):
            status=False
#释放摄像头并销毁所有窗口
capture.release()
cv2.destroyAllWindows()