import cv2 as cv
import numpy as np

video = cv.VideoCapture(0,cv.CAP_DSHOW)
# 设置编码格式
# MP4
fourcc = cv.VideoWriter_fourcc(*"mp4v")
# avi
fourcc_2 = cv.VideoWriter_fourcc(*'XVID')
out_video = cv.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
out_video_2 = cv.VideoWriter('ori.avi',fourcc, 20.0, (640,480))


# 背景减法器 基于自适应混合高斯背景建模的背景减除法
# history：用于训练背景的帧数，默认为500帧，如果不手动设置learningRate，history就被用于计算当前的learningRate，此时history越大，learningRate越小，背景更新越慢；
# varThreshold：方差阈值，用于判断当前像素是前景还是背景。一般默认16，如果光照变化明显，如阳光下的水面，建议设为25,36，具体去试一下也不是很麻烦，值越大，灵敏度越低；
# detectShadows:是否检测影子，设为true为检测，false为不检测，检测影子会增加程序时间复杂度，如无特殊要求，建议设为false
backsub = cv.createBackgroundSubtractorMOG2(history=500,varThreshold=16,detectShadows=False)



while True:
    # ret 读取状态,frame image data
    ret,frame = video.read()
    # 获取掩码
    if ret:
        mask = backsub.apply(frame)
        # print(frame.shape)
        # print(mask.shape)
        # 扩充维度
        mask = np.expand_dims(mask,axis=2).repeat(3,axis=2)
        out_video.write(mask)
        out_video_2.write(frame)
        cv.imshow("frame",mask)
    if cv.waitKey(30) & 0xFF ==ord(' '):
        break
#     释放资源
video.release()
out_video.release()
out_video_2.release()
cv.destroyAllWindows()


