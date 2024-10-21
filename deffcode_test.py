# import the necessary packages
from deffcode import FFdecoder

# formulate the decoder with suitable source
decoder = FFdecoder("/home/atom/video-read-write-test/GX010142_trimmed.MP4").formulate()

# grab RGB24(default) 3D frames from decoder
for frame in decoder.generateFrame():

    # lets print its shape
    print(frame.shape) # (1080, 1920, 3)
    break
import cv2
cv2.imwrite('frame.png', frame)

# terminate the decoder
decoder.terminate()