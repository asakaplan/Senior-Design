import socket
import sys
import threading
import cv2
import time
HOST = ''
PORT = 8898

videoFile = "videoOut.out"
exitCode = False
cap = cv2.VideoCapture(videoFile)
while not cap.isOpened():
    cap = cv2.VideoCapture(videoFile)
    time.sleep(.1)
    print("Wait for the header")
outputVideo = cv2.VideoWriter()
success = outputVideo.open("testOutput.avi",  cv2.cv.CV_FOURCC(*'XVID'),int(round(cap.get(cv2.cv.CV_CAP_PROP_FPS),0)),(int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))),1)
print("Success: ", success)
pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
    if flag:
	# The frame is ready and already captured
	#cv2.imshow('video', frame)
	outputVideo.write(cv2.flip(frame,0))
	pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        if(pos_frame%1000==0):print(str(pos_frame)+" frames")
    else:
	# The next frame is not ready, so we try to read it again
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
	outputVideo.release()
        print("frame is not ready")
	# It is better to wait for a while for the next frame to be ready
        time.sleep(.5)
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
	# If the number of captured frames is equal to the total number of frames,
	# we stop
        break






if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		s.close()
		cap.release()
		cv2.destroyAllWindows()
		exitCode = True
		raise


