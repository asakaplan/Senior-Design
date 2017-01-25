import socket
import sys
import threading
import cv2
import time
HOST = ''
PORT = 8898

videoFile = "videoOut.out"
exitCode = False
def main():
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
	print( 'Socket created')

	try:
		s.bind((HOST,PORT))
	except socket.error as msg:
		print('Bind failed. Error Code : ' + str(msg))
		sys.exit()
	print('Socket bind complete')


	s.listen(10)
	print('Socket now listening')
	global conn
	conn, addr = s.accept()
	print('Connected with ' + addr[0] + ":" + str(addr[1]))
	i = 0
	threading.Thread(target=dataReceive).start()
	cap = cv2.VideoCapture(videoFile)

	cap = cv2.VideoCapture(videoFile)
	while not cap.isOpened():
	    cap = cv2.VideoCapture(videoFile)
	    time.sleep(.1)
	    print("Wait for the header")

	pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	while True:
	    flag, frame = cap.read()
	    if flag:
		# The frame is ready and already captured
		#cv2.imshow('video', frame)
	        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

	        print(str(pos_frame)+" frames")
	    else:
		# The next frame is not ready, so we try to read it again
	        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
	        print("frame is not ready")
		# It is better to wait for a while for the next frame to be ready
	        time.sleep(.5)
	    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
		# If the number of captured frames is equal to the total number of frames,
		# we stop
	        break



def dataReceive():
	while not exitCode:
		data = conn.recv(2**15)
		tempFile = open(videoFile,"ba")

		tempFile.write(data)
		tempFile.close()
		#Process data
		conn.send(data)






if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		s.close()
		cap.release()
		cv2.destroyAllWindows()
		exitCode = True
		raise


