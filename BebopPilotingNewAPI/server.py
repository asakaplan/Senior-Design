import socket
import sys
import threading
import cv2
import time
import os
HOST = ''
PORT = 8000

videoSend = "videoOut.out"
videoReceive = "videoTemp.avi"
exitCode = False
def main():
        try:
           os.remove(videoSend)
           os.remove(videoReceive)
        except Exception:
           pass

        global s
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
        threading.Thread(target=dataSend).start()
        global cap
        cap = cv2.VideoCapture(videoReceive)
        while not cap.isOpened():
            cap = cv2.VideoCapture(videoReceive)
            time.sleep(.1)
            print("Wait for the header")
        global outputVideo
        outputVideo = cv2.VideoWriter()

        while not cap.get(cv2.cv.CV_CAP_PROP_FPS) or not cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT):
                print("Still waiting for the header")
                time.sleep(.1)
        success = outputVideo.open(videoSend,  cv2.cv.CV_FOURCC(*'XVID'),int(round(cap.get(cv2.cv.CV_CAP_PROP_FPS),0)),(int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))),1)
        print("Success: ", success)
	
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            if flag:
                # The frame is ready and already captured
                #cv2.imshow('video', frame)
                pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                
                outputVideo.write(cv2.flip(frame,0))
                pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                if(pos_frame%1000==0):print(str(pos_frame)+" frames")
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
		tempFile = open(videoReceive,"ab")
		#print("Heyo here's some data: %d"%len(data))
		tempFile.write(data)
		tempFile.close()
		#Process data
		#conn.send(data)
def dataSend():
	lastLength = 0
	while 1:
		try:
			outp = open(videoSend, "rb")
			data = "".join(outp.readlines())
			if len(data)==lastLength:continue
			print("New length: ", len(data))
			conn.send(data[lastLength:])
			lastLength=len(data)
		except Exception as e:
			#print("waiting for file to exist", e)
			time.sleep(.1)



if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Trying to quit!")
		s.close()
		cap.release()
		videoSend.release()
		videoReceive.release()
		cv2.destroyAllWindows()
		exitCode = True
		raise


