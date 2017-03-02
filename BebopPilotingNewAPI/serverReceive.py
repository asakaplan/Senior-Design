import socket
import sys
import threading
import cv2
import time
import os
HOST = ''
PORT = 8001

videoReceive = "PLEASEWORK.avi"
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
        print("Here1")
        os.remove(videoReceive)
        print("Here2")
        #os.mkfifo(videoReceive)
        print("Here3")
        tempFile = open(videoReceive,"wb")
        print("Here4")
        while not exitCode:
            #print("In statement")
            #print("Heyo here's some data: %d"%len(data))
            data = conn.recv(2**15)
            #print("Received")
            tempFile.write(data)
            #Process data
            #conn.send(data)



"""def dataReceive():
    os.remove(videoReceive)
    os.mkfifo(videoReceive)
    tempFile = open(videoReceive,"ab")
    while not exitCode:
        print("In statement")
        #print("Heyo here's some data: %d"%len(data))
        data = conn.recv(2**15)
        print("Received")
        tempFile.write(data)
        #Process data
        #conn.send(data)
"""
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("In Interrupt")
        exitCode = True
        os.unlink(videoSend)
        s.close()
        cv2.destroyAllWindows()
        print("End of interrupt")
        raise
