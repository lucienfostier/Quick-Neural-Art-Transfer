import zmq
import numpy as np
import neuralStyle
from zmq_array import send_array, recv_array
from traceback import print_exc
print("before precompute")
neuralStyle.precompute()
print("after precompute")



def server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)    
    socket.bind ("tcp://*:7788")
    i = 0
    while 1:
        print("start loop")
        socket.setsockopt(zmq.RCVTIMEO, -1)
        art_style_filename = socket.recv_string()
        print(art_style_filename)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        photo_content = recv_array(socket)
        print("get photo_content", photo_content.shape)
        for img in neuralStyle.p_transfer(photo_content, art_style_filename):
            send_array(socket, img)
            print("send")
            s = socket.recv_string()
            print("ok", s)
        send_array(socket, np.array([]))
        print("done")


while 1:
    try:
        server()
    except Exception as e:
            print_exc()
            print("time out")            
        
