from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition, SwapTransition, NoTransition
from kivy.uix.carousel import Carousel
from kivy.uix.image import Image
from kivy.clock import Clock, mainthread
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import *
from kivy.config import Config
from kivy.animation import Animation
import threading

import glob
import cv2 
import numpy as np
import skimage.transform
from time import sleep, time
import zmq
from zmq_array import send_array, recv_array


def prep_image(im, IMAGE_W, IMAGE_H):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h*IMAGE_W < w*IMAGE_H:
        im = skimage.transform.resize(im, (IMAGE_H, w*IMAGE_H//h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*IMAGE_W//w, IMAGE_W), preserve_range=True)

    # Central crop
    h, w, _ = im.shape
    im = im[h//2-IMAGE_H//2:h//2+IMAGE_H//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    rawim = im.astype('uint8')
    print(rawim.dtype, im.dtype)

    return rawim

# Create both screens. Please note the root.manager.current: this is how
# you can control the ScreenManager from kv. Each screen has by default a
# property manager that gives you the instance of the ScreenManager used.

# Declare both screens
root = None
class MenuScreen(Screen):
    pass


class ProcessScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__( **kwargs)
        self.context = zmq.Context()
        self.socket = None
        self.output_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        self.style_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        self.photo_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        with self.canvas:
            #self.texture = self.video_texture
            self.output_rect = Rectangle(texture=self.output_texture, pos=(-5,0), size=(1, 1))
            self.style_rect = Rectangle(texture=self.style_texture, pos=(-5,0), size=(1, 1))
            self.photo_rect = Rectangle(texture=self.photo_texture, pos=(-5,0), size=(1, 1))
        self.counter = 0
        self._keyboard = None
        
    def start_anim(self):
        self.style_texture.blit_buffer(cv2.flip(root.art_style, 0).tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.photo_texture.blit_buffer(cv2.flip(root.photo_content, 0).tostring(), colorfmt='bgr', bufferfmt='ubyte')
        w, h = Window.width, Window.height
        rh = int(h *0.8)
        rw = rh*4//3
        self.output_rect.pos =((w-rw)//2,0)
        self.output_rect.size = (rw, rh)
        
        rh2 = int(h *0.3)
        rw2 = rh2*4//3
        self.photo_rect.pos =( -rw2,rh2+int(0.15*h))
        self.style_rect.pos =( -rw2,int(0.05*h))
        self.style_rect.size = (rw2, rh2)
        self.photo_rect.size = (rw2, rh2)
        self.canvas.ask_update()
        
        rh = int(h *0.6)
        rw = rh*4//3
        xpos2 = int((w-(rw+rw2))/3)
        idle1 = Animation(duration=1.)
        idle2 = Animation(duration=2.)
        idle3 = Animation(duration=3.)
        move1 = Animation(pos=(xpos2, rh2+int(0.15*h)), duration=2.)
        move2 = Animation(pos=(xpos2, int(0.05*h)) ,duration=2.)
        move3 = Animation(pos=(2*xpos2+rw2,int(0.1*h)), size=(rw, rh), duration=2.)
        (idle3 + move1).start(self.photo_rect)
        (idle2 + move2).start(self.style_rect)
        (idle1 + move3).start(self.output_rect)
        
    def on_pre_enter(self):
        self.is_done = False
        self.ids.w_pb.value = 0.
        if self.socket is None:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect ("tcp://127.0.0.1:7788")
        print("process screen", root.art_style.shape, root.photo_content.shape)
        self.art_style = root.art_style
        self.counter = 0
        self.update_progress(root.photo_content)
        threading.Thread(target = self.process).start()
        self.start_anim()
        self._keyboard = None


        
    def process(self):
        self.socket.send_string(root.art_style_filename, zmq.SNDMORE)
        self.socket.setsockopt(zmq.RCVTIMEO, 15000)
        self.socket.setsockopt(zmq.SNDTIMEO, 15000)
        try:
            send_array(self.socket, root.photo_content)
        except:
            self.done("網路連線失敗")
            return
        while 1:

            try:
                img = recv_array(self.socket)
                print("recv")
            except:
                img = np.array([])
                self.done("網路連線太慢")
                print("recv timeout")
                return
            if img.shape == (0,):
                print("done")
                break
            print("ok")
            try:
                self.socket.send_string("ok")
            except:
                print("send timeout")
                self.done("網路連線超時")
                return 
            self.update_progress(img)
        self.done()
        
    @mainthread
    def update_progress(self, img):
        N = 5
        #image = skimage.transform.resize(img, (480, 640), preserve_range=True).astype(np.uint8)
        image = cv2.resize(img, (640,480), interpolation=cv2.INTER_LANCZOS4)
        image = cv2.flip(image, 0)
        
        anim = Animation(value=self.counter/N, duration=0.5)
        if self.counter < N:
            self.ids.w_info.text = "處理中 %2d%%"%(self.counter*100//N)
            print(self.ids.w_info.text)
            self.ids.w_info.font_size = 64
            self.counter += 1
            anim += Animation(value=self.counter/N, duration=6.)
        Animation.cancel_all(self.ids.w_pb)
        anim.start(self.ids.w_pb)
        self.output_texture.blit_buffer(image.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.canvas.ask_update()
        
        
    @mainthread
    def done(self, msg=None):
        #root.current="menu"
        if msg:
            self.socket.close()
            self.socket = None
            self.ids.w_info.text = msg  + ", 任意鍵結束重來"
            self.ids.w_info.font_size = 32
        else:
            self.state = 120
            Clock.schedule_interval(self._tick, 1.)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        print("process keyboard bind")
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.is_done = True
    
    def _tick(self, *args):
        if self.state == 0:
            self._on_clicked()
        self.ids.w_info.text = "完成！喜歡的話，請在 %d 內將畫面拍下。然後按鍵重來。"%self.state
        self.ids.w_info.font_size = 32
        self.state -= 1
        

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        self._on_clicked()
    
    def on_touch_down(self, touch):
        self._on_clicked()
        
    def _on_clicked(self):
        if self.is_done:
            root.transition = SlideTransition(duration=1.)
            root.current = "style"
        
    def on_leave(self):
        print("process leave", self._keyboard)
        self._keyboard_closed()
        Clock.unschedule(self._tick)
        
    def _keyboard_closed(self):
        print("process keyboard closed")
        if self._keyboard:
            print("process keyboard unbind")
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard = None


class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__( **kwargs)
        cap = self.cap = cv2.VideoCapture(0)
        self.video_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        with self.canvas:
            w, h = Window.width, Window.height
            rh = int(h *0.8)
            rw = rh*4//3
            Rectangle(texture=self.video_texture, pos=( (w-rw)//2,0), size=(rw, rh))
        
    def on_pre_enter(self):
        print("art style", root.art_style_filename)
        self.art_style = style = prep_image(cv2.imread(root.art_style_filename), 640, 480)
        self.start_wait()
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        Clock.schedule_interval(self._tick, 1./20)
        
    def on_leave(self):
        print("camera leave")
        Clock.unschedule(self._tick)
        self._keyboard_closed()
        
    def _keyboard_closed(self):
        print("camera keyboard closed")
        if self._keyboard:
            print("cambera keyboard unbind")
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard = None
        
    def start_wait(self):
        self.state = "wait"
        self.ids.w_info.text = "用鏡頭拍手機畫面，按鍵準備三秒倒數"
        self.ids.w_info.font_size = 32
    
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        self._on_clicked()
    
    def on_touch_down(self, touch):
        self._on_clicked()
        
    def _on_clicked(self):
        if self.state is "wait":
            self.state = 3*20
        else:
            self.start_wait()

    def _tick(self, *args):
        
        ret, img = self.cap.read()
        img = cv2.flip(img, 1)
        img2 = cv2.addWeighted(img, 0.7, self.art_style, 0.3, 0)
        image = cv2.flip(img2, 0)
        self.video_texture.blit_buffer(image.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.canvas.ask_update()
        img2 = img
        
        if self.state is not 'wait':
            if self.state == 0:
                print("照相")
                root.photo_content = img2
                root.art_style = self.art_style
                root.transition = NoTransition()
                root.current = "process"
            if self.state%20 == 0:
                self.ids.w_info.text = "照相倒數 %d  (按鍵取消)"%(self.state//30)
                self.ids.w_info.font_size = 32
            self.state -=1

class StyleScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__( **kwargs)
        for f in glob.glob("styles/*.*"):
            src = f
            image = Image(source=src, allow_stretch=True)
            self.ids.w_carousel.add_widget(image)
        self._keyboard = None
        self.state = "select"
    
    def on_pre_enter(self):
        Clock.schedule_interval(self._tick, 1.)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.start_select()
        
    def _tick(self, *args):
        if self.state is "select":
            self.ids.w_carousel.load_next()
        else:
            self.state -= 1
            self.ids.w_info.text = "已選擇！ 在 %d 秒內可按鍵重選"%self.state
            self.ids.w_info.font_size = 40
            if self.state == 0:
                root.art_style_filename = self.ids.w_carousel.current_slide.source
                root.current = 'camera'
                
    def _keyboard_closed(self):
        print("style keyboard closed")
        if self._keyboard:
            print("style keyboard unbind")
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard = None

    def on_leave(self):
        Clock.unschedule(self._tick)
        self._keyboard_closed()
        
    def start_select(self):
        self.state = "select"
        self.ids.w_info.text = "按鍵選擇一種藝術風格"
        self.ids.w_info.font_size = 64
        
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        self._on_clicked()
    
    def on_touch_down(self, touch):
        self._on_clicked()
        
    def _on_clicked(self):
        if self.state is "select":
            self.state = 3
        else:
            self.start_select()
            
        return True



class ArtApp(App):

    def build(self):
        global root
        root = ScreenManager(transition=SlideTransition(duration=1.))
        root.add_widget(StyleScreen(name="style"))
        root.add_widget(MenuScreen(name='menu'))
        root.add_widget(CameraScreen(name='camera'))
        root.add_widget(ProcessScreen(name='process'))
        root.art_style_filename = None
        return root

if __name__ == '__main__':
    ArtApp().run()
