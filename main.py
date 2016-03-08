from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.carousel import Carousel
from kivy.uix.image import Image
from kivy.clock import Clock, mainthread
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import *
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
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect ("tcp://127.0.0.1:7788")
        
    def on_pre_enter(self):
        print("process screen", root.art_style.shape, root.photo_content.shape)
        self.art_style = root.art_style
        self.video_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        
        with self.canvas:
            #self.texture = self.video_texture
            Rectangle(texture=self.video_texture, pos=self.pos, size=(640, 480))
        image = cv2.flip(root.photo_content, 0)
        self.video_texture.blit_buffer(image.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.canvas.ask_update()
        threading.Thread(target = self.process).start()
        
    def process(self):
        self.socket.send_string(root.art_style_filename, zmq.SNDMORE)
        send_array(self.socket, root.photo_content)
        while 1:
            img = recv_array(self.socket)
            if img.shape == (0,):
                break
            self.socket.send_string("ok")
            self.update_progress(img)
        self.done()
        
    @mainthread
    def update_progress(self, img):
        #image = skimage.transform.resize(img, (480, 640), preserve_range=True).astype(np.uint8)
        image = cv2.resize(img, (640,480), interpolation=cv2.INTER_LANCZOS4)
        image = cv2.flip(image, 0)
        self.video_texture.blit_buffer(image.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.canvas.ask_update()
        
    @mainthread
    def done(self):
        #root.current="menu"
        self.ids.w_info.text = "完成"

        
    def on_leave(self):
        print("process leave")





class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__( **kwargs)
        cap = self.cap = cv2.VideoCapture(0)
        self.video_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        """
        sleep(2)
        bgs = []
        while len(bgs)<5:
            r, bg = cap.read()
            bg = cv2.flip(bg, 1)
            bg = bg.astype(float)
            if all(np.mean(np.linalg.norm(bg-bg0, axis=2)) < 30 for bg0 in bgs):
                bgs.append(bg)
                print("append")
            else:
                bgs=[bg]
                print("init")
            bg0 = bg
            sleep(0.2)
        self.bgs = bgs
        """
        with self.canvas:
            Rectangle(texture=self.video_texture, pos=self.pos, size=(640, 480))

        
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
        print("keyboard closed")
        if self._keyboard:
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard = None
        
    def start_wait(self):
        self.state = "wait"
        self.ids.w_info.text = "按鍵準備照相"
        self.ids.w_info.font_size = 64
    
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if self.state is "wait":
            self.state = 3*20
        else:
            self.start_wait()

    def _tick(self, *args):
        
        ret, img = self.cap.read()
        img = cv2.flip(img, 1)
        img2 = img
        """
        mask = None
        for bg in self.bgs:
            diff = (np.linalg.norm(img.astype(float)-bg, axis=2)/3).astype(np.uint8)
            r, mask0 = cv2.threshold(diff,6,255,cv2.THRESH_BINARY)
            if mask is None:
                mask = mask0
            else:
                mask = cv2.bitwise_or(mask, mask0)
        mask_inv = cv2.bitwise_not(mask)
        style2 = (cv2.bitwise_and(self.art_style, self.art_style, mask=mask_inv)*0.5).astype(np.uint8)
        img3 = cv2.bitwise_and(img, img, mask=mask)
        img2 = cv2.add(img3, style2)"""
        image = cv2.flip(img2, 0)
        self.video_texture.blit_buffer(image.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.canvas.ask_update()
        if self.state is not 'wait':
            if self.state == 0:
                print("照相")
                root.photo_content = img2
                root.art_style = self.art_style
                root.current = "process"
            if self.state%20 == 0:
                self.ids.w_info.text = "倒數 %d 照相 (按鍵取消重選)"%(self.state//30)
                self.ids.w_info.font_size = 64
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
            self.ids.w_info.text = "已選擇！ 不要的話請在 %d 秒內按鍵取消"%self.state
            self.ids.w_info.font_size = 40
            if self.state == 0:
                root.art_style_filename = self.ids.w_carousel.current_slide.source
                root.current = 'camera'
                
    def _keyboard_closed(self):
        print("keyboard closed")
        if self._keyboard:
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard = None
    def on_leave(self):
        Clock.unschedule(self._tick)
        self._keyboard_closed()
        
    def start_select(self):
        self.state = "select"
        self.ids.w_info.text = "按鍵選擇風格"
        self.ids.w_info.font_size = 64
        
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
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
