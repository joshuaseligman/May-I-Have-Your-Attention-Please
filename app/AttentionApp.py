from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.graphics.texture import Texture

import cv2 as cv
import matplotlib.pyplot as plt

#Class that stores and saves information about the user's session
class User():
    def __init__(self):
        self.current_span = 0.0
        self.span_times = []
        self.is_attentive = False
        self.save_session_graph()
    def save_session_graph(self):
        fig, ax = plt.subplots()
        ax.plot(self.span_times)
        fig.savefig('session_graph.png')
        plt.close(fig)

#Class that handles everything eye detection
class EyeDetection():
    def __init__(self, user):
        self.user = user
        self.video = cv.VideoCapture(0)
        #Load the model
        self.eye_detection = cv.CascadeClassifier('cascade/cascade.xml')

    #Updates the predictions
    def update(self):
        #Get the current reading on the camera
        ret, frame = self.video.read()
        shape = frame.shape
        self.bounding_box = ((int(shape[1] * .3125), int(shape[0] / 6)), \
                            (int(shape[1] * .6875), int(shape[0] * 0.72)))
        #Get the model's predictions
        rectangles = self.eye_detection.detectMultiScale(frame, minNeighbors=50)
        rectangles_new = []
        for obj in rectangles:
            #Only keep the predictions inside the bounding box
            if self.is_in_boudning_box(obj):
                cv.rectangle(frame, (obj[0], obj[1]),\
                            (obj[0] + obj[2], obj[1] + obj[3]), (0, 255, 0), 2)
                rectangles_new.append(obj)
        #Determine the eye box
        eyes = self.get_eye_box(rectangles_new)
        if eyes != None:
            #Draw the possible eyes if not None
            cv.rectangle(frame, (eyes[0], eyes[1]), (eyes[2], eyes[3]),\
                        (0, 0, 255), 3)
            self.user.current_span += 0.1
            #Condition that prevents single-frame predictions from affecting the data
            if self.user.current_span >= 0.3:
                self.user.is_attentive = True
        else:
            if self.user.is_attentive:
                #Graph and save the span data
                self.user.is_attentive = False
                self.user.span_times.append(self.user.current_span)
                self.user.fig = plt.plot(self.user.span_times)
                self.user.save_session_graph()
            self.user.current_span = 0
        cv.rectangle(frame, self.bounding_box[0], self.bounding_box[1], \
                        (255, 0, 0), 2)
        #Convert the camera's view (with drawings) to a Texture for Kivy
        frame = cv.resize(frame, (int(Window.width * .8), int(Window.height * .6)))
        buf = cv.flip(frame, 0)
        buf_string = buf.tostring()
        tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        tex.blit_buffer(buf_string, colorfmt='bgr', bufferfmt='ubyte')
        return tex

    #Determines if the rect parameter is in the bounding box
    def is_in_boudning_box(self, rect):
        if rect[0] < self.bounding_box[0][0]:
            return False
        elif rect[1] < self.bounding_box[0][1]:
            return False
        elif rect[0] + rect[2] > self.bounding_box[1][0]:
            return False
        elif rect[1] + rect[3] > self.bounding_box[1][1]:
            return False
        else:
            return True

    #Determines if 2 predictions are not overlapping minus a slight acceptable overlap
    def are_separate(self, r1, r2):
        if r1[0] <= r2[0]:
            if r1[1] <= r2[1]:
                if r1[0] + r1[2] >= r2[0] - 10 and r1[1] + r1[3] >= r2[1] - 10:
                    return False
                else:
                    return True
            else:
                if r1[0] + r1[2] >= r2[0] - 10 and r1[1] >= r2[1] - 10:
                    return False
                else:
                    return True
        else:
            return self.are_separate(r2, r1)  

    #Gets the box with both eyes
    def get_eye_box(self, rects):
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if self.are_separate(rects[i], rects[j]):
                    x1 = rects[i][0] if rects[i][0] < rects[j][0] else rects[j][0]
                    y1 = rects[i][1] if rects[i][1] < rects[j][1] else rects[j][1]
                    x2 = rects[j][0] + rects[j][2] if x1 == rects[i][0] else rects[i][0] + rects[i][2]
                    y2 = rects[j][1] + rects[j][3] if y1 == rects[i][1] else rects[i][1] + rects[i][3]
                    if abs(rects[i][1] - rects[j][1]) < 50:
                        if x1 == rects[i][0]:
                            if abs(rects[j][0] - rects[i][0] - rects[i][2]) < 50:
                                return [x1, y1, x2, y2]
                        else:
                            if abs(rects[i][0] - rects[j][0] - rects[j][2]) < 50:
                                return [x1, y1, x2, y2]
        return None

#Class manages the live results page
class LiveResults(FloatLayout):
    def __init__(self, user, **kwargs):
        super().__init__(**kwargs)
        self.user = user
        self.ed = EyeDetection(self.user)
        self.title = Label(text='Live Results', font_size='45sp', \
                            pos_hint={'x': -.3, 'y': .4})

        self.key_label = Label(text='Key:', font_size='30sp', pos_hint={'x': -.41, 'y': -.33})
        self.blue_label = Label(text='Blue - Area your eyes should be located\n           in to promote good listening behaviors', \
                                font_size='20sp', pos_hint={'x': -.1, 'y': -.355})
        self.green_label = Label(text='Green - Possible eye locations', \
                                    font_size='20sp', pos_hint={'x': -.18, 'y': -.415})
        self.red_label = Label(text='Red - Predicted location of your eyes', \
                                font_size='20sp', pos_hint={'x': -.145, 'y': -.46})
        
        self.img = Image()
        self.stats_button = Button(text='Show Statistics', font_size='25sp',\
                        pos_hint={'x': .68, 'y': .83}, \
                        size_hint=(.3, .15))
        self.home_button = Button(text='Return Home', font_size='25sp', \
                        pos_hint={'x': .68, 'y': 0.02}, \
                        size_hint=(.3, .15))

        self.add_widget(self.title)
        self.add_widget(self.img)
        self.add_widget(self.key_label)
        self.add_widget(self.blue_label)
        self.add_widget(self.green_label)
        self.add_widget(self.red_label)
        self.add_widget(self.stats_button)
        self.add_widget(self.home_button)
    
    def update(self, dt):
        #Update the visible frame
        self.img.texture = self.ed.update()

#Class handles the statistics of the session
class Statistics(FloatLayout):
    def __init__(self, user, **kwargs):
        super().__init__(**kwargs)
        self.user = user
        
        self.btn = Button(text='Hide Statistics', font_size='25sp', \
                        pos_hint={'x': .68, 'y': .83}, \
                        size_hint=(.3, .15))
        self.add_widget(self.btn)

        self.title = Label(text='Session Information', font_size='45sp', pos_hint={'x': -.2, 'y': .4})
        self.add_widget(self.title)

        self.current_label = Label(text=f'Current Span: 0.0 seconds', \
                                font_size='30sp', halign='left', \
                                size_hint_x=None, width=int(Window.width * .48))
        self.current_label.bind(size=self.current_label.setter('text_size'))

        self.current_notes = Label(text='Time to focus!', font_size='20sp')
        self.current_notes.bind(size=self.current_notes.setter('text_size'))

        self.avg_label = Label(text=f'Average Span: 0.0 seconds', font_size='30sp', halign='left', size_hint_x=None, width=int(Window.width * .48))
        self.avg_label.bind(size=self.avg_label.setter('text_size'))

        self.avg_notes = Label(text='No data', font_size='20sp')
        self.avg_notes.bind(size=self.avg_notes.setter('text_size'))    

        self.long_label = Label(text=f'Longest Span: 0.0 seconds', font_size='30sp', halign='left', size_hint_x=None, width=int(Window.width * .48))
        self.long_label.bind(size=self.long_label.setter('text_size'))

        self.long_notes = Label(text='No data', font_size='20sp', width=int(Window.width * .52))
        self.long_notes.bind(size=self.long_notes.setter('text_size'))

        self.session_image = Image(source='session_graph.png', pos_hint={'x':-.12, 'y':.03}, \
                                    size_hint=(.8, .3))

        self.image_notes = Label(text='No data', font_size='20sp')
        self.image_notes.bind(size=self.image_notes.setter('text_size'))

        self.body = GridLayout(cols=2, row_force_default=True, \
                            row_default_height=int(Window.height * .146), pos_hint={'x':.05, 'y':-0.2})

        self.body.add_widget(self.current_label)
        self.body.add_widget(self.current_notes)
        self.body.add_widget(self.avg_label)
        self.body.add_widget(self.avg_notes)
        self.body.add_widget(self.long_label)
        self.body.add_widget(self.long_notes)
        self.body.add_widget(Label(size_hint_x=None, width=int(Window.width * .52)))
        self.body.add_widget(self.image_notes)
        self.add_widget(self.body)
        self.add_widget(self.session_image)

    #Update the stats live even if the user is not viewing the live page
    def update_stats(self, dt):
        self.current_label.text = f'Current Span: {round(self.user.current_span, 1)} seconds'
        if self.user.current_span > .77:
            self.current_notes.text = 'Way to be attentive!'
        else:
            self.current_notes.text = 'Time to focus!'

        if len(self.user.span_times) != 0:
            avg = sum(self.user.span_times) / len(self.user.span_times)
            self.avg_label.text = f'Average Span: {round(avg, 1)} seconds'
            if avg > 1:
                self.avg_notes.text = 'Way to go! Your attention span is higher\nthan average!'
            elif avg > 0.5:
                self.avg_notes.text = 'Nice work! Your average attention span\nis similar to that of the participants.'
            else:
                self.avg_notes.text = 'Try to pay attention! Your focus needs improvement.'

            long = max(self.user.span_times)
            self.long_label.text = f'Longest Span: {round(long, 1)} seconds'
            if long > 8:
                self.long_notes.text = 'Impressive! Your engagement and\npresence are recognized.'
            elif long > 5:
                self.long_notes.text = 'Your longest attention span is average.\nKeep up the good work!'
            else:
                self.long_notes.text = 'Try to rid yourself of any distractions.'

        self.session_image.reload()

        if len(self.user.span_times) < 5:
            self.image_notes.text = 'Not enough data. Keep working to\nget trends in your attention span.'
        else:
            avg_rate_of_change = (self.user.span_times[-1] - self.user.span_times[-5]) / self.user.span_times[-5]
            if avg_rate_of_change > 0:
                self.image_notes.text = 'Your recent trends are positive. Keep up\nthe good work.'
            else:
                self.image_notes.text = 'Your attention has recently seen negative results. You need to focus more.'

#Class handles the home page
class Home(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.line1 = Label(text='May I', font_size='100sp', \
                            pos_hint={'x': -.32, 'y': .4})
        self.line2 = Label(text='Have Your', font_size='100sp', \
                            pos_hint={'x': -.18, 'y': .2})
        self.line3 = Label(text='Attention,', font_size='100sp', \
                            pos_hint={'x': -.2, 'y': 0})
        self.line4 = Label(text='Please?', font_size='100sp', \
                            pos_hint={'x': -.255, 'y': -.2})
        self.us = Label(text='Creators: Josh Seligman, Meghan Urban, Michael Laffan', \
                        font_size='30sp', pos_hint={'x': 0, 'y': -.35})
        self.datathon = Label(text='Marist-IBM Datathon (April 23-25, 2021)', \
                                font_size='30sp', pos_hint={'x': 0, 'y': -.425})

        self.start_btn = Button(text='Start', font_size='50sp', \
                                pos_hint={'x': .68, 'y': .43}, \
                                size_hint=(.3, .15))
        self.quit_btn = Button(text='Quit', font_size='50sp', \
                                pos_hint={'x': .68, 'y': .26}, \
                                size_hint=(.3, .15))

        #Callback for quitting the app
        def quit_app(value):
            App.get_running_app().stop()
        self.quit_btn.bind(on_press=quit_app)
        
        self.logo = Image(source='logo.png', pos_hint={'x': .42, 'y': .65}, \
                            size_hint=(.8, .3))

        self.add_widget(self.line1)
        self.add_widget(self.line2)
        self.add_widget(self.line3)
        self.add_widget(self.line4)
        self.add_widget(self.us)
        self.add_widget(self.datathon)
        self.add_widget(self.start_btn)
        self.add_widget(self.quit_btn)
        self.add_widget(self.logo)

#Main class
class AttentionApp(App):
    def build(self):
        self.user = User()

        self.title = 'May I Have Your Attention, Please?'
        Window.clearcolor = (14 / 255, 38 / 255, 76 / 255, 1)

        #Manages the traversal of multiple pages
        self.screen_manager = ScreenManager()

        self.home_page = Home()
        screen = Screen(name='Home')
        screen.add_widget(self.home_page)
        self.screen_manager.add_widget(screen)

        self.live_page = LiveResults(self.user)
        screen = Screen(name='Live')
        screen.add_widget(self.live_page)
        self.screen_manager.add_widget(screen)

        self.stats_page = Statistics(self.user)
        screen = Screen(name='Stats')
        screen.add_widget(self.stats_page)
        self.screen_manager.add_widget(screen)

        #Moves between the home page and live results page
        def toggle_home_page(value):
            if self.screen_manager.current == 'Home':
                self.screen_manager.current = 'Live'
                Clock.schedule_interval(self.live_page.update, 1 / 10)
                Clock.schedule_interval(self.stats_page.update_stats, 1 / 10)
            else:
                self.screen_manager.current = 'Home'
                Clock.schedule_interval(self.live_page.update, 0)
                Clock.schedule_interval(self.stats_page.update_stats, 0)

        self.home_page.start_btn.bind(on_press=toggle_home_page)
        self.live_page.home_button.bind(on_press=toggle_home_page)

        #Moves between the live results page and the stats page
        def toggle_stats_page(value):
            if self.screen_manager.current == 'Live':
                self.screen_manager.current = 'Stats'
                Clock.schedule_interval(self.stats_page.update_stats, 1 / 10)
            else:
                self.screen_manager.current = 'Live'
                Clock.schedule_interval(self.stats_page.update_stats, 0)
            
        self.live_page.stats_button.bind(on_press=toggle_stats_page)
        self.stats_page.btn.bind(on_press=toggle_stats_page)
        
        return self.screen_manager

if __name__ == '__main__':
    app = AttentionApp()
    app.run()
    app.live_page.ed.video.release()
