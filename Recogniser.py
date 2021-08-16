from keras_vggface.vggface import VGGFace
import numpy as np
import mediapipe as mp
from scipy import spatial 
import cv2
import pickle
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf




def load_stuff(filename):
    '''Loads the pickle file
    
    Args:
        filename: path of the pickle file (feature.pickle)
        
    Return:
        stuff: varible holding the loaded '''
        
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


class FaceIdentify():
    ''' This class collects the user data through the webcam(laptop) and resize it to 
        224X224 and predictes the facial landmarks by VGGFace module and compares with 
        features in the pickel file using euclidean distance method. The label of is 
        accordance with the feature with min euclidean distance  '''

    
    def __init__(self, precompute_features_file="./features.pickle"):

        self.face_size = 224
        self.precompute_features_map = load_stuff(precompute_features_file)
        self.mpFaceDetection = mp.solutions.face_detection
        self.face_detection = self.mpFaceDetection.FaceDetection(0.50)
        self.model = VGGFace(model='resnet50', include_top=False,
                             input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max
        

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, t=2, l=20, color = (255, 255, 255)):
                   
        # Normalizing the size
        h,w,_= image.shape
        x, y, w, h = int((point.xmin * w )), int((point.ymin * h )), int((point.width * w )), int((point.height * h ))
        x1, y1 = x+w, y+h

        #Top left
        cv2.line(image, (x, y), (x+l, y), color, t )
        cv2.line(image, (x, y), (x, y+l), color, t )
        
        #Top right 
        cv2.line(image, (x1, y), (x1-l, y), color, t )
        cv2.line(image, (x1, y), (x1, y+l), color, t )
        
         #Bottom right 
        cv2.line(image, (x, y1), (x+l, y1), color, t )
        cv2.line(image, (x, y1), (x, y1-l), color, t )
        
         #Bottom left 
        cv2.line(image, (x1, y1), (x1-l, y1), color, t )
        cv2.line(image, (x1, y1), (x1, y1-l), color, t )
        
        cv2.putText(image, label, (x,y- 20 ), font, font_scale, color, t)
        
    def crop_face(self, imgarray, bound_box_c, margin=20, size=224):
        """
        Args:
            imgarray: full image
            bound_box_c: face detected area (xmin, ymin, widht, height)
            margin: add some margin to the face detected area to include a full head
            size: the result image resolution with be (size x size)
        Return:
         resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        (x, y, w, h) = int(bound_box_c.xmin * img_w ), int(bound_box_c.ymin * img_h ), int(bound_box_c.width * img_w ), int(bound_box_c.height * img_h )

        #Boundaries for cropping the image
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h

        #Cropping, Resizing and converting to 1D array
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)

        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def identify_face(self, features, threshold=100):
        ''' Computes the eculidean distance between the predited and generated features(.pkl)
            and'return the label having minimum euclidean distance '''
        
        distances = []
        for person in self.precompute_features_map:
            person_features = person.get("features")
            distance = spatial.distance.euclidean(person_features, features)
            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)
        if min_distance_value < threshold:
            return self.precompute_features_map[min_distance_index].get("name")
        else:
            return "Unknown"

    def detect_face(self, Instances=5):
        '''Detects 5(default) instance of the face and returns the most occured name'''

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        ptime = 0
        person=[]

        # infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                time.sleep(5)

            # Capture frame-by-frame
            ret, frame = video_capture.read()
            faces = self.face_detection.process(frame)
            ctime = time.time()
            fps = (1/(ctime - ptime))
            ptime = ctime 

            # placeholder for cropped 
            if faces.detections:
                face_imgs = np.empty((len(faces.detections), self.face_size, self.face_size, 3))
                for id, detection in enumerate(faces.detections):
                    face_img, cropped = self.crop_face(frame, detection.location_data.relative_bounding_box, margin=10, size=self.face_size)
                    (x, y, w, h) = cropped
                    face_imgs[id, :, :, :] = face_img
                if len(face_imgs) > 0: 
                    # generate features for each face
                    features_faces = self.model.predict(face_imgs)
                    predicted_names = [self.identify_face(features_face) for features_face in features_faces]
                    
                # draw results
                for id, detection in enumerate(faces.detections):
                    label = "{}".format(predicted_names[id])
                    person.append(label)
                    self.draw_label(frame, (detection.location_data.relative_bounding_box) , label)
                
            
            cv2.putText(frame , f'FPS : {int(fps)}', (10,50), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 1)
            cv2.imshow('Detecting', frame)

            # Keyboard Interrupt ('q' to quit)
            if cv2.waitKey(20) == ord('q') or len(person)>(Instances- 1):  
                break
        
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
        return person
