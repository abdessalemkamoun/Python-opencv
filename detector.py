import cv2
import os
import numpy as np
from boot import Boot


class DetectorFace:
    """
     Training Data
     The more images used in training the better.
    Normally a lot of images are used for training a face recognizer so that it can learn different looks of the same person, for example with glasses, without glasses, laughing, sad, happy, crying, with beard, without beard etc. To keep our tutorial simple we are going to use only 12 images for each person.
     So our training data consists of total 2 persons with 12 images of each person. All training data is inside _`training-data`_ folder. _`training-data`_ folder contains one folder for each person and **each folder is named with format `pLabel (e.g. p1, p2)` where label is actually the integer label assigned to that person**. For example folder named s1 means that this folder contains images for person 1. The directory structure tree for training data is as follows:
     training-data
     |-------------- p1
     |               |-- 1.jpg
     |               |-- ...
     |               |-- 12.jpg
     |-------------- p2
     |               |-- 1.jpg
     |               |-- ...
     |               |-- 12.jpg

     The _`test-data`_ folder contains images that we will use to test our face recognizer after it has been successfully trained.
     As OpenCV face recognizer accepts labels as integers so we need to define a mapping between integer labels and persons actual names so below I am defining a mapping of persons integer labels and their respective names.
     **Note:** As we have not assigned `label 0` to any person so **the mapping for label 0 is empty**.
     In[2]:
     there is no label 0 in our training data so subject name for index/label 0 is empty
    """

    def __init__(self):
        self.boot = Boot('jaw', 'training-data')
        employee_index, blacklist_index = self.boot.generate_training_data()
        self.employee_range = range(1, employee_index + 1)
        self.blacklist_range = range(employee_index + 1, employee_index + 1 + blacklist_index)
        self.subjects = self.boot.labels
        self.faces = []
        self.labels = []
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        # or use EigenFaceRecognizer by replacing above line with
        # face_recognizer = cv2.face.EigenFaceRecognizer_create()
        # or use FisherFaceRecognizer by replacing above line with
        # face_recognizer = cv2.face.FisherFaceRecognizer_create()
        self.video = cv2.VideoCapture('rtsp://admin:Esprit@192.168.1.64:554')
        self.training_faces()

    @staticmethod
    def detect_face(img, multi=False):
        """
        # convert the test image to gray image as opencv face detector expects gray images
        # load OpenCV face detector, I am using Haar classifier
        # let's detect multiscale (some images may be closer to camera than others) images
        # result is a list of faces
        # if no faces are detected then return original img
        # under the assumption that there will be only one face,
        # extract the face area

        :return: only the face part of the image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        if len(faces) == 0:
            return None, None
        if multi:
            fcs = []
            for face in faces:
                (x, y, w, h) = face
                fcs.append((gray[y:y + w, x:x + h], face))
            return fcs
        (x, y, w, h) = faces[0]
        return gray[y:y + w, x:x + h], faces[0]

    def prepare_training_data(self, data_folder_path):
        """
        I am using OpenCV's **LBP face detector**, I convert the image to grayscale because most operations in OpenCV
        are performed in gray scale, then on _line 8_ I load LBP face detector using `cv2.CascadeClassifier` class.
        After that on _line 12_ I use `cv2.CascadeClassifier` class' `detectMultiScale` method to detect all the
        faces in the image. on _line 20_, from detected faces I only pick the first face because in one image there
        will be only one face (under the assumption that there will be only one prominent face). As faces returned by
        `detectMultiScale` method are actually rectangles (x, y, width, height) and not actual faces images so we
        have to extract face image area from the main image. So on _line 23_ I extract face area from gray image and
        return both the face image area and face rectangle. Now you have got a face detector and you know the 4 steps
        to prepare the data, so are you ready to code the prepare data step? Yes? So let's do it.
        this function will read all persons' training images, detect face from each image
        and will return two lists of exactly same size, one list
         of faces and another list of labels for each face

        :param data_folder_path:
        :return:
        """
        # ------STEP-1--------
        # get the directories (one directory for each subject) in data folder
        dirs = os.listdir(data_folder_path)
        # let's go through each directory and read images within it
        for dir_name in dirs:
            # our subject directories start with letter 'p' so
            # ignore any non-relevant directories if any
            if not dir_name.startswith("p"):
                continue;

            # ------STEP-2--------
            # extract label number of subject from dir_name
            # format of dir name = slabel
            # , so removing letter 'p' from dir_name will give us label
            label = int(dir_name.replace("p", ""))

            # build path of directory containin images for current subject subject
            # sample subject_dir_path = "training-data/p1"
            subject_dir_path = data_folder_path + "/" + dir_name

            # get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            # ------STEP-3--------
            # go through each image name, read image,
            # detect face and add face to list of faces
            for image_name in subject_images_names:
                # ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue;

                # build image path
                # sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                # read image
                image = cv2.imread(image_path)

                # display an image window to show the image
                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                cv2.waitKey(100)

                # detect face
                face, rect = self.detect_face(image)

                # ------STEP-4--------
                # for the purpose of this tutorial
                # we will ignore faces that are not detected
                if face is not None:
                    # add face to list of faces
                    self.faces.append(face)
                    # add label for this face
                    self.labels.append(label)

    def training_faces(self):
        print("Preparing data...")
        self.prepare_training_data("training-data")
        print("Data prepared")
        # print total faces and labels
        print("Total faces: ", len(self.faces))
        print("Total labels: ", len(self.labels))
        self.face_recognizer.train(self.faces, np.array(self.labels))


    @staticmethod
    def draw_rectangle(img, rect):
        """
        function to draw rectangle on image
        according to given (x, y) coordinates and
        given width and heigh

        :param img:
        :param rect:
        :return:
        """
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    @staticmethod
    def draw_text(img, text, x, y):
        """
        function to draw text on give image starting from
        passed (x, y) coordinates.

        :param img: image
        :param text: Name of the person
        :param x:
        :param y:
        :return:
        """
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def predict(self, test_img):
        """
        this function recognizes the person in image passed
        and draws a rectangle around detected face with name of the
        subject.
        1- make a copy of the image as we don't want to chang original image
        2- detect face from the image
        3- predict the image using our face recognizer
        4- get name of respective label returned by face recognizer
        5- draw a rectangle around face detected
        6- draw name of predicted person
        :param test_img:
        :return:
        """
        if test_img is None:
            return [test_img, self.img_to_byte(test_img), 0]
        img = test_img.copy()
        fcs = self.detect_face(img, multi=True)
        for fc in fcs:
            if fc is None:
                return [img, self.img_to_byte(img), 0]
            face, rect = fc
            label, confidence = self.face_recognizer.predict(face)
            if label in self.blacklist_range:
                label_text = self.subjects[label]
                self.draw_rectangle(img, rect)
                self.draw_text(img, label_text, rect[0], rect[1] - 5)
                #TODO: Notification for a blacklist person
            elif label not in self.employee_range:
                self.predict_customer_number(face)

        return [img, self.img_to_byte(img), 0]

    @staticmethod
    def img_to_byte(img):
        """
        We are using Motion JPEG, but OpenCV defaults to capture raw images,
        so we must encode it into JPEG in order to correctly display the
        video stream
        :param img:
        :return:
        """
        if img is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def get_frame(self):
        success, image = self.video.read()
        return image
