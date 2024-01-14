import cv2
import time
import os
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(1000)


class Detector:
    def __init__(self) -> None:
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

            # color list
            self.colorList = np.random.uniform(0, 255, (len(self.classesList)))
            print(len(self.classesList), len(self.colorList))

    def downloadModel(self, modelUrl):
        fileName = os.path.basename(modelUrl)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./pretrained_models"
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
            print(f"Directory '{self.cacheDir}' created successfully.")
        else:
            print(f"Directory '{self.cacheDir}' already exists.")

        # os.makedirs(self.cacheDir)
        get_file(fname=fileName, origin=modelUrl, cache_dir=self.cacheDir,
                 cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("#######################################")
        print("loading Model" + self.modelName)
        print("#######################################")
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(
            self.cacheDir, "checkpoints", self.modelName, "saved_model"))

    def creatBoundingBox(self, image):
        imH, imW, imC = image.shape
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxes = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(
            np.int32)
        classScores = detections['detection_scores'][0].numpy()
        bboxId = tf.image.non_max_suppression(
            bboxes, classScores, max_output_size=50, iou_threshold=0.5, score_threshold=0.5)
        if len(bboxId) != 0:
            for i in range(0, len(bboxId)):
                bbox = tuple(bboxes[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]
                classlabelText = self.classesList[classIndex]
                classColor = self.colorList[classIndex]
                displayText = f'{classlabelText}:{classConfidence}%'
                ymin, xmin, ymax, xmax = bbox
                # print(bbox)
                ymin, xmin, ymax, xmax = (
                    ymin*imH, xmin*imW, ymax*imH, xmax*imW)
                ymin, xmin, ymax, xmax = int(ymin), int(
                    xmin), int(ymax), int(xmax)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                              color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin-10),
                            cv2.QT_FONT_NORMAL, 1, classColor, 1)
                ##############################################
                lineWidth = min(int((xmax-xmin)*0.2), int((ymax-ymin)*0.2))
                cv2.line(image, (xmin, ymin), (xmin+lineWidth, ymin),
                         classColor, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin+lineWidth),
                         classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax-lineWidth, ymin),
                         classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin+lineWidth),
                         classColor, thickness=5)
                ##############################################
                cv2.line(image, (xmin, ymax), (xmin+lineWidth, ymax),
                         classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax-lineWidth),
                         classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax-lineWidth, ymin),
                         classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin+lineWidth),
                         classColor, thickness=5)
            return image

        # resized_image = cv2.resize(image, (new_width, new_height))
        return

    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(1)
        image = cv2.resize(image, (500, 500))
        bbox = self.creatBoundingBox(image)
        cv2.imshow("img", bbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Faild.........")

        sucess, image = cap.read()
        print(image.shape)

        startTime = 0
        while sucess:
            currenTime = time.time()
            fps = 1/(currenTime-startTime)
            startTime = currenTime

            cv2.putText(image, f"FPS:{str(fps)}", (20, 70),
                        cv2.QT_FONT_NORMAL, 1, (0, 255, 0), 1)
            bbox = self.creatBoundingBox(image)
            cv2.imshow("img", bbox)

            key = cv2.waitKey(90) & 0xFF
            if key == ord('q'):
                break
            (sucess, image) = cap.read()
        cv2.destroyAllWindows()
