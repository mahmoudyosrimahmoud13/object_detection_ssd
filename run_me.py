from Detector import *
classFile = "coco.names"
imagePath = r"D:\obj_d\Busy people walking the city streets in London, HD Stock Footage.mp4"
modelUrl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelUrl=modelUrl)
detector.loadModel()
# detector.predictImage(imagePath)
detector.predictVideo(imagePath, 50)
