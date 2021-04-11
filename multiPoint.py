from tqdm import tqdm
import cv2, pickle
import tensorflow as tf
import numpy as np
from multiprocessing import Pool

class imagePreProcess():
  def __init__(self, original_images):
    if type(original_images) != list:
      self.original_images = [original_images]
    else:
      self.original_images = original_images

  def mainEntry(self, poolSize=4, pickleAfterWhite=False, picklePathAfterWhite=None, protoText=None, caffeModel=None):
    with Pool(poolSize) as p:
          whiteImages = list(tqdm(p.imap(self.flaggingWhiteFrames, self.original_images), total=len(self.original_images), position=0, leave=True))
    
    temp = [x for x in whiteImages if x is not None]
    afterFlagged = list(set(self.original_images) - set(temp))
    if pickleAfterWhite and (picklePathAfterWhite is not None):
      self.picklizingFunc(afterFlagged, picklePathAfterWhite)
    else:
      print('Please give valid path... and skipping to next step.')
    self.protoText, self.caffeModel = protoText, caffeModel
    # print(f'\nLoss of images: {1-((len(afterFlagged)/len(self.original_images)) * 100)}')
    # self.faceDetectedPaths, self.faceDetectedBox, self.failedInstances = [], [], []
    self.failedInstances = []
    with Pool(poolSize) as p:
        self.pathAndBoxes = list(tqdm(p.imap(self.face_detector, afterFlagged), total=len(afterFlagged), position=0, leave=True))
    # return self.pathAndBoxes
    if len(self.failedInstances) > 0:
        print(f'Failed Instances: {len(self.failedInstances)}')
    with Pool(poolSize) as p:
        afterEmptyRemoved = list(tqdm(p.imap(self.removeEmptyBoxes, self.pathAndBoxes), total=len(self.pathAndBoxes), position=0, leave=True))
    temp = [x for x in afterEmptyRemoved if x is not None]
    self.afterEmptyRemoved = temp
    with Pool(poolSize) as p:
        final = list(tqdm(p.imap(self.checkTrueBoundingBox, self.afterEmptyRemoved), total=len(self.afterEmptyRemoved), position=0, leave=True))
    temp = [x for x in final if x is not None]
    print('Upto here')
    return temp


  def picklizingFunc(self, object, picklePath):
    print('Picklization starts....')
    with open(picklePath, 'wb') as handle:
      pickle.dump(object, handle)
    print('Picklization done!')


  def flaggingWhiteFrames(self, x):
    temp = tf.io.read_file(x)
    temp = tf.image.decode_jpeg(temp, channels=3)
    temp = temp.numpy().astype(np.uint8)
    if np.mean(temp) == 255:
      return x
    else:
      return 

  def face_detector(self, imgPath):
    # print(self.protoText, self.caffeModel)
    cv2DNN = cv2.dnn.readNetFromCaffe(self.protoText, self.caffeModel)
    x = tf.io.read_file(imgPath)
    image = tf.image.decode_jpeg(x, channels=3)
    image = image.numpy().astype(np.uint8)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cv2DNN.setInput(blob)
    detections = cv2DNN.forward()
    curr = []
    for i in range(0, detections.shape[2]):
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      confidence = detections[0, 0, i, 2]
      # If confidence > 0.5, save it as a separate file
      if (confidence > 0.5):
        # frame = x[startY:endY, startX:endX]
        final3, final4 = endY - startY, endX - startX
        # For detecting any thing less than 0 error
        if final3 < 0 or final4 < 0:
          # Appending for clearly 
          self.failedInstances.append(imgPath)
        curr.append([startY, startX, final3, final4])
    return imgPath, curr

  def removeEmptyBoxes(self, imageAndBox):
    img, box = imageAndBox[0], imageAndBox[1]
    if len(box) > 0:
      return (img, box)


  def checkTrueBoundingBox(self, imageAndBox):
    imgPath, boxes = imageAndBox[0], imageAndBox[1] 
    img = tf.io.read_file(imgPath)
    img = tf.image.decode_jpeg(img, channels=3)
    height, width = img.numpy().shape[0], img.numpy().shape[1]
    curr = []
    for box in boxes:
        if box[0] > 0 and box[1] > 0:
          compare_height, compare_width = height - box[2], width - box[3]
          if compare_height > box[2] and compare_width > box[3] and (box[1] + box[3]) <= width and (box[0] + box[2]) <= height:
            curr.append(box)
    if len(curr) > 0:
      return (imgPath, curr)

if __name__ == '__main__':
  imagePreProcess()