from tqdm import tqdm
import cv2, pickle
import tensorflow as tf
import numpy as np
from multiprocessing import Pool


def flaggingWhiteFramesAndDetectFaces(imgPath):
    try:
        temp = tf.io.read_file(imgPath)
        temp = tf.image.decode_jpeg(temp, channels=3)
        image = temp.numpy().astype(np.uint8)
        if np.mean(image) != 255:
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            cv2DNN.setInput(blob)
            detections = cv2DNN.forward()
            curr, failed = [], []
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
                        failed.append(imgPath)
                    curr.append([startY, startX, final3, final4])
            return imgPath, curr
        else:
            return
    except:
        # readingProblem.append(imgPath)
        return

def removeEmptyBoxes(imageAndBox):
	img, box = imageAndBox[0], imageAndBox[1]
	if len(box) > 0:
		return (img, box)

def checkTrueBoundingBox(imageAndBox):
	imgPath, boxes = imageAndBox[0], imageAndBox[1] 
	img = tf.io.read_file(imgPath)
	img = tf.image.decode_jpeg(img, channels=3)
	n = img.numpy()
	height, width = n.shape[0], n.shape[1]
	curr = []
	for box in boxes:
		if box[0] > 0 and box[1] > 0:
			compare_height, compare_width = height - box[2], width - box[3]
			if compare_height > box[2] and compare_width > box[3] and (box[1] + box[3]) <= width and (box[0] + box[2]) <= height:
				curr.append(box)
	if len(curr) > 0:
		return (imgPath, curr)

def start(files, poolSize, protoText, caffeModel, superClean=False, youWantDict=False):
  global cv2DNN
  print('CV2 Model Initiated')
  cv2DNN = cv2.dnn.readNetFromCaffe(protoText, caffeModel)
	# readingProblem = []
  print('*****************White frame removal and face detection starts*****************')
  with Pool(poolSize) as p:		
    faceDetected = list(tqdm(p.imap(flaggingWhiteFramesAndDetectFaces, files), total=len(files), position=0, leave=True))
  faceDetectedCleaned = [x for x in faceDetected if x is not None]
  print(f'\nLoss of files: {(1-(len(files)/len(faceDetectedCleaned)))*100}')
  print('\n***************** Removal of empty boxes started *****************')
  with Pool(poolSize) as p:
    removedEmptyBoxes = list(tqdm(p.imap(removeEmptyBoxes, faceDetectedCleaned), total=len(faceDetectedCleaned), position=0, leave=True))
  removedEmptyBoxesCleaned = [x for x in removedEmptyBoxes if x is not None]
  print(f'\nLoss of files: {(1-(len(files)/len(removedEmptyBoxesCleaned)))*100}')
  print('\n***************** Removal of erronous function started *****************')
  with Pool(poolSize) as p:
    imgAndBox = list(tqdm(p.imap(checkTrueBoundingBox, removedEmptyBoxesCleaned), total=len(removedEmptyBoxesCleaned), position=0, leave=True))
  imgAndBoxCleaned = [x for x in imgAndBox if x is not None]
  print(f'\nLoss of files: {(1-(len(files)/len(imgAndBoxCleaned)))*100}')
  if superClean:
    img, box = [x[0] for x in imgAndBoxCleaned], [x[1][0] for x in imgAndBoxCleaned]
    if youWantDict:
      return dict(zip(img, box))
    return (img, box)
  return imgAndBoxCleaned

if __name__ == '__main__':
	start()
