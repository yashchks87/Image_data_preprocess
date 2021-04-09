from tqdm import tqdm
import cv2, pickle
import tensorflow as tf
import numpy as np

class imagePreProcess():
  def __init__(self, original_images, read=False):
    if type(original_images) != list:
      	self.original_images = [original_images]
    else:
      	self.original_images = original_images
    if read:
		    self.readImages = self.read_images(original_images)

  # Read all images and store them in array itself
  def read_images(self, allFiles):
  	readImages, failedInstances = [], []
  	for x in tqdm(allFiles, leave=True, position=0):
      try:
    		img = tf.io.read_file(x)
    		img = tf.image.decode_jpeg(img, channels = 3)
    		readImages.append(img.numpy())
      except:
        failedInstances.append(x)
        continue
    print(f'Number of files with issues: {len(failedInstances)}')
  	return readImages

  # This function will return the file paths which has full white pictures
  def flaggingWhiteFrames(self, isPickled=True, picklePath=None):
    filePaths, readImages = self.original_images, self.readImages
    resultPath, resultArray, whiteImages = [], [], []
    for x in tqdm(range(len(filePaths)), leave=True, position=0):
    	temp = readImages[x].astype(np.uint8)
    	meanValue = np.mean(temp)
    	if meanValue != 255:
    		resultPath.append(filePaths[x])
    		resultArray.append(readImages[x])
    	elif meanValue == 255:
    		whiteImages.append(filePaths[x])
    if isPickled:
      assert(picklePath != None)
      print('Picklization starts....')
      with open(picklePath, 'wb') as handle:
        pickle.dump(resultPath, handle)
      print('Picklization done!')
    self.afterRemovedWhites, self.readImagesAfterWhites = resultPath, resultArray
    return self.afterRemovedWhites, whiteImages

  def face_detector(self, protoText, caffeModel):
    try:
      cv2DNN = cv2.dnn.readNetFromCaffe(protoText, caffeModel)
    except:
      print('\nERROR: Please put right path for proto and caffe.')
    filePaths, readImages = self.afterRemovedWhites, self.readImagesAfterWhites

    print(f'\nLoss of images: {(1-(len(self.afterRemovedWhites)/len(self.original_images)))*100} %')
    temp, failedInstances = [], []
    print('******Face detection starts******')
    for x in tqdm(range(len(filePaths)), leave=True, position=0):
      image = readImages[x].astype(np.uint8)
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
            failedInstances.append(filePaths[x])
          curr.append([startY, startX, final3, final4])
      temp.append(curr)
    self.detectedArray = temp
    if len(failedInstances) > 0:
      print('failedInstances are found.')
    return temp
  
  def removeEmptyBoxes(self, isPickled=False, picklePath=None):
    updatedFilePaths, temp, readImages = self.afterRemovedWhites, self.detectedArray, self.readImagesAfterWhites
    print(f'\nLoss of images: {1-(len(temp)/len(self.original_images))}')
    updatedFiles, updatedBox, updatedReadImages = [], [], []
    for x in tqdm(range(len(temp)), position=0, leave=True):
      if len(temp[x]) > 0:
        updatedFiles.append(updatedFilePaths[x])
        updatedReadImages.append(readImages[x])
        updatedBox.append(temp[x])
    if isPickled:
      assert(picklePath != None)
      print('Picklization starts....')
      result = dict(zip(updatedFiles, updatedBox))
      with open(picklePath, 'wb') as handle:
        pickle.dump(result, handle)
      print('Picklization done!')
    self.filePathAEBox, self.BoxAEBox, self.readImagesAEBox = updatedFiles, updatedBox, updatedReadImages
    return updatedFiles, updatedBox
  
  def checkTrueBoundingBox(self, isPickled=False, picklePath=None):
    clearBoxPath, clearBox = [], []
    updatedFiles, updatedBox, updatedReadImages = self.filePathAEBox, self.BoxAEBox, self.readImagesAEBox
    print(f'\nLoss of images: {(1-(len(updatedFiles)/len(self.original_images)))*100}')
    for x in tqdm(range(len(updatedFiles)), leave=True, position=0):
      height, width = updatedReadImages[x].shape[0], updatedReadImages[x].shape[1]
      curr = []
      for box in updatedBox[x]:
        if box[0] > 0 and box[1] > 0:
          compare_height, compare_width = height - box[2], width - box[3]
          if compare_height > box[2] and compare_width > box[3] and (box[1] + box[3]) <= width and (box[0] + box[2]) <= height:
            curr.append(box)
      if len(curr) > 0:
        clearBoxPath.append(updatedFiles[x])
        clearBox.append(curr)
    if isPickled:
      assert(picklePath != None)
      print('Picklization starts....')
      result = dict(zip(clearBoxPath, clearBox))
      with open(picklePath, 'wb') as handle:
        pickle.dump(result, handle)
      print('Picklization done!')
    return clearBoxPath, clearBox

if __name__ == '__main__':
  imagePreProcess()