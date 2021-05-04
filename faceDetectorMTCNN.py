from tqdm import tqdm
import cv2, pickle
import tensorflow as tf
import numpy as np
from multiprocessing import Pool
import pickle
from facenet_pytorch import MTCNN

img = tf.io.gfile.glob('gs://wiki_data_faces/files1/*.jpg')

def flaggingWhiteFramesAndDetectFaces(imgPath):
    try:
        temp = tf.io.read_file(imgPath)
        temp = tf.image.decode_jpeg(temp, channels=3)
        shapes = temp.numpy()
        image = shapes.astype(np.uint8)
        height, width = shapes.shape[0], shapes.shape[1]
        if np.mean(image) != 255:
            temp = mtcnn.detect(image, landmarks=True)
            boxes = []
            # Takes length of confidence scores
            for x in range(len(temp[1])):
              if temp[1][x] > 0.9:
                # zero, one = int(temp[0][x][1]), int(temp[0][x][0])
                zero = 0 if int(temp[0][x][1]) < 0 else int(temp[0][x][1])
                one = 0 if int(temp[0][x][0]) < 0 else int(temp[0][x][0])
                two, three = int((temp[0][x][3]-temp[0][x][1])), int((temp[0][x][2]-temp[0][x][0]))
                if two > 0 and three > 0:
                  compareHeight, compareWidth = height - two, width - three
                  if compareHeight > two and compareWidth > three and (one+three) <= width and (zero + two) <= height:
                    boxes.append([zero, one, two, three])
            if len(boxes) > 0:
              return imgPath, boxes
            else:
              noBoxDetected.append(imgPath)
        else:
            whiteImages.append(imgPath)
    except:
        readingProblem.append(imgPath)



# Not testing use in actual
def mtCNNStart(files, poolSize, isPickled=False, picklePath=None):
    global readingProblem, mtcnn, whiteImages, noBoxDetected
    # global readingProblem, mtcnn, whiteImages, noBoxDetected
    readingProblem, whiteImages, noBoxDetected = [], [], []
    print('Python lists and global variable created.')
    print('************************** Model initialized **************************')
    mtcnn = MTCNN(keep_all=True, margin=20)
    print('************************** Main function starts **************************')
    print('Steps:\n1: Read the file from google cloud.\n2: Extract height and width of image.\n3: Detect weather there is only white image is there.')
    print('4: Create bounding box with confidenceof 90%')
    print('5: Make sure dimension does not exceed original image size.')
    with Pool(poolSize) as p:   
      faceDetected = list(tqdm(p.imap(flaggingWhiteFramesAndDetectFaces, files), total=len(files), position=0, leave=True))
    faceDetectedCleaned = [x for x in faceDetected if x is not None]
    p1 = '{:.2f}'.format((1-(len(faceDetectedCleaned)/len(files)))*100)
    print(f'\nLoss of files: {p1} %')
    print('Create multiple image paths and boxes tupled array.')
    multiplePaths = []
    for x in faceDetectedCleaned:
      for y in x[1]:
        multiplePaths.append((x[0], y))
    print(f'After creating multiple paths and array final count is: {len(multiplePaths)}')
    if isPickled:
      if picklePath is not None:
        with open(picklePath, 'wb') as handle:
          pickle.dump(multiplePaths, handle)
        print('Picklization done!')
      else:
        print('Please enter valid pickle path.')
    else:
        print('Returning object with picklized because isPickled value is given as False.')
    return multiplePaths

if __name__ == '__main__':
  mtCNNStart()
