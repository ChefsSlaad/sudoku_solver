import os
import cv2 as cv
import numpy as np
import sys
from sudoku_imgreader import sudoku_image

def classify_digits(path = None, sampleData = None, responseData = None, imagesData = None):



    if path == None:
        path = '/home/marc/projects/sudoku_solver/sudoku_images/'
    if responseData == None:
        responseData = 'generalresponses.data'
    if sampleData == None:
        sampleData = 'generalsamples.data'
    if imagesData == None:
        imagesData = 'generalimages.data'

    files = list(f for f in os.listdir(path) if f.endswith('.jpg'))
    responses = []
    sample = np.empty((0,2500))
    images = np.empty((0,10000))

    for file in sorted(files):
        print('\nprocessing', file, end = ' ')
        s = sudoku_image(path+file)
        for i, img in enumerate(s.cell_images):
            #pre-processing
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray,(5,5),0)
            thresh = cv.adaptiveThreshold(blur,255,1,1,11,2)

            # find contours that match the shape
            contours, _ = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                [x,y,w,h] = cv.boundingRect(cnt)
                if (cv.contourArea(cnt)>500 and w < 100):
#                    print(x,y,w,h, 'contour', cv.contourArea(cnt))
                    cv.rectangle(img,(x,y), (x+w, y+h), (0,255,0),6)
                    cv.imshow('im',img)
                    k = cv.waitKey(0)
                    if k ==27:
                        save_files(path, sampleData, responseData, imagesData)
                        sys.exit()

                    elif k in [46,49,50,51,52,53,54,55,56,57]:
                        responses.append(int(chr(k)))
                        roi = thresh[y:y+h,x:x+w]
                        roi_small = cv.resize(roi, (50,50))
                        img_small = cv.resize(thresh, (100,100))
                        sample = np.append(sample, roi_small.reshape((1,2500)), 0)
                        images = np.append(images, img_small.reshape((1,10000)),0)
                    print(chr(k), end = ' ')
                    cv.destroyAllWindows()
    save_files(path, sampleData, responseData, imagesData)

    def save_files(path, sampleData, responseData, imagesData):
        responses = np.array(responses,np.float32)
        responses = responses.reshape((responses.size,1))
        np.savetxt(path+sampleData,sample)
        np.savetxt(path+responseData,responses)
        np.savetxt(path+imagesData, images)


def create_model(samplesData, responseData, path = None):
    if path == None:
        path = '/home/marc/projects/sudoku_solver/sudoku_images/'

    samples = np.loadtxt(path+samplesData,np.float32)
    responses = np.loadtxt(path+responseData,np.float32)
    responses = responses.reshape((responses.size,1))

    model = cv.ml.KNearest_create()
    model.train(samples, cv.ml.ROW_SAMPLE, responses)

    path = '/home/marc/projects/sudoku_solver/sudoku_images/cell_images/'
    files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    for file in sorted(files):
        img = cv.imread(path+file, cv.IMREAD_GRAYSCALE)
        blur = cv.GaussianBlur(img,(5,5),0)
        thresh = cv.adaptiveThreshold(blur,255,1,1,11,2)

        contours,hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            [x,y,w,h] = cv.boundingRect(cnt)
            if cv.contourArea(cnt)>500 and w < 100:
                roi = thresh[y:y+h,x:x+w]
                roismall = cv.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                cv.putText(img,string,(x,y+h),0,1,(0,255,0))

                cv.imshow('image', img)
                cv.waitKey(0)
                cv.destroyAllWindows()
            else string = '.'


if __name__ == '__main__':

    create_model(samplesData='generalsamples.data',
                 responseData='generalresponses.data')

#    classify_digits(sampleData = 'Samples01.data',
#                    responseData = 'Respnses01.data',
#                    imagesData = 'Images01.data' )
