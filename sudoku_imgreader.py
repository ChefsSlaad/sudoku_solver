import numpy as np
import cv2 as cv
from statistics import mean

class sudoku_image:
    '''use to open an image of a sudoku, transform it so that only
    the sudoku is visible and the values can be read '''

    def __init__(self, image = None):
        self.original = cv.imread(image)
        self.mask  =    None # the mask we will use to show only the relevant bits
        self.result =   None # the final output, in greyscale
        self.contour =  None
        self.res_rgb =  None # the final output, in rgb

        self.__pre_processing__()
        self.grid_point_img = cv.bitwise_and(self.__find_verticals__(),self.__find_horizontals__())
        self.gridpoints = self.__find_gridpoints__(self.grid_point_img)


    def __pre_processing__(self):
        img =           cv.GaussianBlur(self.original, (5, 5), 0) #blur img to remove noise
        gray =          cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to greyscale for easy processing
        self.mask =     np.zeros((gray.shape), np.uint8) # create a complely black mask the same size as the image
        kernel1 =       cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))

        close =         cv.morphologyEx(gray,cv.MORPH_CLOSE, kernel1)
        div =           np.float32(gray)/(close)
        image =         np.uint8(cv.normalize(div, div, 0, 255, cv.NORM_MINMAX)) # normalise the image to remove extreme shadows
        self.res_rgb =  cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        # find the outline of the sudoku
        thresh =        cv.adaptiveThreshold(image,255,0,1,19,2)
        contour, _ =    cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


        # find the largest contour in the set of all contours

        max_area = 0
        best_cnt = None
        for cnt in contour:
            area = cv.contourArea(cnt)
            if area > 1000:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt

        correction = 0.1
        epsilon = correction*cv.arcLength(best_cnt,True)
        self.contour = cv.approxPolyDP(best_cnt,epsilon,True)

        cv.drawContours(self.mask,[best_cnt],0,255,-1)
        cv.drawContours(self.mask,[best_cnt],0,0,2)

        self.result = cv.bitwise_and(image, self.mask)

    def __find_verticals__(self):
        kernelx = cv.getStructuringElement(cv.MORPH_RECT,(2,10))

        dx = cv.Sobel(self.result, cv.CV_16S,1,0)
        dx = cv.convertScaleAbs(dx)
        cv.normalize(dx,dx,0,255,cv.NORM_MINMAX)
        ret,close = cv.threshold(dx,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        close = cv.morphologyEx(close,cv.MORPH_DILATE,kernelx,iterations = 1)

        contour, hier = cv.findContours(close,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv.boundingRect(cnt)
            if h/w > 5:
                cv.drawContours(close,[cnt],0,255,-1)
            else:
                cv.drawContours(close,[cnt],0,0,-1)
        close = cv.morphologyEx(close,cv.MORPH_CLOSE,None,iterations = 2)
        return close.copy()

    def __find_horizontals__(self):

        kernely = cv.getStructuringElement(cv.MORPH_RECT,(10,2))
        dy = cv.Sobel(self.result, cv.CV_16S,0,2)
        dy = cv.convertScaleAbs(dy)
        cv.normalize(dy,dy,0,255,cv.NORM_MINMAX)
        ret,close = cv.threshold(dy,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        close = cv.morphologyEx(close,cv.MORPH_DILATE,kernely)

        contour, hier = cv.findContours(close,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv.boundingRect(cnt)
            if w/h > 5:
                cv.drawContours(close,[cnt],0,255,-1)
            else:
                cv.drawContours(close,[cnt],0,0,-1)

        close = cv.morphologyEx(close,cv.MORPH_DILATE,None,iterations = 2)
        return close.copy()

    def __find_gridpoints__(self, grid_point_img):
        contour, _  = cv.findContours(grid_point_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        centroids   = []
        for cnt in contour:
#            print(cnt)
            mom = cv.moments(cnt)
            try:
                (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
                cv.circle(self.result,(x,y),4,(0,255,0),-1)
                centroids.append((x,y))
            except ZeroDivisionError:
                pass

        centroids = np.array(centroids, dtype = np.float32)
        centroids[np.argsort(centroids[0:,1])]

#        print(centroids)
#        print(self.contour)
#        print(len(centroids))

        return centroids

#        print(len(centroids))
#        print(centroids[np.argsort(centroids[0:,1])])
#        c = centroids.reshape((100,2))
#        c2 = c[np.argsort(centroids[:,1])]
#        print(c2)
#        b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
#        b2 = b.reshape((10,10,2))
#        return b2

def resize_img(image, pct):
    return cv.resize(image, (int(image.shape[1]*pct), int(image.shape[0]*pct)))

def test_image(image, percentage=0.2):
    sudoku = sudoku_image(image)
    orig = sudoku.original
    grid = cv.cvtColor(sudoku.grid_point_img, cv.COLOR_GRAY2BGR)

    contour_gridpoints = [(c[0][0],c[0][1]) for c in sudoku.contour]
    x_avg = mean(c[0][0] for c in sudoku.contour)
    y_avg = mean(c[0][1] for c in sudoku.contour)

    grid_array = np.zeros(shape = (10,10,2), dtype = np.int32)

    # draw a polygo around the contour
    cv.polylines(orig,[sudoku.contour],True,(0,255,0),7)

    # add number to each corner of the grid
#    for i, c  in enumerate(sudoku.contour):
#        cv.putText(orig, str(i), tuple(c[0]), cv.FONT_HERSHEY_SIMPLEX, 3,(255,0,0),5)

    o_pnt = sudoku.contour[1][0]
    x_end = sudoku.contour[0][0]
    y_end = sudoku.contour[2][0]

    cv.putText(orig, 'o_pnt', tuple(o_pnt), cv.FONT_HERSHEY_SIMPLEX, 3,(255,0,0),5)
    cv.putText(orig, 'x_end', tuple(x_end), cv.FONT_HERSHEY_SIMPLEX, 3,(255,0,0),5)
    cv.putText(orig, 'y_end', tuple(y_end), cv.FONT_HERSHEY_SIMPLEX, 3,(255,0,0),5)



    vector = [[int(x),int(y)] for x, y in zip(np.linspace(o_pnt[0], x_end[0],10), np.linspace(o_pnt[1], x_end[1],10))]
    add_vector = [[int(x),int(y)] for x, y in zip(np.linspace(0, y_end[0]-o_pnt[0],10), np.linspace(0, y_end[1]-o_pnt[1],10))]
#    y_vector = [[int(x),int(y)] for x, y in zip(np.linspace(o_pnt[0], y_end[0],10), np.linspace(o_pnt[1], y_end[1],10))]

#    print(vector)
    for i, v in enumerate(add_vector):
        row = [[int(vec[0]+v[0]) , int(vec[1]+v[1])] for vec in vector]

        grid_array[i] = row

    print(grid_array)

    for gp in grid_array.reshape((100,2)):
        cv.circle(orig, tuple(gp), 10, (0,0,255), -1)

#    for gp in vector:
#        cv.circle(orig, tuple(gp), 10, (0,0,255), -1)
#    for gp in y_vector:
#        cv.circle(orig, tuple(gp), 10, (0,0,255), -1)



    # show all the found gridpoints
    #for gp in sudoku.gridpoints:
    #    cv.circle(orig, tuple(gp), 10, (0,0,255), -1)

    # combine both fimages
    combiImg = np.hstack((orig, grid))
#    print(sudoku_image.gridpoints)
    cv.imshow(image, resize_img(combiImg, percentage))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    import os
    path = '/home/marc/projects/sudoku_solver/sudoku_images/'
    for f in os.listdir(path):
        file_str = path+f
        print('attempting to open', file_str)
        test_image(file_str)
