import numpy as np
import cv2 as cv

class sudoku_image:
    '''use to open an image of a sudoku, transform it so that only
    the sudoku is visible and the values can be read '''

    def __init__(self, image = None):
        self.original = cv.imread(image)
        self.mask = None # the mask we will use to show only the relevant bits
        self.result = None # the final output, in greyscale
        self.res_rgb = None # the final output, in rgb

        self.__pre_processing__()
        grid_point_img = cv.bitwise_and(self.__find_verticals__(),self.__find_horizontals__())
        self.gridpoints = self.__find_gridpoints__(grid_point_img)
    def __pre_processing__(self):
        img =       cv.GaussianBlur(self.original, (5, 5), 0) #blur img to remove noise
        gray =      cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to greyscale for easy processing
        self.mask = np.zeros((gray.shape), np.uint8) # create a complely black mask the same size as the image
        kernel1 =   cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))

        close =     cv.morphologyEx(gray,cv.MORPH_CLOSE, kernel1)
        div =       np.float32(gray)/(close)
        image =     np.uint8(cv.normalize(div, div, 0, 255, cv.NORM_MINMAX)) # normalise the image to remove extreme shadows
        self.res_rgb = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        # find the outline of the sudoku
        thresh = cv.adaptiveThreshold(image,255,0,1,19,2)
        contour, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_cnt = None
        for cnt in contour:
            area = cv.contourArea(cnt)
            if area > 1000:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt

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

    def __find_gridpoints__(self):
        

if __name__ == '__main__':
    my_sudoku = sudoku_image('sudoku_images/test1.jpg')
    cv.imshow('test', my_sudoku.result)
    cv.imshow('grid_points', my_sudoku.grid_points)

    cv.waitKey(0)
    cv.destroyAllWindows()
