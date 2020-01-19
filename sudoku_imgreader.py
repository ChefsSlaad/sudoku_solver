import numpy as np
import os
import cv2 as cv


# https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv
# https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square

#example of finding what point is top-left, top-right etc.
# usefull to create a consistent grid in__find_gridpoints__
#https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/


class sudoku_image:
    '''use to open an image of a sudoku, transform it so that only
    the sudoku is visible and the values can be read '''

    def __init__(self, image = None):
        self.original =       cv.imread(image)
        self.image =          cv.GaussianBlur(self.original, (5, 5), 0) # blur image to remove imperfections and make it easier to convert
        self.result =         None # the final output, in greyscale
        self.contour =        None # the four cardinal points of the sudoku

        # populate the above parameters
        self.__pre_processing__()
        self.__process_img__()

    def __pre_processing__(self):
        '''
        pre-processing  mask all parts of the image that are not a sudoku and
        sets self.result to that Value
        forthermore, it calculates the cardinal points of the sodoku and returns
        those as self.contour
        '''

        gray =          cv.cvtColor(self.image, cv.COLOR_BGR2GRAY) # convert to greyscale for easy processing
        mask =          np.zeros((gray.shape), np.uint8) # create a complely black mask the same size as the image
        kernel1 =       cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))

        close =         cv.morphologyEx(gray,cv.MORPH_CLOSE, kernel1)
        div =           np.float32(gray)/(close)
        image =         np.uint8(cv.normalize(div, div, 0, 255, cv.NORM_MINMAX)) # normalise the image to remove extreme shadows

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
        epsilon =   correction*cv.arcLength(best_cnt,True)
        cont_pnts = cv.approxPolyDP(best_cnt,epsilon,True).reshape((4,2))

        self.contour = self.__order_winding__(cont_pnts)

        cv.drawContours(mask,[best_cnt],0,255,-1)
        cv.drawContours(mask,[best_cnt],0,0,2)

        self.result = cv.bitwise_and(image, mask)

    def __process_img__(self):

        self.warped_grid =    self.__find_gridpoints__(self.find_aprox_gridpoints())
        self.cell_images = self.__restore_warp__()

    def find_aprox_gridpoints(self):
        # detect the grid in the puzzle. a gridline is an aproximately straight
        # line that has a much larger width component than a height
        # component (or vice versa). The ratio should be at least 1:5

        # the function first looks for all the vertical gridlines and then the
        # horizontal gridlines.

        # finally it combines both to arrive at the final set of gridpoints

        ### verticals ###
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
        verticals = close.copy()

        ### horizontals ###
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
        horizontals = close.copy()

        ### join the two images to get the gridpoints ###
        return cv.bitwise_and(verticals,horizontals)

    def __find_gridpoints__(self, grid_point_img):

        #draw a contour around each of the gridoints we found
        #and calculate the center of mass for the contour and add that to the
        #list centroids
        #ignore contours with no mass
        contour, _  = cv.findContours(grid_point_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        centroids   = []
        for cnt in contour:
            mom = cv.moments(cnt)
            try:
                (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
                cv.circle(self.result,(x,y),4,(0,255,0),-1)
                centroids.append((x,y))
            except ZeroDivisionError:
                pass

        centroids = np.array(centroids, dtype = "float32")
        centroids[np.argsort(centroids[0:,1])]

        # next, based on the countour of the image, calculate the approx
        # location of the gridpoints, based on the fact that there should be
        # a 9x9 grid.
        #then check the found centroids againts the aprox location of the gridoints
        # and assign the closest value to that point
        # this also eliminates any false positives we may have found
        # and makes a best guess for any gridpoints we may have missed
        grid_array = np.zeros(shape = (10,10,2), dtype = "float32")
        o_pnt = self.contour[0]
        x_end = self.contour[1]
        y_end = self.contour[3]
        vector = [[x,y] for x, y in zip(np.linspace(o_pnt[0], x_end[0],10), np.linspace(o_pnt[1], x_end[1],10))]
        add_vector = [[x,y] for x, y in zip(np.linspace(0, y_end[0]-o_pnt[0],10), np.linspace(0, y_end[1]-o_pnt[1],10))]
        for i, v in enumerate(add_vector):
            row = np.around([[vec[0]+v[0] , vec[1]+v[1]] for vec in vector])
            grid_array[i] = row

        grid_array = grid_array.reshape((100,2))
        for i, gp in enumerate(grid_array):
            closest_gp, _ = self.__find_closest_dist__(centroids, gp)
            if closest_gp is not None:
                grid_array[i] = closest_gp

        grid_array = grid_array.astype('float32')
        return grid_array.reshape((10,10,2))

    def __find_closest_dist__(self, gridpoints, point):
        # find the cosest point in an array of gridpoints to another point
        # using the least squares method.
        closest_dist = float('inf')
        closest_gp = None
        for gp in gridpoints:
            dist_sqr = (gp[0]-point[0])**2 + (gp[1]-point[1])**2
            if dist_sqr < closest_dist:
                gp = closest_gp
                closest_dist = dist_sqr
        return closest_gp, closest_dist

    def __restore_warp__(self):
        # unwarp a sudoku cell and retusn an array of cells that can be easily
        # compined to create a new image


        #size is the smallest dimension of the sudoku devided by 9
        cell_size = int(min( (max(self.contour[:,0])-min(self.contour[:,0]))//9,
                             (max(self.contour[:,1])-min(self.contour[:,1]))//9))
#        print('cell_size =', cell_size)

        result_cell = np.array([[0,0],
                                [cell_size-1,0],
                                [cell_size-1,cell_size-1],
                                [0,cell_size-1] ],
                                dtype="float32")

        ys, xs, _ = self.warped_grid.shape
        images = []

        for y in range(ys-1):
            for x in range(xs-1):
                #warped_gp is a set of orinial gridpoints orderd tl, tr, br, bl
                warped_cell = np.zeros((4, 2), dtype = "float32")
                warped_cell = self.warped_grid[y:y+2,x:x+2,:].reshape((4,2))
                warped_cell = self.__order_winding__(warped_cell)
                out_cell =    result_cell.copy()
                cell_img = self.__four_point_transform__(self.image, warped_cell, out_cell)

#                print('working on cell', y, x)
                images.append(cell_img)

        return images

    def __order_winding__(self, points):
        # return the points in the order tl, tr, br, bl
        s = points.sum(axis = 1)
        diff = np.diff(points, axis = 1)
        winding = np.zeros((4,2), dtype="float32")
        winding[0] = points[np.argmin(s)]
        winding[2] = points[np.argmax(s)]
        winding[1] = points[np.argmin(diff)]
        winding[3] = points[np.argmax(diff)]
        return winding

    def __four_point_transform__(self, image, original, result):
        # apply transform to a given warped cell and retun the cell image
        # assume cells are allways organised tl, tr, br, bl
        #  tr  tl
        #  br  bl
        width =  result[1][0] - result[0][0]
        height = result[3][1] - result[0][1]

        M =      cv.getPerspectiveTransform(original, result)
        return   cv.warpPerspective(image, M, (width, height))

    def add_cell_number(self, id, value):
        cell = self.cell_images[id]
        font = cv.FONT_HERSHEY_SIMPLEX
        hight = 2.5
        color = (0,255,0) # Green
        thickness = 9

        textsize = cv.getTextSize(value, font, hight, thickness)
        center = ((cell.shape[0] - textsize[0][0])// 2,
                  (cell.shape[1] + textsize[0][1]) // 2)

        cv.putText(cell, value, center, font, hight, color, thickness)

    def show_image(self, images = None, scaling = 0.3):
        if images == None:
            images = self.cell_images
        y_size,x_size,c  = images[0].shape

        new_image = np.zeros((y_size*9,x_size*9,c), dtype = "uint8" )
        for i, img in enumerate(images):
            y, x = divmod(i,9)
            new_image[y*y_size:(y+1)*y_size, x*x_size:(x+1)*x_size,:] = img

        new_image = cv.resize(new_image,
                              (int(new_image.shape[1]*scaling),
                               int(new_image.shape[0]*scaling))
                              )
        cv.imshow('image', new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def cells_to_numbers(self, images = None):
        '''
        take a processed sudoku image and perform OCR on it to find the cells
        that contain a number. output a string that can be read using sudoku_image
        lib.

        '''
        if images == None:
            images = self.cell_images

        #load and create the model
        sampleData = 'generalsamples.data'
        responseData = 'generalresponses.data'

        sudoku_numbers = []

        samples = np.loadtxt(sampleData,np.float32)
        responses = np.loadtxt(responseData,np.float32)
        responses = responses.reshape((responses.size,1))

        model = cv.ml.KNearest_create()
        model.train(samples, cv.ml.ROW_SAMPLE, responses)

        for i, img in enumerate(images):
            string = '.'

            # pre-processing
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(grey,(5,5),0)
            thresh = cv.adaptiveThreshold(blur,255,1,1,11,2)

            # find the contours
            contours, _ = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                [x,y,w,h] = cv.boundingRect(cnt)
                # potential optimization: turn these criteria into variables...
                # n.b. area of smallest digit, 1, is aprox 700 pixels

                if cv.contourArea(cnt)>500 and w < 100:
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                    string = str(int((results[0][0])))

            sudoku_numbers.append(string)
        return ''.join(sudoku_numbers)

def scale_img(image, pct):
    '''return an image resized by x percent'''
    return cv.resize(image, (int(image.shape[1]*pct), int(image.shape[0]*pct)))

def test_image(image):
    sudoku = sudoku_image(image)
    print(sudoku.cells_to_numbers())
    for i, cell in enumerate(sudoku.cell_images):
        sudoku.add_cell_number(i, str(i))

    sudoku.show_image()

def loop_images():
    import os
    path = '/home/marc/projects/sudoku_solver/sudoku_images/'
    files = [f for f in os.listdir(path) if f.endswith('.jpg') ]
    files =sorted(files)
    for f in files:
        file_str = path+f
        print('attempting to open', file_str)
        test_image(file_str)


if __name__ == '__main__':
    loop_images()
