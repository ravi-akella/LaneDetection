import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height), (1100,height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),thickness=10)
    return line_image

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        print(slope,intercept)
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg = np.average(left_fit,axis=0)
    print('left_fit_avg is %s'%left_fit_avg)
    right_fit_avg = np.average(right_fit,axis=0)
    print('line parameters are %s,%s'%(left_fit_avg,right_fit_avg))
    left_line = make_coordinates(image,left_fit_avg)
    right_line = make_coordinates(image,right_fit_avg)
    return np.array([left_line,right_line])

def run_algo_on_image(frame):
    # load image
    # image = cv2.imread("test_image.jpg")
    # cv2.imshow("Original", image)

    # convert to gray
    lane_image = np.copy(frame)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("Gray", gray)

    # gaussian blur: optional with canny (as inbuilt in it)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # canny edge detect
    canny = cv2.Canny(blur, 50, 150)
    # cv2.imshow("Canny",canny)

    # roi
    cropped_image = region_of_interest(canny)
    # cv2.imshow("result",cropped_image)

    # Hough Lines detection
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    print(lines)
    average_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, average_lines)
    # cv2.imshow("Line_image",line_image)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.namedWindow("Lane Detection Output",cv2.WINDOW_NORMAL)
    cv2.imshow("Lane Detection Output", combo_image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":

    #create capture handle
    cap = cv2.VideoCapture('test2.mp4')

    #until video stream is ON
    while(cap.isOpened()):
        _,frame = cap.read()
        #run algo for each frame
        run_algo_on_image(frame)
