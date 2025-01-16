import cv2
import numpy as np

image = cv2.imread(r"C:\Users\HP\Desktop\Projects 2k24\Deep Learning\Open CV\Datasets\Images\Obj_size_est_Samples\20241213_152511.jpg")
scale = 2
width = 212 * scale
height = 296 * scale

def preprocessing_image(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(gray_scale, (5, 5), 1)
    canny_edge = cv2.Canny(imageBlur, 100, 100)
    kernel = np.ones((5, 5), np.uint8)
    image_dilation = cv2.dilate(canny_edge, kernel, iterations=3)
    image_erosion = cv2.erode(image_dilation, kernel, iterations=2)
    return image_erosion

def find_draw_contours(preprocessed_image):
    contours, hirearchy = cv2.findContours(preprocessed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Length of the Contours", len(contours))
    for cnt in contours:
        cv2.drawContours(original_image_copy, cnt, -1, (0, 255, 0), 3)
        area = cv2.contourArea(cnt)
        print(f"Area for each of the contour in the image: {area}")
    return original_image_copy

def get_contours(preprocessed_image, draw_image, minArea=1000, filter=4, draw=False):
    contours, hirearchy = cv2.findContours(preprocessed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    finalContours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, cnt])
            else:
                finalContours.append([len(approx), area, approx, bbox, cnt])
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for cont in finalContours:
            cv2.drawContours(draw_image, cont[4], -1, (0, 0, 255), 2)
    return draw_image, finalContours

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def warp_image(image, points, width, height):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imageWrap = cv2.warpPerspective(image, matrix, (width, height))
    imageCropped = imageWrap[20:imageWrap.shape[0] - 20, 20:imageWrap.shape[1] - 20]
    imageCropped = cv2.resize(imageCropped, (width, height))
    return imageCropped

def calculate_distance(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5

image = cv2.resize(image, (0, 0), None, 0.2, 0.2)
cv2.imwrite("resized_image.jpg", image)

original_image_copy = image.copy()
original_image_copy_two = image.copy()

preprocessed_image = preprocessing_image(image)
draw_contours = find_draw_contours(preprocessed_image)

contour_area, finalContours = get_contours(preprocessed_image, original_image_copy_two, minArea=1000, draw=True)

if len(finalContours) != 0:
    biggest = finalContours[0][2]
    warp_image_one = warp_image(image, biggest, width, height)
    warp_image_one_copy = warp_image_one.copy()
    preprocessed_image_warp_two = preprocessing_image(warp_image_one_copy)
    contours_area2, finalContours2 = get_contours(preprocessed_image_warp_two, warp_image_one_copy, minArea=1000, draw=True)
    if len(finalContours2) != 0:
        for cnt2 in finalContours2:
            cv2.polylines(contours_area2, [cnt2[2]], True, (0, 255, 0), 2)
            reorder_points_2 = reorder(cnt2[2])
            NW = round((calculate_distance(reorder_points_2[0][0] // scale, reorder_points_2[1][0] // scale) / 10), 1)
            NH = round((calculate_distance(reorder_points_2[0][0] // scale, reorder_points_2[2][0] // scale) / 10), 1)
            cv2.arrowedLine(contours_area2, (reorder_points_2[0][0][0], reorder_points_2[0][0][1]), 
                            (reorder_points_2[1][0][0], reorder_points_2[1][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(contours_area2, (reorder_points_2[0][0][0], reorder_points_2[0][0][1]), 
                            (reorder_points_2[2][0][0], reorder_points_2[2][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = cnt2[3]
            cv2.putText(contours_area2, '{}cm'.format(NW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)
            cv2.putText(contours_area2, '{}cm'.format(NH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)

cv2.imshow("Input Image", image)
cv2.imshow("Preprocessed Image", preprocessed_image)
cv2.imshow("Find and Draw Contours", draw_contours)
cv2.imshow("Get Contours", contour_area)
cv2.imshow("Warped Image One", warp_image_one)
cv2.imshow("Preprocessed Image Warp Two", preprocessed_image_warp_two)
cv2.imshow("Two Finding the contours and Area of the contour on the Warped Image", contours_area2)
cv2.waitKey(0)
