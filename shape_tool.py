import cv2
import numpy as np

def is_smallpolygon_covered_by_largeone(large_polygon, small_polygon):

    # #fill the two polygons
    # cv2.fillPoly()

    large_polygon_points = np.array(large_polygon, np.int).reshape((-1, 2))
    small_polygon_points = np.array(small_polygon, np.int).reshape((-1, 2))
    for point in small_polygon_points:
        if cv2.pointPolygonTest(contour= large_polygon_points,pt= tuple(point), measureDist= False) == -1:
            del large_polygon_points, small_polygon_points
            return False
    del large_polygon_points, small_polygon_points
    return True

def generate_rect_points(shape):
    # organize its vertex points to quadrangle
    points = np.array(shape['points']).reshape((2, 2))
    pt0 = np.min(points[:,0])
    pt1 = np.min(points[:,1])
    pt4 = np.max(points[:,0])
    pt5 = np.max(points[:,1])
    pt2 = pt4
    pt3 = pt1
    pt6 = pt0
    pt7 = pt5
    del  points
    #pts = np.zeros(4, 2)
    return np.array([[pt0, pt1],[pt2, pt3],[pt4, pt5],[pt6, pt7]]).reshape((4,2))