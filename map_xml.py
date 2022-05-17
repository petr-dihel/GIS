from cmath import pi
import math
import re
from tkinter import W
import xml.etree.ElementTree as ET
import cv2
from cv2 import imshow
from cv2 import cvtColor
from sympy import im
import numpy as np
from collections import defaultdict
import heapq as heap
from queue import PriorityQueue

class ClassNode:

    def __init__(self, id, point, highwayId):
        self.id = id
        self.point = point
        # more nodes
        self.neighbours = {}
        self.highwayId = highwayId
        self.uniqueId = highwayId + "_" + id


class ClassHighway:

    def __init__(self, id, h_node, type, nodes = []):
        self.id = id
        self.h_node = h_node
        self.nodes = nodes
        self.type = type
    

def getClassNodeFromNode(node, highwayId):
    n_node = ClassNode(
        node.get('id'),
        getPointFromNode(node),
        highwayId
    )
    return n_node


def getClassHighwayFromNode(node, type):
    id = node.get('id')
    n_highway = ClassHighway(
        id,
        node,
        type
    )
    return n_highway


def getPointFromNode(node):
    lat = float(node.get('lat'))
    lon = float(node.get('lon'))
    return [lon, lat]


def getTransportStopPoints(tree):
    print("Loading public_transport")
    transports_nodes = tree.findall(".//node/tag/[@k='public_transport']/..")
    points = []
    for node in transports_nodes:
        points.append(getPointFromNode(node))
    
    print("Done")
    return points


def loadHighwayNeighbours(dic_highways):
    print("Loading crossins highways len {}".format(len(dic_highways)))
    #un_searched = dic_highways.copy()

    for h_id, highway in dic_highways.items():
        for node in highway.nodes.values():
            for un_h in dic_highways.values():
                if node.id in un_h.nodes:
                    node.neighbours[un_h.nodes[node.id].uniqueId] = un_h.nodes[node.id]


def getHighway(tree, type, dic_highways):
    print("Loading highway type: {0}".format(type))
    nodes = tree.findall(".//way/tag/[@k='highway'][@v='" + type + "']/..")
    
    highways = []
    for highway in nodes:
        c_highway = getClassHighwayFromNode(highway, type)        
        dic_highways[c_highway.id] = c_highway

        nds = highway.findall('.//nd')
        dict_of_nodes = {}
        last_id = False
        for nd in nds:
            id = nd.get('ref')    
            node = tree.find(".//node[@id='{0}']".format(id))
            if id in dict_of_nodes: 
                print("missing id")
            else:    
                dict_of_nodes[id] = getClassNodeFromNode(node, c_highway.id)
            if last_id != False:
                dict_of_nodes[id].neighbours[dict_of_nodes[last_id].uniqueId] = dict_of_nodes[last_id]
                dict_of_nodes[last_id].neighbours[dict_of_nodes[id].uniqueId] = dict_of_nodes[id]
                #print("{} - {} - {}".format(len(dict_of_nodes), dict_of_nodes[id].id, len(dict_of_nodes[id].neighbours)))
                #print("LAST {} - {} - {}".format(len(dict_of_nodes), dict_of_nodes[last_id].id, len(dict_of_nodes[last_id].neighbours)))
            last_id = id 
        c_highway.nodes = dict_of_nodes

    print("Done")
    return dic_highways


def getRelPoint(min_point, max_point, point, img_shape):
    margin = 50
    r_x = point[0] - min_point[0]
    r_y = point[1] - min_point[1]

    delta_x = max_point[0] - min_point[0] 
    delta_y = max_point[1] - min_point[1]

    r_x = (r_x / delta_x) * (img_shape[0] - margin*2) + margin 
    r_y = (r_y / delta_y) * (img_shape[1] -  margin*2) + margin
    
    return (int(r_x), int(r_y))


def getMaxMin(lists_of_points, dict_of_highways):
    points = []

    for highway in dict_of_highways.values():
        [points.append(node.point) for node in highway.nodes.values()]

    [points.extend(list) for list in lists_of_points]

    min = points[0].copy()
    max = points[0].copy()

    for point in points:
        if point[0] < min[0]:
            min[0] = point[0]

        if point[1] < min[1]:
            min[1] = point[1]

        if point[0] > max[0]:
            max[0] = point[0]
    
        if point[1] > max[1]:
            max[1] = point[1]
    
    return (max, min)


def drawHighway(img, highway, color, thickness, min, max):
    last_point = False
    for node in highway.nodes.values():
        r_point = getRelPoint(min, max, node.point, img.shape)
        
        if last_point != False:
            img = cv2.line(img, last_point, r_point, color, thickness)
        last_point = r_point


def draw(transports_points, dic_highways, max, min):
    img = np.zeros([600, 600, 3], dtype=np.uint8)
    img.fill(255)
    img_shapes = img.shape

    for highway in dic_highways.values():
        color = (255, 0, 0)
        thickness = 1
        if highway.type == 'primary':
            #color = (0, 0, 0)
            thickness = 3

        if highway.type == 'secondary':
            #color = (255, 255, 0)
            thickness = 2
        drawHighway(img, highway, color, thickness, min, max)

    for point in transports_points:
        r_point = getRelPoint(min, max, point, img_shapes)
        img = cv2.circle(img, r_point, 3, (0, 0, 255), 1)
        
    return img


def getPathInternal(path, parentsMap, node):
    path.append(node.point)
    if node.uniqueId in parentsMap:
        return getPathInternal(path, parentsMap, parentsMap[node.uniqueId])
    return path
     

def getPointById(dic_highways, id):
    for highway in dic_highways.values():
        if id in highway.nodes:
            return highway.nodes[id]
    return False

def myNav2(fromPoint, toPoint):
    visited = set()
    parentsMap = {}
    nodeCosts = defaultdict(lambda: float('inf'))
    nodeCosts[fromPoint.uniqueId] = 0

    que = PriorityQueue()
    que.put((0, fromPoint.uniqueId, fromPoint))

    node = fromPoint
    while que.qsize():

        cost, nodeiniqueId, node = que.get()
        if node.uniqueId in visited:
            continue
        visited.add(node.uniqueId)
        
        for uniqueId, n_node in node.neighbours.items():
            x_dist = node.point[0] - n_node.point[0]
            y_dist = node.point[1] - n_node.point[1]
            distance = x_dist*x_dist + y_dist * y_dist
            newCost = nodeCosts[node.uniqueId] + distance
            if nodeCosts[n_node.uniqueId] > newCost:
                parentsMap[n_node.uniqueId] = node
                nodeCosts[n_node.uniqueId] = newCost
                que.put((newCost, n_node.uniqueId, n_node))
        
    path = []
    return getPathInternal(path, parentsMap, toPoint)

def myNav(fromPoint, toPoint):
    visited = set()
    parentsMap = {}
    nodeCosts = defaultdict(lambda: float('inf'))
    nodeCosts[fromPoint.uniqueId] = 0

    que = []
    que.append(fromPoint)

    node = fromPoint
    while len(que):

        node = que.pop()
        if node.uniqueId in visited:
            continue
        visited.add(node.uniqueId)
        
        for uniqueId, n_node in node.neighbours.items():
            x_dist = n_node.point[0] - node.point[0]
            y_dist = n_node.point[1] - node.point[1]
            distance = x_dist*x_dist + y_dist * y_dist
            newCost = nodeCosts[node.uniqueId] + distance
            if nodeCosts[n_node.uniqueId] > newCost:
                parentsMap[n_node.uniqueId] = node
                nodeCosts[n_node.uniqueId] = newCost
            
            que.append(n_node)
        
    path = []
    return getPathInternal(path, parentsMap, toPoint)

def drawnei(fromPoint, min, max):
    img2 = np.zeros([600, 600, 3], dtype=np.uint8)
    img2.fill(255)
    visited = set()

    que = []
    que.append(fromPoint)
    nodes = []
    last_node = False

    while len(que):
        node = que.pop()
        if node.uniqueId in visited:
            continue
        visited.add(node.uniqueId)
        r_point = getRelPoint(min, max, node.point, img2.shape)

        if last_node != False:
            if last_node.uniqueId in node.neighbours:
                r_point2 = getRelPoint(min, max, last_node.point, img2.shape)
                img2 = cv2.line(img2, r_point2, r_point, (0, 255, 0), 1)
        last_node = node
        for n_node in node.neighbours.values():
            que.append(n_node)

    print(len(fromPoint.neighbours))
    return img2

def main():

    tree = ET.parse('poruba.osm')

    #cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

    transports_points = getTransportStopPoints(tree) 
    dic_highways = {}
    getHighway(tree, "secondary", dic_highways)
    getHighway(tree, "residential", dic_highways)
    getHighway(tree, "primary", dic_highways)
    max, min = getMaxMin([transports_points], dic_highways)
    loadHighwayNeighbours(dic_highways)

    print("Max {0}, Min {1}".format(max, min))

    startPoint = getPointById(dic_highways, "2258852362")
    endPoint = getPointById(dic_highways, "368348579")
    
    img = draw(transports_points, dic_highways, max, min)
    print("Done drawing")
    img_to_save = cv2.flip(img, 0)
    cv2.imwrite("map_draw.png", img_to_save)

    path = myNav2(startPoint, endPoint)
    last_point = False
    for point in path:
        r_point = getRelPoint(min, max, point, img.shape)
        if last_point != False:
            img = cv2.line(img, last_point, r_point, (0, 255, 0), 2)
        last_point = r_point

    for node in [startPoint, endPoint]:
        r_point = getRelPoint(min, max, node.point, img.shape)
        img = cv2.circle(img, r_point, 3, (0, 0, 255), 2)
    img = cv2.flip(img, 0)

    cv2.imwrite("map_draw_nav.png", img)

    img = drawnei(startPoint, min, max)
    img = cv2.flip(img, 0)
    cv2.imwrite("map_draw_debug.png", img)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()