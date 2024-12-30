import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import cv2
import numpy as np

# compute intersection of two bounding boxes
def intersection(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1 + w1, x2 + w2) - x
    h = min(y1 + h1, y2 + h2) - y

    # return boolean
    return w > 0 and h > 0

# visualize bounding box
def visualize_box(img, bbox, label, color):
    bbox = [int(i) for i in bbox]
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    # draw rectangle for background behind the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_padding = 5

    # Calculate text size (width, height) and baseline
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Calculate rectangle coordinates such that the rectangle is inside the box, top left
    rect_start = (bbox[0], bbox[1] - text_height - 2 * text_padding)
    rect_end = (bbox[0] + text_width + 2 * text_padding, bbox[1])
    # if negative, move the rectangle to the left
    if rect_start[0] < 0:
        rect_start = (0, rect_start[1])
        rect_end = (text_width + 2 * text_padding, rect_end[1])
    if rect_end[0] > img.shape[1]:
        rect_start = (img.shape[1] - text_width - 2 * text_padding, rect_start[1])
        rect_end = (img.shape[1], rect_end[1])
    if rect_start[1] < 0:
        rect_start = (rect_start[0], 0)
        rect_end = (rect_end[0], text_height + 2 * text_padding)
    if rect_end[1] > img.shape[0]:
        rect_start = (rect_start[0], img.shape[0] - text_height - 2 * text_padding)
        rect_end = (rect_end[0], img.shape[0])

    # Draw background rectangle
    cv2.rectangle(img, rect_start, rect_end, color, cv2.FILLED)

    # Draw text
    cv2.putText(img, label, (rect_start[0] + text_padding, rect_end[1] - text_padding), font, font_scale, (255, 255, 255), font_thickness)

    return img

def visualize_graph(relations, objects, img, color='blue'):
    # img from PIL format to cv2 format
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # {"relationship_id": 1, "subject_id": 10000004, "predicate": "overlaps", "object_id": 10000003, "confidence": 1.0}
    G = nx.MultiDiGraph()
    objects = {obj['object_id']: obj for obj in objects}
    objects_idx_to_id = {obj['object_id']:i  for i, obj in enumerate(objects.values())}
    for i, rel in enumerate(relations):
        pred = rel['predicate']
        sub_id = objects_idx_to_id[rel['subject_id']]
        obj_id = objects_idx_to_id[rel['object_id']]
        sub = str(sub_id) + "_" + objects[rel['subject_id']]['names']
        obj = str(obj_id) + "_" + objects[rel['object_id']]['names']
        G.add_edge(str(sub), str(obj), label=pred, color=color)

        sub_bbox = [objects[rel['subject_id']]['x'], objects[rel['subject_id']]['y'], objects[rel['subject_id']]['w'], objects[rel['subject_id']]['h']]
        obj_bbox = [objects[rel['object_id']]['x'], objects[rel['object_id']]['y'], objects[rel['object_id']]['w'], objects[rel['object_id']]['h']]

        # draw bounding boxes
        img = visualize_box(img, sub_bbox, sub, (0, 0, 255))
        img = visualize_box(img, obj_bbox, obj, (0, 0, 255))

    # draw networkx graph with graphviz, display edge labels
    G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    G.graph['graph'] = {'scale': '2'}
    G.graph['node'] = {'shape': 'rectangle'}
    # all graph color to blue
    G.graph['edge']['color'] = color
    G.graph['node']['color'] = color

    img_graph = to_agraph(G)
    img_graph.graph_attr.update(dpi=str(300))
    # Layout the graph
    img_graph.layout('dot')

    # Draw the graph directly to a byte array
    png_byte_array = img_graph.draw(format='png', prog='dot')

    # Convert the byte array to an OpenCV image without redundant conversion
    img_cv2 = cv2.imdecode(np.frombuffer(png_byte_array, np.uint8), cv2.IMREAD_COLOR)

    # Resize the graph image if necessary
    scale_factor = 2  # Adjust this factor as needed
    img_cv2 = cv2.resize(img_cv2, (img_cv2.shape[1] * scale_factor, img_cv2.shape[0] * scale_factor))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_cv2, img