import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import cv2
import numpy as np
import colorsys

import torch
from PIL import Image

import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
from matplotlib.backends.backend_agg import FigureCanvasAgg

import random
class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

class SAMProcessor:
    def __init__(self, device=None):
        from transformers import SamModel, SamProcessor
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.output = None  # Placeholder for output image


        # self._default_font_size = max(
        #     np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        # )
        self._default_font_size = 18

    def apply_sam(self, image, input_box=None):
        inputs = self.sam_processor(image, input_boxes=input_box, return_tensors="pt").to(self.device)
        self.output = VisImage(image, scale=1.0)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        scores = outputs.iou_scores[0, 0]

        mask_selection_index = scores.argmax()
        mask_np = masks[mask_selection_index].numpy()
        return mask_np

    def generate_masks(self, image, boxes, colors=[(0.0, 0.0, 1.0),(1.0, 0.0, 0.0)], annotate=True, cropping=False):
        # Blue is subject and red is object of the relation
        # img = np.asarray(image).astype(float) / 255.0
        for i, b in enumerate(boxes):
            mask = self.apply_sam(image, input_box=[[[b[0], b[1], b[0] + b[2], b[1] + b[3]]]])
            self.output = self.draw_binary_mask_with_number(mask, color=colors[i], text=str(i+1))

            # if annotate:
            #     # add label to the mask in the center of the bounding box
            #     label = str(i+1)
            #     x, y, w, h = b
            #     center_x = int(x + w / 2)
            #     center_y = int(y + h / 2)
            #     # create background rectangle for the text
            #     text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            #     rect_x1 = center_x - text_size[0] // 2 - 5
            #     rect_y1 = center_y - text_size[1] // 2 - 5
            #     rect_x2 = center_x + text_size[0] // 2 + 5
            #     rect_y2 = center_y + text_size[1] // 2 + 5
            #     # draw rectangle
            #     cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), colors[i], -1)
            #     # draw text
            #     cv2.putText(img, label, (center_x - text_size[0] // 2, center_y + text_size[1] // 2), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        if cropping:
            assert len(boxes) == 2, "Cropping requires exactly two boxes (subject and object)."
            # Crop the image to the union of the two boxes
            x1, y1, w1, h1 = boxes[0]
            x2, y2, w2, h2 = boxes[1]

            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
            # if possible, add a few pixels around the crop to include context
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1], w + padding)
            h = min(img.shape[0], h + padding)

            img = img[y:y+h, x:x+w]

        # convert back to PIL image
        img = Image.fromarray((self.output.img * 255.0).astype(np.uint8))
        return img
    
    def add_contour(self, img, mask, color=(0, 0, 255)):
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)  # Normalize color to [0, 1] range for float images
        cv2.drawContours(img, contours, -1, color, thickness=2)

        return img
    
    def draw_binary_mask_with_number(
        self, binary_mask, color=None, *, edge_color=None, text=None, label_mode='1', alpha=0.1, anno_mode=['Mask'], area_threshold=10
    ):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component smaller than this area will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        if color is None:
            randint = random.randint(0, len(self.color_proposals)-1)
            color = self.color_proposals[randint]
        color = mplc.to_rgb(color)

        has_valid_segment = True
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])
        bbox = mask.bbox()

        if 'Mask' in anno_mode:
            if not mask.has_holes:
                # draw polygons for regular masks
                for segment in mask.polygons:
                    area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                    if area < (area_threshold or 0):
                        continue
                    has_valid_segment = True
                    segment = segment.reshape(-1, 2)
                    self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
            else:
                # TODO: Use Path/PathPatch to draw vector graphics:
                # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
                rgba = np.zeros(shape2d + (4,), dtype="float32")
                rgba[:, :, :3] = color
                rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
                has_valid_segment = True
                self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))

        if text is not None and has_valid_segment:
            # lighter_color = tuple([x*0.2 for x in color])
            lighter_color = [1,1,1] # self._change_color_brightness(color, brightness_factor=0.7)
            self._draw_number_in_mask(binary_mask, text, lighter_color, label_mode)
        return self.output
    
    def _draw_number_in_mask(self, binary_mask, text, color, label_mode='1'):
        """
        Find proper places to draw text given a binary mask.
        """

        def number_to_string(n):
            chars = []
            while n:
                n, remainder = divmod(n-1, 26)
                chars.append(chr(97 + remainder))
            return ''.join(reversed(chars))

        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]

        if label_mode == 'a':
            text = number_to_string(int(text))
        else:
            text = text

        self.draw_text(text, (coords_x[len(coords_x)//2] + 2, coords_y[len(coords_y)//2] - 6), color=color)

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.15)
        color[np.argmax(color)] = max(0.8, np.max(color))

        def contrasting_color(rgb):
            """Returns 'white' or 'black' depending on which color contrasts more with the given RGB value."""
            
            # Decompose the RGB tuple
            R, G, B = rgb

            # Calculate the Y value
            Y = 0.299 * R + 0.587 * G + 0.114 * B

            # If Y value is greater than 128, it's closer to white so return black. Otherwise, return white.
            return 'black' if Y > 128 else 'white'

        bbox_background = contrasting_color(color*255)

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": bbox_background, "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output
    
    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


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