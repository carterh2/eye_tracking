"""
This is our place for post processing utils.
Meaning utils, that are applied after the fixations are classified.

There is a master function called `run_post_processing`, which filters and merges the data to our needs
and will return a dataframe, the shape of which we have agreed on.
"""
import pandas as pd
from svg.path import parse_path
from shapely.geometry import Polygon
from xml.dom import minidom

########################################################################################
## Insert your util functions here                                                    ##
########################################################################################
def get_point_at(path, distance, scale, offset):
    pos = path.point(distance)
    pos = pos.real + (offset.imag - pos.imag)*1j
    pos *= scale
    return pos.real, pos.imag


def points_from_path(path, density, scale, offset):
    step = int(path.length() * density)
    last_step = step - 1

    if last_step == 0:
        yield get_point_at(path, 0, scale, offset)
        return

    for distance in range(step):
        yield get_point_at(
            path, distance / last_step, scale, offset)


def polygons_from_doc(doc, density=0.05, scale=1, offset=(0,1020)):
    offset = offset[0] + offset[1] * 1j
    polygons = {}
    for element in doc.getElementsByTagName("path"):
        points = []
        id = element.getAttribute("id")
        for path in parse_path(element.getAttribute("d")):
            points.extend(points_from_path(path, density, scale, offset))
        polygons[id] = Polygon(points)
    return polygons

def get_polygons_from_svg(filepath):
    with open(filepath) as f:
        svg_string = f.read()
        doc = minidom.parseString(svg_string)
        return polygons_from_doc(doc)


########################################################################################
##                                                                                    ##
########################################################################################

def run_post_processing(processed_fixations: pd.DataFrame) -> pd.DataFrame: 
    """
    
    """
    return processed_fixations