"""
This is our place for post processing utils.
Meaning utils, that are applied after the fixations are classified.

There is a master function called `run_post_processing`, which filters and merges the data to our needs
and will return a dataframe, the shape of which we have agreed on.
"""
import pandas as pd
from shapely.geometry import Point
from svg.path import parse_path
from shapely.geometry import Polygon
from xml.dom import minidom
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

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


def polygons_from_doc(doc, density=0.05, scale=1, offset=(0, 1020)):
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


def dob_to_age(ID_col: pd.Series, DoB_col: pd.Series) -> pd.Series:
    # Ensure that ID_col is treated as a string
    ID_col = ID_col.astype(str)

    # Extract year and month from the ID
    # Extract the year from the first 4 characters
    year = ID_col.str.slice(0, 4).astype(int)
    # Extract the month from the next 2 characters
    month = ID_col.str.slice(4, 6).astype(int)

    # Convert DoB to integer
    DoB_year = DoB_col.astype(int)

    # Use np.where to calculate age based on the condition
    age = np.where(month >= 7, 1 + year - DoB_year, year - DoB_year)

    return pd.Series(age)


def classify_roi(row, regions):
    x = row["avg_x"]
    y = row["avg_y"]
    point = Point(x, y)
    for region in regions:
        if regions[region].contains(point):
            return region
    # Avoid None values, to not run into shape errors when fitting regressions.
    return "None"

def get_roi_area(row, regions):
    # this function requires classify_roi to be called beforehand
    region = row["ROI"]
    if region in regions:
        return regions[region].area

def categorize_hue(hue):
    """
    Categorizes a hue value into its corresponding color.

    Args:
        hue (float): A hue value between 0 and 360.

    Returns:
        str: The name of the color category.
    """
    if not (0 <= hue <= 360):
        return "Invalid hue value. It must be between 0 and 360."

    # Use modulo 360 to handle hue wraparound, ensuring hue stays in [0, 360)
    hue %= 360

    # Categorize hue into colors
    if hue <= 30:
        return "Red"
    elif hue <= 60:
        return "Orange"
    elif hue <= 90:
        return "Yellow"
    elif hue <= 150:
        return "Green"
    elif hue <= 180:
        return "Cyan/Teal"
    elif hue <= 240:
        return "Blue"
    elif hue <= 270:
        return "Purple"
    elif hue <= 300:
        return "Magenta/Pink"
    else:
        return "Red"

def get_pix_hsb(row, image, type="color", width = 1920, height = 1080):
    """
    Extracts the spectral color value (so either red, green or blue) from a given fixation.

    Parameters:
        row: row form the compiled fixations dataframe
        image: PIL.Image instance

    Returns None if the avg_x and avg_y values are out of bounds
    """
    x = int(row["avg_x"])
    y = int(row["avg_y"])
    ind = 0
    if type == "s":
        ind = 1
    elif type == "b":
        ind = 2
    if (x >= 0 and x <= width) and (y >= 0 and y<= height): 
        value = image[x-1, y-1][ind]
        if type == "color":
            value = categorize_hue(value)
        return value

def get_pix_spectral(row, image, col="r"):
    x = int(row["avg_x"])
    y = int(row["avg_y"])
    ind = 0
    if col == "g":
        ind = 1
    elif col == "b":
        ind = 2
    if (x >= 0 and x <= image.size[0]) and (y >= 0 and y<= image.size[1]): 
        return image.load()[x-1, y-1][ind]

########################################################################################
##                                                                                    ##
########################################################################################

def run_post_processing() -> pd.DataFrame:
    print("\tfetching processed data..")
    # Read in .csv files
    obs = pd.read_csv("./results/processed_fixations.csv")
    ppl = pd.read_csv("./results/participant_info.csv")
    # Merge them into one DataFrame by performing an Inner Join on ID
    df = pd.merge(obs, ppl, left_on='ID', right_on='ID', how="inner")
    # Create Age column
    df['age'] = dob_to_age(df['ID'], df['DoB'])
    # Create Gender Binary Variables while avoiding dummy variable trap
    df['female'] = np.where(df['Gender'] == 'FEMALE', 1, 0)
    
    print("\tfiltering valid data...")
    # Filter out participants with only "valid demographics"
    result = df.loc[(df['age'] >= 5) & (df['age'] <= 59) & (df['DoB'] != 2000) & (
        df['Gender'] != "OTHER") & (df['duration'] > 0.06) & (df['duration'] < 2)
        & (df["Valid"] == True), :].copy()
    
    print("\trunning age binning clustering...")
    # Utilize K-Means Clustering to bin ages
    X = result['age'].values.reshape(-1, 1)
    kmeans_5 = KMeans(n_clusters=5, random_state=0)
    labels_5 = kmeans_5.fit_predict(X)
    # Bin Ages accordingly
    # Here we will choose to make age buckets based on 5 clusters
    result['Age Clusters'] = kmeans_5.fit_predict(X)

    # Step 2: Determine age ranges within each cluster
    age_ranges = result.groupby('Age Clusters')['age'].agg(
        ['min', 'max']).sort_values(by='min')
    age_ranges['Age_Range'] = age_ranges.apply(
        lambda row: f"{row['min']}-{row['max']}", axis=1)
    # create bin labels
    bins = [5, 17, 26, 35, 45, 59]
    labels = ['5-17', '18-26', '27-35', '36-45', '46-59']
    # assign bin labels to new column
    result['Age_Group_Cluster'] = pd.cut(
        result['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    # generate dummy variables
    age_dummies = pd.get_dummies(
        result['Age_Group_Cluster'], prefix='Age', drop_first=True).astype(int)

    # attach dummies to data frame
    result = pd.concat([result, age_dummies], axis=1)
    print("\tgenerating log duration...")
    
    # log-transform the duration variable
    result['log_duration'] = np.log(result['duration'].copy())
    
    print("\trunning ROI classification...")
    # attach ROI classification
    regions = get_polygons_from_svg("./data/regions.svg")

    result["ROI"] = result.apply(lambda row: classify_roi(row, regions), axis=1)

    print("\tfetching ROI areas")
    result["ROI_area"] = result.apply(lambda row: get_roi_area(row, regions), axis=1)
    
    print("\tgenerating ROI dummies...")
    roi_dummies = pd.get_dummies(
        result['ROI'], drop_first = True
    ).astype(int)
    result = pd.concat([result,roi_dummies], axis = 1)

    print("\tgenerating rgb values for fixations...")
    stim = Image.open("./data/stimulus.jpg")
    result["red"] = result.apply(lambda row: get_pix_spectral(row, stim, col="r"), axis=1)
    result["green"] = result.apply(lambda row: get_pix_spectral(row, stim, col="g"), axis=1)
    result["blue"] = result.apply(lambda row: get_pix_spectral(row, stim, col="b"), axis=1)

    print("\tcreating differences from average rgb value...")
    avg_rgb = np.array(stim).mean(axis=(0, 1))[:3]
    result["red_diff"] = result.red - avg_rgb[0]
    result["green_diff"] = result.green - avg_rgb[1]
    result["blue_diff"] = result.blue - avg_rgb[2]

    result["rgb_diff_euclidean"] = np.sqrt(result.red_diff**2 + result.green_diff**2 + result.blue_diff**2)
    
    print("\tcreating color, saturation and brightness values...")
    stim = stim.convert("HSV").load()
    result["Color"] = result.apply(lambda row: get_pix_hsb(row, stim), axis=1)
    result["Saturation"] = result.apply(lambda row: get_pix_hsb(row, stim, type="s"), axis=1)
    result["Brightness"] = result.apply(lambda row: get_pix_hsb(row, stim, type="b"), axis=1)

    result["Saturation_normed"] = (result.Saturation - result.Saturation.mean())/result.Saturation.std()
    result["Brightness_normed"] = (result.Brightness - result.Brightness.mean())/result.Brightness.std()
    
    print("\tExcluding out of bounds fixations...")
    result = result.loc[~result.Saturation.isna(), :]
    
    return result
