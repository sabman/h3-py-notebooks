# %% [markdown]
# <h1> Uber H3 API examples on Urban Analytics in the city of Toulouse (France)</h1>
# 
# <br/><br/>
# <font size="3"><b> Author: Camelia Ciolac </b></font><br/>
# ciolac_c@inria-alumni.fr

# %% [markdown]
# <font size=3>
# 
# This notebook presents several spatial analytics using the H3 geospatial index, using open data from the city of Toulouse.  <br/>
# Highlights:
# 
# * Spatial search (K nearest neighbors) and spatial join (point in polygon)
# * Aggregates accompanied by 3D visualizations
# * Tensorflow classifier of spatial point patterns using hexagonal convolutions
# 
# Sections:
#  
# <ol style="list-style-type: upper-roman;">
# <li>Preliminaries  </li>
# <li>Use H3 indexing for spatial operations  </li>
# <li>Use H3 spatial index for aggregated analytics </li>
# <li>Global Spatial Autocorrelation</li>
#     IV.4. Spatial Autocorrelation Prediction with Tensorflow
# <li>3D visualizations in JavaScript with deck.gl</li>
# </ol>     
#     
# <br/>
# Relevant references about H3:   <br/> 
# <a href="https://h3geo.org/docs">https://h3geo.org/docs</a> <br/>
# <a href="https://eng.uber.com/h3/"> https://eng.uber.com/h3/ </a><br/>    
# <a href="https://portal.opengeospatial.org/files/?artifact_id=90338">https://portal.opengeospatial.org/files/?artifact_id=90338</a>
# </font>

# %%
!pip show h3

# %%
from IPython.core.display import display, HTML
from IPython.display import Image, display
from IPython.utils.text import columnize
from IPython import get_ipython
#display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
import h3

print(columnize(dir(h3), displaywidth=100))

# %% [markdown]
# ----------------

# %% [markdown]
# # Setup steps

# %% [markdown]
# Virtual environment was set up as follows:

# %% [markdown]
# ```
# virtualenv -p /usr/bin/python3.6 ./projenv_demo_h3   
# 
# source projenv_demo_h3/bin/activate  
# 
# pip3 install ipython==7.2.0 jupyter==1.0.0  
# 
# jupyter notebook  
# ```

# %% [markdown]
# For rtree (used in geopandas sjoin) the following is required  on Ubuntu:
# 
# ```
# sudo apt-get install libspatialindex-dev
# ```

# %%
import sys
sys.version_info

# %%
%%sh
cat <<EOF > requirements_demo.txt
pandas>=1.5.3
statsmodels>=0.13.5
tensorflow-macos>=2.11.0

h3>=3.7.6
geopandas>=0.12.2
geojson>=3.0.1
Rtree>=1.0.1
pygeos>=0.14
Shapely>=2.0.1
libpysal>=4.7.0
esda>=2.4.3
pointpats>=2.3.0
descartes>=1.1.0

annoy>=1.17.1

folium>=0.14.0
seaborn>=0.12.2
pydeck

more-itertools
tqdm
line-profiler
flake8
pycodestyle
pycodestyle_magic
EOF

# %%
%%sh
# pip3 install -U \
#             --disable-pip-version-check \
#             -r requirements_demo.txt

# %%
%%bash
VAR_SEARCH='h3|pydeck|pandas|tensorflow|shapely|geopandas|esda|pointpats|libpysal|annoy'
pip3 freeze | grep -E $VAR_SEARCH

# %% [markdown]
# To enable pydeck for Jupyter:
#     
# ```
# jupyter nbextension install --sys-prefix --symlink --overwrite --py pydeck
# jupyter nbextension enable --sys-prefix --py pydeck
# ```
# 
# 
# For PlotNeuralNet:
# 
# ```
# sudo apt-get install texlive-latex-extra
# ```

# %%
# !git clone https://github.com/HarisIqbal88/PlotNeuralNet

# %% [markdown]
# -------------------

# %% [markdown]
# ## Data sources for the examples:
# 
# Bus stops:
# https://data.toulouse-metropole.fr/explore/dataset/arrets-de-bus0/information/
# 
# City subzones:
# https://data.toulouse-metropole.fr/explore/dataset/communes/information/
# 
# Residential districts:
# https://data.toulouse-metropole.fr/explore/dataset/recensement-population-2015-grands-quartiers-logement/information/
# 
# 
# 
# <b>Note:</b>  
# We analyze only bus stops data for this example, however, the city of Toulouse has also trams and metro (underground) as part of the urban public transport network.  
# 

# %%
%%sh
mkdir -p datasets_demo

# %%
%%sh
# wget -O datasets_demo/busstops_Toulouse.geojson --content-disposition -q "https://data.toulouse-metropole.fr/explore/dataset/arrets-de-bus0/download/?format=geojson&timezone=Europe/Helsinki"

# %%
%%sh
ls -alh datasets_demo/busstops_*.geojson

# %%
%%sh
# wget -O datasets_demo/subzones_Toulouse.geojson --content-disposition -q \
#     "https://data.toulouse-metropole.fr/explore/dataset/communes/download/?format=geojson&timezone=Europe/Helsinki"

# %%
%%sh
ls -alh datasets_demo/subzones_*.geojson

# %%
%%sh
# wget -O datasets_demo/districts_Toulouse.geojson --content-disposition -q \
#     "https://data.toulouse-metropole.fr/explore/dataset/recensement-population-2015-grands-quartiers-logement/download/?format=geojson&timezone=Europe/Helsinki"

# %%
%%sh
ls -alh datasets_demo/districts_*.geojson

# %% [markdown]
# ---

# %% [markdown]
# ## Imports

# %%
import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

import statistics
import statsmodels as sm
import statsmodels.formula.api as sm_formula
from scipy import stats

import tensorflow as tf
from tensorflow.keras import layers, models

print(tf.__version__)

# %%
import warnings
warnings.filterwarnings('ignore')


# don't use scientific notation
np.set_printoptions(suppress=True) 
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# %%
import h3

import geopandas as gpd

from shapely import geometry, ops
import libpysal as pys
import esda
import pointpats as pp

from geojson.feature import *

# %%
from annoy import AnnoyIndex

import bisect
import itertools
from more_itertools import unique_everseen

import math
import random
import decimal
from collections import Counter

from pprint import pprint
import copy

from tqdm import tqdm

# %%
import pydeck

from folium import Map, Marker, GeoJson
from folium.plugins import MarkerCluster
import branca.colormap as cm
from branca.colormap import linear
import folium

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.gridspec as gridspec

from PIL import Image as pilim

%matplotlib inline

# %%
import sys
sys.path.append('./PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks  import *

# %%
%load_ext line_profiler
%load_ext pycodestyle_magic

# %% [markdown]
# See https://www.flake8rules.com/ for codes

# %%
# %flake8_on --ignore E251,E703,W293,W291 --max_line_length 90
# %flake8_off

# %% [markdown]
# --------------------

# %% [markdown]
# #  I. Preliminaries

# %% [markdown]
# ## I.1. Metadata of H3 cells
# 
# For various H3 index resolutions, display metadata about the corresponding haxagon cells

# %%
max_res = 15
list_hex_edge_km = []
list_hex_edge_m = []
list_hex_perimeter_km = []
list_hex_perimeter_m = []
list_hex_area_sqkm = []
list_hex_area_sqm = []

for i in range(0, max_res + 1):
    ekm = h3.edge_length(resolution=i, unit='km')
    em = h3.edge_length(resolution=i, unit='m')
    list_hex_edge_km.append(round(ekm, 3))
    list_hex_edge_m.append(round(em, 3))
    list_hex_perimeter_km.append(round(6 * ekm, 3))
    list_hex_perimeter_m.append(round(6 * em, 3))

    akm = h3.hex_area(resolution=i, unit='km^2')
    am = h3.hex_area(resolution=i, unit='m^2')
    list_hex_area_sqkm.append(round(akm, 3))
    list_hex_area_sqm.append(round(am, 3))

df_meta = pd.DataFrame({"edge_length_km": list_hex_edge_km,
                        "perimeter_km": list_hex_perimeter_km,
                        "area_sqkm": list_hex_area_sqkm,
                        "edge_length_m": list_hex_edge_m,
                        "perimeter_m": list_hex_perimeter_m,
                        "area_sqm": list_hex_area_sqm
                        })

df_meta[["edge_length_km", "perimeter_km", "area_sqkm", 
         "edge_length_m", "perimeter_m", "area_sqm"]]

# %% [markdown]
# <h3> Index a central point in Toulouse at various resolutions of the H3 index</h3>
# 
# To better make sense of resolutions, we index spatially with H3 a central GPS point of the French city Toulouse: 

# %%
lat_centr_point = 43.600378
lon_centr_point = 1.445478
list_hex_res = []
list_hex_res_geom = []
list_res = range(0, max_res + 1)

for resolution in range(0, max_res + 1):
    # index the point in the H3 hexagon of given index resolution
    h = h3.geo_to_h3(lat = lat_centr_point,
                     lng = lon_centr_point,
                     resolution = resolution
                     )

    list_hex_res.append(h)
    # get the geometry of the hexagon and convert to geojson
    h_geom = {"type": "Polygon",
              "coordinates": [h3.h3_to_geo_boundary(h = h, geo_json = True)]
              }
    list_hex_res_geom.append(h_geom)


df_res_point = pd.DataFrame({"res": list_res,
                             "hex_id": list_hex_res,
                             "geometry": list_hex_res_geom
                             })
df_res_point["hex_id_binary"] = df_res_point["hex_id"].apply(
                                                lambda x: bin(int(x, 16))[2:])

pd.set_option('display.max_colwidth', 63)
df_res_point


# %% [markdown]
# Visualize on map:

# %%
!mkdir -p maps
!mkdir -p images

# %%
map_example = Map(location = [43.600378, 1.445478],
                  zoom_start = 5.5,
                  tiles = "cartodbpositron",
                  attr = '''© <a href="http://www.openstreetmap.org/copyright">
                          OpenStreetMap</a>contributors ©
                          <a href="http://cartodb.com/attributions#basemaps">
                          CartoDB</a>'''
                  )

list_features = []
for i, row in df_res_point.iterrows():
    feature = Feature(geometry = row["geometry"],
                      id = row["hex_id"],
                      properties = {"resolution": int(row["res"])})
    list_features.append(feature)

feat_collection = FeatureCollection(list_features)
geojson_result = json.dumps(feat_collection)


GeoJson(
        geojson_result,
        style_function = lambda feature: {
            'fillColor': None,
            'color': ("green"
                      if feature['properties']['resolution'] % 2 == 0
                      else "red"),
            'weight': 2,
            'fillOpacity': 0.05
        },
        name = "Example"
    ).add_to(map_example)

map_example.save('maps/1_resolutions.html')
map_example


# %%
# fig, ax = plt.subplots(1, 1, figsize = (15, 10))

# im1 = pilim.open('images/1_resolutions.png', 'r')
# ax.imshow(np.asarray(im1))
# ax.set_axis_off();

# %% [markdown]
# Note: the color scheme of hexagons boundaries was coded with green for even resolution (0,2,4,etc) and red of odd resolution(1,3,5,etc)  
# 

# %% [markdown]
# ## I.2. Inspect the parent - children relationship in the H3 hierarchy
# 
# 

# %% [markdown]
# This section is particularly useful for understanding the implications of replacing children with the parent cell (as it is the case of using h3.compact)

# %%
res_parent = 9
h3_cell_parent = h3.geo_to_h3(lat = lat_centr_point,
                              lng = lon_centr_point,
                              resolution = res_parent
                              )
h3_cells_children = list(h3.h3_to_children(h = h3_cell_parent))
assert(len(h3_cells_children) == math.pow(7, 1))
# ------
h3_cells_grandchildren = list(h3.h3_to_children(h = h3_cell_parent, 
                                                res = res_parent + 2))
assert(len(h3_cells_grandchildren) == math.pow(7, 2))
# ------
h3_cells_2xgrandchildren = list(h3.h3_to_children(h = h3_cell_parent, 
                                                  res = res_parent + 3))
assert(len(h3_cells_2xgrandchildren) == math.pow(7, 3))

# ------
h3_cells_3xgrandchildren = list(h3.h3_to_children(h = h3_cell_parent, 
                                                  res = res_parent + 4))
assert(len(h3_cells_3xgrandchildren) == math.pow(7, 4))
# ------

msg_ = """Parent cell: {} has :
          {} direct children, 
          {} grandchildren,
          {} grandgrandchildren, 
          {} grandgrandgrandchildren"""
print(msg_.format(h3_cell_parent, len(h3_cells_children),
                  len(h3_cells_grandchildren), 
                  len(h3_cells_2xgrandchildren),
                  len(h3_cells_3xgrandchildren)))
      

# %%
def plot_parent_and_descendents(h3_cell_parent, h3_cells_children, ax=None):
                                
    list_distances_to_center = []
                                
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    
    boundary_parent_coords = h3.h3_to_geo_boundary(h=h3_cell_parent, geo_json=True)
    boundary_parent = geometry.Polygon(boundary_parent_coords)
    # print(boundary_parent.wkt, "\n")
    res_parent = h3.h3_get_resolution(h3_cell_parent)
    
    # get the central descendent at the resolution of h3_cells_children
    res_children = h3.h3_get_resolution(h3_cells_children[0])
    centerhex = h3.h3_to_center_child(h = h3_cell_parent, res = res_children)

    # get the boundary of the multipolygon of the H3 cells union
    boundary_children_union_coords = h3.h3_set_to_multi_polygon(
                                               hexes = h3_cells_children,
                                               geo_json = True)[0][0]
    # close the linestring
    boundary_children_union_coords.append(boundary_children_union_coords[0])
    boundary_children_union = geometry.Polygon(boundary_children_union_coords)
    # print(boundary_children_union.wkt, "\n")
    
    # compute the overlapping geometry
    # (the intersection of the boundary_parent with boundary_children_union):
    overlap_geom = boundary_parent.intersection(boundary_children_union)
    print("overlap approx: {}".format(round(overlap_geom.area / boundary_parent.area, 4))) 

    # plot
    dict_adjust_textpos = {7: 0.0003, 8: 0.0001, 9: 0.00005, 10: 0.00002}
    
    for child in h3_cells_children:
        boundary_child_coords = h3.h3_to_geo_boundary(h = child, geo_json = True)
        boundary_child = geometry.Polygon(boundary_child_coords)
        ax.plot(*boundary_child.exterior.coords.xy, color = "grey", linestyle="--")
        
        dist_to_centerhex = h3.h3_distance(h1 = centerhex, h2 = child)
        list_distances_to_center.append(dist_to_centerhex)
                                
        if res_children <= res_parent + 3:
            # add text
            ax.text(x = boundary_child.centroid.x - dict_adjust_textpos[res_parent],
                    y = boundary_child.centroid.y - dict_adjust_textpos[res_parent],
                    s = str(dist_to_centerhex),
                    fontsize = 12, color = "black", weight = "bold")
    
    ax.plot(*boundary_children_union.exterior.coords.xy, color = "blue")
    ax.plot(*boundary_parent.exterior.coords.xy, color = "red", linewidth=2)
                                
    return list_distances_to_center

# %%
fig, ax = plt.subplots(2, 2, figsize = (20, 20))
list_distances_to_center_dc = plot_parent_and_descendents(h3_cell_parent, 
                                                          h3_cells_children, 
                                                          ax = ax[0][0])
list_distances_to_center_gc = plot_parent_and_descendents(h3_cell_parent,
                                                          h3_cells_grandchildren,
                                                          ax = ax[0][1])
list_distances_to_center_2xgc = plot_parent_and_descendents(h3_cell_parent, 
                                                            h3_cells_2xgrandchildren, 
                                                            ax = ax[1][0])
# list_distances_to_center_3xgc = plot_parent_and_descendents(h3_cell_parent,
#                                                             h3_cells_3xgrandchildren,
#                                                             ax = ax[1][1])


ax[0][0].set_title("Direct children (res 10)")
ax[0][1].set_title("Grandchildren (res 11)")
ax[1][0].set_title("Grandgrandchildren (res 12)")
# ax[1][1].set_title("Grandgrandgrandchildren (res 13)");
# ax[1][1].axis('off');

# %% [markdown]
# We could buffer the parent, so that all initial descendents are guaranteed to be included.   
# For this, we determine the incomplete hollow rings relative to the central child at given resolution.
# 
# By default (if complete), on hollow ring k there are $k * 6$ cells, for $k >=1$ 

# %%
def highlight_incomplete_hollowrings(list_distances_to_center):
    c = Counter(list_distances_to_center)
    print(c)
    list_incomplete = []
    for k in c:
        if (k > 1) and (c[k] != 6 * k):
            list_incomplete.append(k)
    print("List incomplete hollow rings:", sorted(list_incomplete))

# %%
highlight_incomplete_hollowrings(list_distances_to_center_dc)
print("-----------------------------------------------------")
highlight_incomplete_hollowrings(list_distances_to_center_gc)
print("-----------------------------------------------------")
highlight_incomplete_hollowrings(list_distances_to_center_2xgc)
print("-----------------------------------------------------")
# highlight_incomplete_hollowrings(list_distances_to_center_3xgc)

# %% [markdown]
# ## I.3. Spatial arrangement of H3 cells in the ij coordinate system
# 
# Read: https://h3geo.org/docs/core-library/coordsystems
# 

# %%
help(h3.experimental_h3_to_local_ij)

# %%
def explore_ij_coords(lat_point, lon_point, num_rings = 3, ax = None):

    # an example at resolution 9
    hex_id_ex = h3.geo_to_h3(lat = lat_point,
                             lng = lon_point,
                             resolution = 9
                             )
    assert(h3.h3_get_resolution(hex_id_ex) == 9)

    # get its rings
    list_siblings = list(h3.hex_range_distances(h = hex_id_ex, 
                                                K = num_rings))

    dict_ij = {}
    dict_color = {}
    dict_s = {}

    if ax is None:
        figsize = (min(6 * num_rings, 15), min(6 * num_rings, 15))
        fig, ax = plt.subplots(1, 1, figsize = figsize)

    for ring_level in range(len(list_siblings)):

        if ring_level == 0:
            fontcol = "red"
        elif ring_level == 1:
            fontcol = "blue"
        elif ring_level == 2:
            fontcol = "green"
        else:
            fontcol = "brown"

        if ring_level == 0:
            # on ring 0 is only hex_id_ex
            geom_boundary_coords = h3.h3_to_geo_boundary(hex_id_ex,
                                                         geo_json = True)
            geom_shp = geometry.Polygon(geom_boundary_coords)
            ax.plot(*geom_shp.exterior.xy, color = "purple")

            ij_ex = h3.experimental_h3_to_local_ij(origin = hex_id_ex,
                                                   h = hex_id_ex)
            s = " {} \n \n (0,0)".format(ij_ex)

            dict_ij[hex_id_ex] = ij_ex
            dict_color[hex_id_ex] = "red"
            dict_s[hex_id_ex] = s        

            ax.text(x = geom_shp.centroid.x - 0.0017,
                    y = geom_shp.centroid.y - 0.0005,
                    s = s,
                    fontsize = 11, color = fontcol, weight = "bold")
        else:
            # get the hex ids resident on ring_level
            siblings_on_ring = list(list_siblings[ring_level])

            k = 1
            for sibling_hex in sorted(siblings_on_ring):
                geom_boundary_coords = h3.h3_to_geo_boundary(sibling_hex,
                                                             geo_json=True)
                geom_shp = geometry.Polygon(geom_boundary_coords)
                ax.plot(*geom_shp.exterior.xy, color = "purple")

                ij = h3.experimental_h3_to_local_ij(origin = hex_id_ex,
                                                    h = sibling_hex)
                ij_diff = (ij[0] - ij_ex[0], ij[1] - ij_ex[1])
                s = " {} \n \n {}".format(ij, ij_diff)
                k = k + 1

                dict_ij[sibling_hex] = ij    
                dict_color[sibling_hex] = fontcol
                dict_s[sibling_hex] = s

                ax.text(x = geom_shp.centroid.x - 0.0017,
                        y = geom_shp.centroid.y - 0.0005,
                        s = s,
                        fontsize = 11, color = fontcol, weight = "bold")

    ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")
    
    return dict_ij, dict_color, dict_s

# %%
dict_ij, dict_color, dict_s = explore_ij_coords(lat_point = lat_centr_point,
                                                lon_point = lon_centr_point)

# %%
dict_ij

# %% [markdown]
# Note that choosing a GPS point in other parts of the world results in different relative i and j arrangements (with respect to compass NESW).  
# Here is an illustration for ring 1 neighbours: 
# 

# %%
fig, ax = plt.subplots(2, 2, figsize = (12, 12))

# in Toulouse
_ = explore_ij_coords(lat_point = lat_centr_point,
                      lon_point = lon_centr_point,
                      num_rings = 1,
                      ax = ax[0][0])
ax[0][0].set_title("Toulouse (FR)")

# in New York
_ = explore_ij_coords(lat_point = 40.665634, 
                      lon_point = -73.964768,
                      num_rings = 1,
                      ax = ax[0][1])
ax[0][1].set_title("New York (US)")

# in Singapore
_ = explore_ij_coords(lat_point = 1.282892, 
                      lon_point = 103.862396,
                      num_rings = 1,
                      ax = ax[1][0])
ax[1][0].set_title("Singapore (SG)")


# in Stockholm 
_ = explore_ij_coords(lat_point = 59.330506, 
                      lon_point = 18.072043,
                      num_rings = 1,
                      ax = ax[1][1])
ax[1][1].set_title("Stockholm (SE)");


# %% [markdown]
# Anticipating the ML section of this notebook, we put these 4 rings of hexagons in a 2d array.  
# A preliminary step is to transform i and j as follows:

# %%
dict_ij

# %%
min_i = min([dict_ij[h][0] for h in dict_ij])
min_j = min([dict_ij[h][1] for h in dict_ij])

max_i = max([dict_ij[h][0] for h in dict_ij])
max_j = max([dict_ij[h][1] for h in dict_ij])

print("i between {} and {}".format(min_i, max_i))
print("j between {} and {}".format(min_j, max_j))

# rescale
dict_ij_rescaled = {}
for h in dict_ij:
    dict_ij_rescaled[h] = [dict_ij[h][0] - min_i, dict_ij[h][1] - min_j]
    print(dict_ij[h], "-->", dict_ij_rescaled[h])


# %%
fig, ax = plt.subplots(1, 1, figsize = (10, 10))

i_range = list(range(0, max_i - min_i + 1))
j_range = list(range(0, max_j - min_j + 1))


ax.set_xticks(np.arange(len(j_range)))
ax.set_yticks(np.arange(len(i_range)))
ax.set_xticklabels(j_range)
ax.set_yticklabels(i_range)

minor_ticks_x = np.arange(-1, max_j - min_j + 1, 0.5)
minor_ticks_y = np.arange(-1, max_i - min_i + 1, 0.5)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(minor_ticks_y, minor=True)

for h in dict_ij_rescaled:
    ax.text(x = dict_ij_rescaled[h][1],
            y = dict_ij_rescaled[h][0],
            s = dict_s[h],
            fontsize = 11, color = dict_color[h],
            ha="center", va="center", weight = "bold")
    
ax.set_xlim(-1, max_j - min_j + 1)
ax.set_ylim(-1, max_i - min_i + 1)

ax.grid(which='major', alpha = 0.1)
ax.grid(which='minor', alpha = 0.9)

ax.set_xlabel("J")
ax.set_ylabel("I")

ax.invert_yaxis()

fig.tight_layout();

# %% [markdown]
# ------------------------

# %% [markdown]
# -------------------

# %% [markdown]
# # II. Use H3 indexing for spatial operations

# %% [markdown]
# ## II.1. Prepare data - GeoJSON file of bus stops

# %%
def load_and_prepare_busstops(filepath):
    """Loads a geojson files of point geometries and features,
    extracts the latitude and longitude into separate columns,
    deduplicates busstops (since multiple buslines share them)"""

    gdf_raw = gpd.read_file(filepath, driver="GeoJSON")
    print("Total number of bus stops in original dataset", gdf_raw.shape[0]) 

    gdf_raw["latitude"] = gdf_raw["geometry"].apply(lambda p: p.y)
    gdf_raw["longitude"] = gdf_raw["geometry"].apply(lambda p: p.x)

    # reset index to store it in a column
    gdf_raw.reset_index(inplace=True, drop = False)
    
    return gdf_raw

# %%
input_file_busstops = "datasets_demo/busstops_Toulouse.geojson"
gdf_raw = load_and_prepare_busstops(filepath = input_file_busstops)

# display first 5 rows of the geodataframe, transposed
gdf_raw.head().T

# %%
def base_empty_map():
    """Prepares a folium map centered in a central GPS point of Toulouse"""
    m = Map(location = [43.600378, 1.445478],
            zoom_start = 9.5,
            tiles = "cartodbpositron",
            attr = '''© <a href="http://www.openstreetmap.org/copyright">
                      OpenStreetMap</a>contributors ©
                      <a href="http://cartodb.com/attributions#basemaps">
                      CartoDB</a>'''
            )
    return m

# %%
# quick visualization on map of raw data

m = base_empty_map()
mc = MarkerCluster()

gdf_dedup = gdf_raw.drop_duplicates(subset=["latitude", "longitude"])
print("Total number of bus stops in deduplicated dataset", gdf_dedup.shape[0]) 

for i, row in gdf_dedup.iterrows():
    mk = Marker(location=[row["latitude"], row["longitude"]])
    mk.add_to(mc)

mc.add_to(m)
m

# %%
# fig, ax = plt.subplots(1, 1, figsize = (15, 10))

# im1 = pilim.open('images/2_markers_busstops.png', 'r')
# ax.imshow(np.asarray(im1))
# ax.set_axis_off();


# %% [markdown]
# Better yet, we can plot a heatmap with pydeck (Docs at https://pydeck.gl/index.html):

# %%
help(pydeck.Deck.__init__)

# %%
# print(dir(cm.linear))

steps = 5
color_map = cm.linear.RdYlGn_10.scale(0, 1).to_step(steps)

# in reverse order (green to red)
for i in range(steps-1, -1, -1):
    # would be fractional values, but we need them as RGB in [0,255]
    # also drop the alpha (4th element)
    print([int(255 * x) for x in color_map.colors[i][:-1]])

color_map

# %%
COLOR_SCALE = [
    [0, 104, 55],
    [117, 195, 100],
    [235, 231, 139],
    [246, 125, 74],
    [165, 0, 38]
]

busstops_layer = pydeck.Layer(
                    "HeatmapLayer",
                    data = gdf_dedup,
                    opacity = 0.2,
                    get_position = ["longitude", "latitude"],
                    threshold = 0.05,
                    intensity = 1,
                    radiusPixels = 30,
                    pickable = False,
                    color_range=COLOR_SCALE,
                )

view = pydeck.data_utils.compute_view(gdf_dedup[["longitude", "latitude"]])
view.zoom = 6

MAPBOX_TOKEN = 'pk.eyJ1Ijoic2FiIiwiYSI6ImNsNDE3bGR3bzB2MmczaXF5dmxpaTloNmcifQ.NQ-B8jBPtOd53tNYt42Gqw';

r = pydeck.Deck(
    layers=[busstops_layer],
    initial_view_state = view,
    api_keys = {"mapbox": MAPBOX_TOKEN},
    map_style='mapbox://styles/mapbox/light-v9',
    map_provider='mapbox'
)



# %%
r.to_html('busstops.html', notebook_display=True)

# %%


# %%


# %%
# fig, ax = plt.subplots(1, 1, figsize = (15, 10))

# im1 = pilim.open('images/heatmap_busstop_.png', 'r')
# ax.imshow(np.asarray(im1))
# ax.set_axis_off();

# %% [markdown]
# **Create a new dataframe to work with throughout the notebook:**

# %%
gdf_raw.head(2)
# rename columen conc_ligne to ligne
# gdf_raw.rename(columns={"conc_ligne": "ligne"}, inplace=True)

# %%
gdf_raw_cpy = gdf_raw.reset_index(inplace = False, drop = False)
df_stops_to_buslines = gdf_raw_cpy.groupby(by=["longitude", "latitude"]).agg(
                                    {"index": list, "ligne": set, "nom_log": "first"})

df_stops_to_buslines["info"] = df_stops_to_buslines[["nom_log", "ligne"]].apply(
                                  lambda x: "{} ({})".format(x[0], ",".join(list(x[1]))), 
                                  axis = 1)
df_stops_to_buslines.reset_index(inplace = True, drop = False)
df_stops_to_buslines.head()


# %%
# count rows in pandas dataframe
df_stops_to_buslines.shape[0]

# %% [markdown]
# ## II.2. Index data spatially with H3

# %%
# index each data point into the spatial index of the specified resolution
for res in range(7, 11):
    col_hex_id = "hex_id_{}".format(res)
    col_geom = "geometry_{}".format(res)
    msg_ = "At resolution {} -->  H3 cell id : {} and its geometry: {} "
    print(msg_.format(res, col_hex_id, col_geom))

    df_stops_to_buslines[col_hex_id] = df_stops_to_buslines.apply(
                                        lambda row: h3.geo_to_h3(
                                                    lat = row["latitude"],
                                                    lng = row["longitude"],
                                                    resolution = res),
                                        axis = 1)

    # use h3.h3_to_geo_boundary to obtain the geometries of these hexagons
    df_stops_to_buslines[col_geom] = df_stops_to_buslines[col_hex_id].apply(
                                        lambda x: {"type": "Polygon",
                                                   "coordinates":
                                                   [h3.h3_to_geo_boundary(
                                                       h=x, geo_json=True)]
                                                   }
                                         )
# transpose for better display
df_stops_to_buslines.head().T

# %% [markdown]
# ## II.3 Compute K Nearest Neighbors (spatial search) using the H3 index

# %%
help(h3.hex_ring)

# %% [markdown]
# Create an inverted index hex_id_9 to list of row indices in df_stops_to_buslines:

# %%
resolution_lookup = 9
hexes_column = "hex_id_{}".format(resolution_lookup)
print("Will operate on column: ", hexes_column)
df_aux = df_stops_to_buslines[[hexes_column]]
df_aux.reset_index(inplace = True, drop = False)
# columns are [index, hex_id_9]
lookup_hex_to_indices = pd.DataFrame(
                          df_aux.groupby(by = hexes_column)["index"].apply(list)
                        ).reset_index(inplace = False, drop = False)
lookup_hex_to_indices.rename(columns = {"index": "list_indices"}, inplace = True)
lookup_hex_to_indices["num_indices"] = lookup_hex_to_indices["list_indices"].apply(
                                                                   lambda x: len(x))

lookup_hex_to_indices.set_index(hexes_column, inplace = True)

print("Using {} hexagons".format(lookup_hex_to_indices.shape[0]))
lookup_hex_to_indices.sort_values(by = "num_indices", ascending = False).head()

# %% [markdown]
# For a given GPS location, we index it and then iterate over its hollow rings until we collect the candidates. Last step for computing result in descending distance, is to compute the actual Haversine distance: 

# %%
chosen_point = (43.595707, 1.452252)
num_neighbors_wanted = 10

hex_source = h3.geo_to_h3(lat = chosen_point[0],
                          lng = chosen_point[1], 
                          resolution = 9)

list_candidates = []
rest_needed = num_neighbors_wanted - len(list_candidates)
ring_seqno = 0
hexes_processed = []

while rest_needed > 0:
    list_hexes_hollow_ring = list(h3.hex_ring(h = hex_source, k = ring_seqno))
    for hex_on_ring in list_hexes_hollow_ring:
        try:
            new_candidates = lookup_hex_to_indices.loc[hex_on_ring]["list_indices"]
            list_candidates.extend(new_candidates)
        except Exception:
            # we may get KeyError when no entry in lookup_hex_to_indices for a hex id
            pass
        hexes_processed.append(hex_on_ring)
    
    msg_ = "processed ring: {}, candidates before: {}, candidates after: {}"
    print(msg_.format(ring_seqno, 
                      num_neighbors_wanted - rest_needed, 
                      len(list_candidates)))
    
    rest_needed = num_neighbors_wanted - len(list_candidates)
    ring_seqno = ring_seqno + 1
    
print("Candidate rows: \n", list_candidates)

# %%
def haversine_dist(lon_src, lat_src, lon_dst, lat_dst):
    '''returns distance between GPS points, measured in meters'''

    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, 
                                                 [lon_src, lat_src, lon_dst, lat_dst])

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * \
        np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km * 1000

# %%
ordered_candidates_by_distance = []

for candid in list_candidates:
    candid_busstop_lat = df_stops_to_buslines.iloc[candid]["latitude"]
    candid_busstop_lon = df_stops_to_buslines.iloc[candid]["longitude"]
    
    # compute Haversine to source
    dist_to_source = haversine_dist(lon_src = chosen_point[1], 
                                    lat_src = chosen_point[0], 
                                    lon_dst = candid_busstop_lon,
                                    lat_dst = candid_busstop_lat)
    
    if len(ordered_candidates_by_distance) == 0:
        ordered_candidates_by_distance.append((dist_to_source, candid))
    else:
        bisect.insort(ordered_candidates_by_distance, (dist_to_source, candid))


pprint(ordered_candidates_by_distance)

print("-------------------------------------------------")
# the final result
final_result = ordered_candidates_by_distance[0:num_neighbors_wanted]
list_candidates_result = [x[1] for x in final_result]
print(list_candidates_result)

# %%
# plot the candidates
fig, ax = plt.subplots(1, 1, figsize = (10, 10))

for hex_id in hexes_processed:
    geom_boundary_coords = h3.h3_to_geo_boundary(hex_id,
                                                 geo_json = True)
    geom_shp = geometry.Polygon(geom_boundary_coords)
    ax.plot(*geom_shp.exterior.xy, color = "purple")
    
# the source in red
circle_source = plt.Circle((chosen_point[1], chosen_point[0]), 
                           0.00025, color='red')
ax.add_artist(circle_source)

print("Nearest bus stops: \n======================================")

# the nearest candidates in green, the rest of the candidates in orange
for candid in list_candidates:
    candid_busstop_lat = df_stops_to_buslines.iloc[candid]["latitude"]
    candid_busstop_lon = df_stops_to_buslines.iloc[candid]["longitude"]
    candid_busstop_info = df_stops_to_buslines.iloc[candid]["info"]
    
    print("{}".format(candid_busstop_info))
    
    if candid in list_candidates_result:
        circle_candid = plt.Circle((candid_busstop_lon, candid_busstop_lat), 
                                   0.00025, color='green')
        # draw a line if it's in he nearest neighbours final result
        ax.plot([chosen_point[1], candid_busstop_lon], 
                [chosen_point[0], candid_busstop_lat], 
                'green', linestyle=':', marker='')
    else:    
        circle_candid = plt.Circle((candid_busstop_lon, candid_busstop_lat), 
                                   0.00025, color='orange')
    ax.add_artist(circle_candid)
    

# %% [markdown]
# Note: there exist bus stops on the 2nd hollow ring that are nearer to the source (which is marked by red circle) than some of the bus stops on the 1st hollow ring.  
# So it is adviseabale to always include one additional hollow ring of candidates before computing Haversine distance.

# %% [markdown]
# -------------

# %% [markdown]
# ## II.4. Compute Point in Polygon (spatial join) using the H3 index

# %% [markdown]
# For this demo, we use the set of districts of Toulouse:

# %%
def load_and_prepare_districts(filepath):
    """Loads a geojson files of polygon geometries and features,
    swaps the latitude and longitude and stores geojson"""

    gdf_districts = gpd.read_file(filepath, driver="GeoJSON")
    
    gdf_districts["geom_geojson"] = gdf_districts["geometry"].apply(
                                              lambda x: geometry.mapping(x))

    gdf_districts["geom_swap"] = gdf_districts["geometry"].map(
                                              lambda polygon: ops.transform(
                                                  lambda x, y: (y, x), polygon))

    gdf_districts["geom_swap_geojson"] = gdf_districts["geom_swap"].apply(
                                              lambda x: geometry.mapping(x))
    
    # convert multipolygons to polygons
    gdf_districts["geom_polygon"] = gdf_districts["geom_geojson"].apply(
        lambda x: x["coordinates"][0] if x["type"] == "MultiPolygon" else x["coordinates"]
    )
    
    return gdf_districts


# %%
input_file_districts = "datasets_demo/districts_Toulouse.geojson"
gdf_districts = load_and_prepare_districts(filepath = input_file_districts) 
 
print(gdf_districts.shape)
print("\n--------------------------------------------------------\n")
list_districts = list(gdf_districts["libelle_du_grand_quartier"].unique())
list_districts.sort()
print(columnize(list_districts, displaywidth=100))
print("\n--------------------------------------------------------\n")

gdf_districts[["libelle_du_grand_quartier", "geometry", 
               "geom_swap", "geom_swap_geojson", "geom_polygon"]].head()

# %% [markdown]
# The approach is to fill each district geometry with hexgons at resolution 13 and then compact them.
# 
# **Initial fill:**

# %%
help(h3.polyfill)

# %%
def fill_hexagons(geom_geojson, res, flag_swap = False, flag_return_df = False):
    """Fills a geometry given in geojson format with H3 hexagons at specified
    resolution. The flag_reverse_geojson allows to specify whether the geometry
    is lon/lat or swapped"""

    set_hexagons = h3.polyfill(geojson = geom_geojson,
                               res = res,
                               geo_json_conformant = flag_swap)
    list_hexagons_filling = list(set_hexagons)

    if flag_return_df is True:
        # make dataframe
        df_fill_hex = pd.DataFrame({"hex_id": list_hexagons_filling})
        df_fill_hex["value"] = 0
        df_fill_hex['geometry'] = df_fill_hex.hex_id.apply(
                                    lambda x:
                                    {"type": "Polygon",
                                     "coordinates": [
                                        h3.h3_to_geo_boundary(h=x,
                                                              geo_json=True)
                                        ]
                                     })
        assert(df_fill_hex.shape[0] == len(list_hexagons_filling))
        return df_fill_hex
    else:
        return list_hexagons_filling


# %%
gdf_districts["hex_fill_initial"] = gdf_districts["geom_swap_geojson"].apply(
                                         lambda x: list(fill_hexagons(geom_geojson = x, 
                                                                      res = 13))
                                          )
gdf_districts["num_hex_fill_initial"] = gdf_districts["hex_fill_initial"].apply(len)

total_num_hex_initial = gdf_districts["num_hex_fill_initial"].sum()
print("Until here, we'd have to search over {} hexagons".format(total_num_hex_initial))

gdf_districts[["libelle_du_grand_quartier", "geometry", "num_hex_fill_initial"]].head()

# %% [markdown]
# To reduce the number of hexagons we can benefit from H3 cells compacting.
# 
# **Compacted fill:**

# %%
help(h3.compact)

# %%
gdf_districts["hex_fill_compact"] = gdf_districts["hex_fill_initial"].apply(
                                                lambda x: list(h3.compact(x)))
gdf_districts["num_hex_fill_compact"] = gdf_districts["hex_fill_compact"].apply(len)

print("Reduced number of cells from {} to {} \n".format(
            gdf_districts["num_hex_fill_initial"].sum(),
            gdf_districts["num_hex_fill_compact"].sum()))

# count cells by index resolution after compacting

gdf_districts["hex_resolutions"] = gdf_districts["hex_fill_compact"].apply(
                                            lambda x: 
                                            [h3.h3_get_resolution(hexid) for hexid in x])
gdf_districts["hex_resolutions_counts"] = gdf_districts["hex_resolutions"].apply(
                                            lambda x: Counter(x))


gdf_districts[["libelle_du_grand_quartier", "geometry", 
               "num_hex_fill_initial", "num_hex_fill_compact", 
               "hex_resolutions_counts"]].head()

# %%
# this column of empty lists is a placeholder, will be used further in this section
gdf_districts["compacted_novoids"] = [[] for _ in range(gdf_districts.shape[0])]

# %%
def plot_basemap_region_fill(df_boundaries_zones, initial_map = None):
    
    """On a folium map, add the boundaries of the geometries in geojson formatted
       column of df_boundaries_zones"""

    if initial_map is None:
        initial_map = base_empty_map()

    feature_group = folium.FeatureGroup(name='Boundaries')

    for i, row in df_boundaries_zones.iterrows():
        feature_sel = Feature(geometry = row["geom_geojson"], id=str(i))
        feat_collection_sel = FeatureCollection([feature_sel])
        geojson_subzone = json.dumps(feat_collection_sel)

        GeoJson(
                geojson_subzone,
                style_function=lambda feature: {
                    'fillColor': None,
                    'color': 'blue',
                    'weight': 5,
                    'fillOpacity': 0
                }
            ).add_to(feature_group)

    feature_group.add_to(initial_map)
    return initial_map

# ---------------------------------------------------------------------------


def hexagons_dataframe_to_geojson(df_hex, hex_id_field,
                                  geometry_field, value_field,
                                  file_output = None):

    """Produce the GeoJSON representation containing all geometries in a dataframe
     based on a column in geojson format (geometry_field)"""

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(geometry = row[geometry_field],
                          id = row[hex_id_field],
                          properties = {"value": row[value_field]})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    geojson_result = json.dumps(feat_collection)

    # optionally write to file
    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    return geojson_result

# ---------------------------------------------------------------------------------


def map_addlayer_filling(df_fill_hex, layer_name, map_initial, fillcolor = None):
    """ On a folium map (likely created with plot_basemap_region_fill),
        add a layer of hexagons that filled the geometry at given H3 resolution
        (df_fill_hex returned by fill_hexagons method)"""

    geojson_hx = hexagons_dataframe_to_geojson(df_fill_hex,
                                               hex_id_field = "hex_id",
                                               value_field = "value",
                                               geometry_field = "geometry")

    GeoJson(
            geojson_hx,
            style_function=lambda feature: {
                'fillColor': fillcolor,
                'color': 'red',
                'weight': 2,
                'fillOpacity': 0.1
            },
            name = layer_name
        ).add_to(map_initial)

    return map_initial

# -------------------------------------------------------------------------------------


def visualize_district_filled_compact(gdf_districts, 
                                      list_districts_names, 
                                      fillcolor = None):
       
    overall_map = base_empty_map()
    gdf_districts_sel = gdf_districts[gdf_districts["libelle_du_grand_quartier"]
                                      .isin(list_districts_names)] 
    
    map_district = plot_basemap_region_fill(gdf_districts_sel, 
                                            initial_map = overall_map)
    
    for i, row in gdf_districts_sel.iterrows():
    
        district_name = row["libelle_du_grand_quartier"]
        if len(row["compacted_novoids"]) > 0:
            list_hexagons_filling_compact = row["compacted_novoids"]
        else:
            list_hexagons_filling_compact = []
            
        list_hexagons_filling_compact.extend(row["hex_fill_compact"])
        list_hexagons_filling_compact = list(set(list_hexagons_filling_compact))

        # make dataframes
        df_fill_compact = pd.DataFrame({"hex_id": list_hexagons_filling_compact})
        df_fill_compact["value"] = 0
        df_fill_compact['geometry'] = df_fill_compact.hex_id.apply(
                                        lambda x: 
                                        {"type": "Polygon",
                                         "coordinates": [
                                             h3.h3_to_geo_boundary(h=x,
                                                                   geo_json=True)
                                         ]
                                         })

        map_fill_compact = map_addlayer_filling(df_fill_hex = df_fill_compact, 
                                                layer_name = district_name,
                                                map_initial = map_district,
                                                fillcolor = fillcolor)
        
    folium.map.LayerControl('bottomright', collapsed=True).add_to(map_fill_compact)

    return map_fill_compact

# %%
list_districts_names = ["MIRAIL-UNIVERSITE", "BAGATELLE", "PAPUS",
                        "FAOURETTE", "CROIX-DE-PIERRE"]
visualize_district_filled_compact(gdf_districts = gdf_districts,
                                  list_districts_names = list_districts_names)

# %%
# fig, ax = plt.subplots(1, 1, figsize=(16, 16))

# im1 = pilim.open('images/districts_fill_compact.png', 'r')
# ax.imshow(np.asarray(im1))
# ax.set_title("Polyfill compacted for selected districts")
# ax.set_axis_off();


# %% [markdown]
# In the detail zoom that follows, we can observe that some small areas remained uncovered after compacting the set of hexagons used for filling districts geometries.   
# These small voids occur at the juxtaposition of hexagon cells of different H3 resolutions. 
# As explained in section I.2 of the preliminaries, the parent's polygon does not overlap completely with the multipolygon of its children union.
# 
# A consequence of this, for our spatial join, is that any point that would fall exactly in such a void would  be wrongly labelled as outside the district.
# 

# %%
# fig, ax = plt.subplots(1, 1, figsize=(16, 16))

# im1 = pilim.open('images/compacted_voids.png', 'r')
# ax.imshow(np.asarray(im1))
# ax.set_title("Polyfill compacted for selected districts (voids zoomed in)")
# ax.set_axis_off();

# %% [markdown]
# So far, how many hexagons belonged to more than one district (i.e were on the border between districts)?

# %%
def check_hexes_on_multiple_districts(gdf_districts, hexes_column):
    
    # map district name --> list of cells after compacting
    dict_district_hexes = dict(zip(gdf_districts["libelle_du_grand_quartier"], 
                                   gdf_districts[hexes_column]))

    # reverse dict to map cell id --> district name 
    # basically we're performing an inverting of a dictionary with list values

    dict_hex_districts = {}
    for k, v in dict_district_hexes.items():
        for x in v:
            dict_hex_districts.setdefault(x, []).append(k)

    list_keys = list(dict_hex_districts.keys())
    print("Total number of keys in dict reversed:", len(list_keys))
    print("Example:", list_keys[0], " ==> ", dict_hex_districts[list_keys[0]])

    print("---------------------------------------------------")
    # check if any hex maps to more than 1 district name
    dict_hex_of_multiple_districts = {}
    for k, v in dict_hex_districts.items():
        if len(v) > 1:
            dict_hex_of_multiple_districts[k] = v

    print("Hexes mapped to multiple districts:", 
          len(dict_hex_of_multiple_districts.keys()))
    c = Counter([h3.h3_get_resolution(k) for k in dict_hex_of_multiple_districts])
    pprint(c)
    
    return dict_hex_districts

# %%
_ = check_hexes_on_multiple_districts(gdf_districts, hexes_column = "hex_fill_compact")

# %% [markdown]
# **Fill the voids**

# %%
help(h3.h3_line)

# %%
def get_hexes_traversed_by_borders(gdf_districts, res):
    """Identify the resolution 12 hexagons that are traversed by districts boundaries"""
    set_traversed_hexes = set()
    
    for i, row in gdf_districts.iterrows():
        coords = row["geometry"].boundary.coords
        for j in range(len(coords)-1):
            # for each "leg" (segment) of the linestring
            start_leg = coords[j]
            stop_leg = coords[j]
            # note: they are (lon,lat)
            start_hexid = h3.geo_to_h3(lat = start_leg[1],
                                       lng = start_leg[0],
                                       resolution = res)
            stop_hexid = h3.geo_to_h3(lat = stop_leg[1],
                                      lng = stop_leg[0],
                                      resolution = res)
            traversed_hexes = h3.h3_line(start = start_hexid,
                                         end = stop_hexid) 
            set_traversed_hexes |= set(traversed_hexes)
            
    return list(set_traversed_hexes)   
    

# %%
boundary_hexes_res11 = get_hexes_traversed_by_borders(gdf_districts, res = 11)
boundary_hexes_res12 = get_hexes_traversed_by_borders(gdf_districts, res = 12)
boundary_hexes_res13 = get_hexes_traversed_by_borders(gdf_districts, res = 13)

print("{} hexes on boundary at res {}".format(len(boundary_hexes_res11), 11))
print("{} hexes on boundary at res {}".format(len(boundary_hexes_res12), 12))
print("{} hexes on boundary at res {}".format(len(boundary_hexes_res13), 13))

# %%
def fill_voids(row, fill_voids_res = 12):
    """For each cell resulted from compacting, get its central child at resolution
    fill_voids_res; compute specific hollow rings of this central child, overall achieving
    an envelope(buffer) of each of the coarser hexagons with more fine-grained hexagons"""
    
    hexes_compacted = row["hex_fill_compact"]
    
    set_fillvoids = set()
    for i in range(len(hexes_compacted)):
        hex_id = hexes_compacted[i]
        res_hex = h3.h3_get_resolution(hex_id)
        if res_hex < fill_voids_res:
            center_hex = h3.h3_to_center_child(h = hex_id, 
                                               res = fill_voids_res)
            if res_hex - fill_voids_res == -4:
                # e.g. res_hex = 8, fill_voids_res = 12
                # ==> include 3xgrandchildren on rings [30, .., 32, 33]
                for j in range(30, 34):
                    hollow_ring = h3.hex_ring(h = center_hex, k = j)
                    set_fillvoids |= hollow_ring                    
            elif res_hex - fill_voids_res == -3:
                # e.g. res_hex = 9, fill_voids_res = 12
                # ==> include 2xgrandchildren on rings [10,11,12]
                for j in range(10, 13):
                    hollow_ring = h3.hex_ring(h = center_hex, k = j)
                    set_fillvoids |= hollow_ring  
            elif res_hex - fill_voids_res == -2:
                # e.g. res_hex = 10, fill_voids_res = 12
                # ==> include grandchildren on rings 4 and 5
                for j in [4, 5]:
                    hollow_ring = h3.hex_ring(h = center_hex, k = j)
                    set_fillvoids |= hollow_ring 
            elif res_hex - fill_voids_res == -1:
                # e.g. res_hex = 11, fill_voids_res = 12
                # ==> include children on ring 1
                for j in [1]:
                    hollow_ring = h3.hex_ring(h = center_hex, k = j)
                    set_fillvoids |= hollow_ring 

    # exclude any hexagon that would be on border
    set_interior = (set_fillvoids - set(boundary_hexes_res13)) - set(boundary_hexes_res12)
    list_interior = list(set_interior)
    return list_interior

# %%
%%time
gdf_districts["compacted_novoids"] = gdf_districts.apply(lambda r: fill_voids(r), axis = 1)

# %%
_ = check_hexes_on_multiple_districts(
                          gdf_districts, 
                          hexes_column = "compacted_novoids")

# %%
list_districts_names = ["MIRAIL-UNIVERSITE", "BAGATELLE", "PAPUS",
                        "FAOURETTE", "CROIX-DE-PIERRE"]
visualize_district_filled_compact(gdf_districts = gdf_districts,
                                  list_districts_names = list_districts_names)

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/districts_fill_compact_novoids.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("Polyfill compacted for selected districts (filled voids)")
ax.set_axis_off();

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/filled_voids.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("Polyfill compacted for selected districts (filled voids zoomed in)")
ax.set_axis_off()

# %%
# sidenote - how it works itertools.chain.from_iterable
l1 = ["a", "b"]
l2 = ["a", "c"]
list(itertools.chain.from_iterable([l1, l2]))

# %%
gdf_districts["union_compacted_novoids"] = \
             gdf_districts[["compacted_novoids", "hex_fill_compact"]].apply(
             lambda x: list(itertools.chain.from_iterable([x[0], x[1]])), axis = 1)
gdf_districts["union_compacted_novoids"] = gdf_districts["union_compacted_novoids"].apply(
             lambda x: list(set(x)))
gdf_districts["num_final"] = gdf_districts["union_compacted_novoids"].apply(
             lambda x: len(x))

gdf_districts["num_final"].sum()

# %% [markdown]
# Note: these 282148 multi-resolution H3 cells seem as a good trade-off compared with the former 2 extremes: the initial dense filling at resolution 13 with 2851449 hexagons versus the 94287 hexagons after compacting which left uncovered areas(voids) 

# %%
dict_hex_districts = check_hexes_on_multiple_districts(
                          gdf_districts, 
                          hexes_column = "union_compacted_novoids")

# %% [markdown]
# Now, for a given point, index it at all resolutions between 6 and 12 and search starting from coarser resolution towards finer resolutions:

# %%
def spatial_join_districts(row, dict_hex_districts, minres_compact, maxres_compact):
    for res in range(minres_compact, maxres_compact + 1):
        hexid = h3.geo_to_h3(lat = row["latitude"], 
                             lng = row["longitude"], 
                             resolution = res)
        if hexid in dict_hex_districts:
            if len(dict_hex_districts[hexid]) > 1:
                return ",".join(dict_hex_districts[hexid])
            else:
                return dict_hex_districts[hexid][0]
    return "N/A"

# %%
# dict_hex_districts

# %%
list_res_after_compact_novoids = [h3.h3_get_resolution(x) for x in dict_hex_districts]
finest_res = max(list_res_after_compact_novoids)
coarsest_res = min(list_res_after_compact_novoids)
print("Resolution between {} and {}".format(coarsest_res, finest_res))

# %%
# list_res_after_compact_novoids

# %%
%%time

df_sjoin_h3 = df_stops_to_buslines.copy()

df_sjoin_h3["district"] = df_sjoin_h3.apply(spatial_join_districts, 
                                            args=(dict_hex_districts,
                                                  coarsest_res,
                                                  finest_res), 
                                            axis = 1)

# %%
counts_by_district = pd.DataFrame(df_sjoin_h3["district"].value_counts())
counts_by_district.columns = ["num_busstops"]
counts_by_district.head()

# %% [markdown]
# Note: the N/A category includes all busstops that are outside the districts (but in the wider metropolitan area of Toulouse)

# %%
# the number of bus stops that were found inside the districts
counts_by_district[counts_by_district.index != "N/A"]["num_busstops"].sum()

# %%
# bus stops situated on the border of 2 districts
counts_by_district[counts_by_district.index.str.contains(",")]

# %%
# get the index as a list
list_districts_at_boundary_names = counts_by_district[counts_by_district.index.str.contains(",")].index.tolist()
print(list_districts_at_boundary_names)

# %%
special_map = visualize_district_filled_compact(
                     gdf_districts = gdf_districts,
                     list_districts_names =["AMIDONNIERS", "CASSELARDIT"],
                     fillcolor="pink")

# counts_by_district[counts_by_district.index.str.contains(",")]
# get df subset who's "district" name is in list_districts_at_boundary_names
df_on_border = df_sjoin_h3[df_sjoin_h3["district"].isin(list_districts_at_boundary_names)]
# df_on_border = df_sjoin_h3[df_sjoin_h3["district"] == "AMIDONNIERS,CASSELARDIT"]

for i, row in df_on_border.iterrows():
    mk = Marker(location=[row["latitude"], row["longitude"]],
                icon = folium.Icon(icon='circle', color='darkgreen'),
                popup=str(row["info"]))
    mk.add_to(special_map)
    
special_map

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/onborder_districts.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("Polyfill compacted for selected districts")
ax.set_axis_off()

# %% [markdown]
# **After having computed the spatial join, we can use the results for identifying which are the districts served by each bus line**

# %%
gdf_raw.head()

# %%
selected_busline = "14"
print(gdf_raw[gdf_raw["ligne"] == selected_busline]["pty"].unique())
print(gdf_raw[gdf_raw["ligne"] == selected_busline].groupby(by="pty")["sens"].apply(set))

df_route_busline = pd.merge(left = gdf_raw[gdf_raw["pty"].isin(['14/106', '14/13'])], 
                            right = df_sjoin_h3,
                            left_on = ["latitude", "longitude"],
                            right_on = ["latitude", "longitude"],
                            how = "left")

df_route_busline.sort_values(by = ["pty", "sens", "ordre"], inplace = True)
df_route_busline[["pty", "sens", "ordre", "info", "district"]]

# %%
direction_0 = df_route_busline[df_route_busline["sens"] == 0]["district"]
list(unique_everseen(direction_0))

# %%
list_aux = list(unique_everseen(df_route_busline["district"].values))
list_distr = []
for s in list_aux:
    if "," in s:
        # if on border, add both districts 
        list_distr.extend(s.split(","))
    else:
        list_distr.append(s)
        
gdf_bus_traversed_districts = gdf_districts[
                          gdf_districts["libelle_du_grand_quartier"].isin(list_distr)]
gdf_bus_traversed_districts = gdf_bus_traversed_districts[
                                            ["geometry", "libelle_du_grand_quartier"]]
gdf_bus_traversed_districts.to_file("datasets_demo/bus_14_districts.geojson",
                                    driver = "GeoJSON")

# %%
!ls -alh datasets_demo/bus_14_districts.geojson

# %% [markdown]
# Recall that we have comma in district when the point was found on the border between 2 districts.
# 
# Prepare files for the section V.1

# %%
list_projected_columns = ["latitude", "longitude", "sens", "ordre", "info", "district"]
df_route_busline_cpy = df_route_busline[list_projected_columns]
df_route_busline_cpy.sort_values(by = ["sens", "ordre"], inplace = True)

# shift
df_route_busline_cpy['next_longitude'] = df_route_busline_cpy["longitude"].shift(-1)
df_route_busline_cpy['next_latitude'] = df_route_busline_cpy["latitude"].shift(-1)
df_route_busline_cpy['next_sens'] = df_route_busline_cpy["sens"].shift(-1)
df_route_busline_cpy['next_ordre'] = df_route_busline_cpy["ordre"].shift(-1)

# the last row will have next_{} all none, we manually match it to the start of the route
df_route_busline_cpy["next_latitude"].fillna(df_route_busline_cpy.iloc[0]["latitude"],
                                             inplace=True)
df_route_busline_cpy["next_longitude"].fillna(df_route_busline_cpy.iloc[0]["longitude"],
                                              inplace=True)
df_route_busline_cpy["next_sens"].fillna(0, inplace=True)
df_route_busline_cpy["next_ordre"].fillna(1, inplace=True)

df_route_busline_cpy

# %%
json_rep = df_route_busline_cpy.to_dict(orient='record')

with open("datasets_demo/bus_14_route.json", mode="w") as f:
    json.dump(json_rep, f)

# %%
%%sh
ls -alh datasets_demo/bus_14_route.json

# %% [markdown]
# **See the corresponding 3d visualization with Deck.gl in section V.1 at the end of this notebook.**

# %% [markdown]
# ------------------

# %% [markdown]
# -------------------

# %% [markdown]
# # III. Use H3 spatial index for aggregated analytics

# %% [markdown]
# ## III.1. Count busstops groupped by H3 cell

# %%
def counts_by_hexagon(df, res):
    """Aggregates the number of busstops at hexagon level"""

    col_hex_id = "hex_id_{}".format(res)
    col_geometry = "geometry_{}".format(res)

    # within each group preserve the first geometry and count the ids
    df_aggreg = df.groupby(by = col_hex_id).agg({col_geometry: "first",
                                                "latitude": "count"})

    df_aggreg.reset_index(inplace = True)
    df_aggreg.rename(columns={"latitude": "value"}, inplace = True)

    df_aggreg.sort_values(by = "value", ascending = False, inplace = True)
    return df_aggreg

# %%
# demo at resolution 8
df_aggreg_8 = counts_by_hexagon(df = df_stops_to_buslines, res = 8)
print(df_aggreg_8.shape)
df_aggreg_8.head(5)

# %% [markdown]
# ## III.2. Visualization with choropleth map

# %%
def hexagons_dataframe_to_geojson(df_hex, hex_id_field,
                                  geometry_field, value_field,
                                  file_output = None):

    """Produce the GeoJSON representation containing all geometries in a dataframe
     based on a column in geojson format (geometry_field)"""

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(geometry = row[geometry_field],
                          id = row[hex_id_field],
                          properties = {"value": row[value_field]})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    geojson_result = json.dumps(feat_collection)

    # optionally write to file
    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    return geojson_result


# --------------------------------------------------------------------


def choropleth_map(df_aggreg, hex_id_field, geometry_field, value_field,
                   layer_name, initial_map = None, kind = "linear",
                   border_color = 'black', fill_opacity = 0.7,
                   with_legend = False):

    """Plots a choropleth map with folium"""

    if initial_map is None:
        initial_map = base_empty_map()

    # the custom colormap depends on the map kind
    if kind == "linear":
        min_value = df_aggreg[value_field].min()
        max_value = df_aggreg[value_field].max()
        m = round((min_value + max_value) / 2, 0)
        custom_cm = cm.LinearColormap(['green', 'yellow', 'red'],
                                      vmin = min_value,
                                      vmax = max_value)
    elif kind == "outlier":
        # for outliers, values would be -1,0,1
        custom_cm = cm.LinearColormap(['blue', 'white', 'red'],
                                      vmin=-1, vmax=1)
    elif kind == "filled_nulls":
        min_value = df_aggreg[df_aggreg[value_field] > 0][value_field].min()
        max_value = df_aggreg[df_aggreg[value_field] > 0][value_field].max()
        m = round((min_value + max_value) / 2, 0)
        custom_cm = cm.LinearColormap(['silver', 'green', 'yellow', 'red'],
                                      index = [0, min_value, m, max_value],
                                      vmin = min_value,
                                      vmax = max_value)

    # create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_aggreg, hex_id_field,
                                                 geometry_field, value_field)

    # plot on map
    GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': custom_cm(feature['properties']['value']),
            'color': border_color,
            'weight': 1,
            'fillOpacity': fill_opacity
        },
        name = layer_name
    ).add_to(initial_map)

    # add legend (not recommended if multiple layers)
    if with_legend is True:
        custom_cm.add_to(initial_map)

    return initial_map

# %%
m_hex = choropleth_map(df_aggreg = df_aggreg_8,
                       hex_id_field = "hex_id_8",
                       geometry_field = "geometry_8",
                       value_field = "value",
                       layer_name = "Choropleth 8",
                       with_legend = True)
m_hex

# %%
fig, ax = plt.subplots(1, 1, figsize = (15, 10))

im1 = pilim.open('images/4_choroplth_multiresol_8_region.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_axis_off();

# %% [markdown]
# Better yet, plot it 3d with pydeck:

# %%
norm = mpl.colors.Normalize(vmin = df_aggreg_8["value"].min(), 
                            vmax = df_aggreg_8["value"].max())
f2rgb = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.get_cmap('RdYlGn_r'))


def get_color(value):
    return [int(255 * x) for x in f2rgb.to_rgba(value)[:-1]]


get_color(value = 10)

# %%
df_aux = df_aggreg_8.copy()
df_aux["coloring"] = df_aux["value"].apply(lambda x: get_color(value = x))

aggreg_layer = pydeck.Layer(
                    "H3HexagonLayer",
                    data = df_aux,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    extruded=True,
                    get_hexagon="hex_id_8",
                    get_fill_color= "coloring",
                    get_line_color=[255, 255, 255],
                    line_width_min_pixels=1,
                    get_elevation="value",
                    elevation_scale=500,
                    opacity=0.9
                )

view = pydeck.data_utils.compute_view(gdf_dedup[["longitude", "latitude"]])
view.zoom = 6


r = pydeck.Deck(
    layers=[aggreg_layer],
    initial_view_state = view,
    api_keys= {"mapbox": MAPBOX_TOKEN},
    tooltip={"text": "Count: {value}"}
)

r.show()
r.to_html("images/4_choroplth_multiresol_8_region.html")

# %%
fig, ax = plt.subplots(1, 1, figsize = (15, 10))

im1 = pilim.open('images/3d_aggreg_res8.png', 'r')
ax.imshow(np.asarray(im1));
ax.set_axis_off()

# %% [markdown]
# **Aggregate at  coarser and at finer resolutions:**
# 

# %%
# coarser resolutions than 8
df_aggreg_7 = counts_by_hexagon(df = df_stops_to_buslines, res = 7)

# finer resolutions than 8
df_aggreg_9 = counts_by_hexagon(df = df_stops_to_buslines, res = 9)
df_aggreg_10 = counts_by_hexagon(df = df_stops_to_buslines, res = 10)

# %%
# make a dictionary of mappings resolution -> dataframes, for future use
dict_aggreg_hex = {7: df_aggreg_7,
                   8: df_aggreg_8,
                   9: df_aggreg_9,
                   10: df_aggreg_10}

msg_ = "At resolution {} we used {} H3 cells for indexing the bus stops"
for res in dict_aggreg_hex:
    print(msg_.format(res, dict_aggreg_hex[res].shape[0]))


# %%
initial_map = base_empty_map()

for res in dict_aggreg_hex:
    initial_map = choropleth_map(df_aggreg = dict_aggreg_hex[res],
                                 hex_id_field = "hex_id_{}".format(res),
                                 geometry_field = "geometry_{}".format(res),
                                 value_field = "value",
                                 initial_map = initial_map,
                                 layer_name = "Choropleth {}".format(res),
                                 with_legend = False)

folium.map.LayerControl('bottomright', collapsed=True).add_to(initial_map)

initial_map

# %% [markdown]
# First we focus (zoom in) on the city center and display H3 cells covering the same zone at various resolutions:
# 

# %%
fig, ax = plt.subplots(2, 2, figsize=(20, 14))

im1 = pilim.open('images/4_choropleth_multiresol_10.png', 'r')
ax[0][0].imshow(np.asarray(im1))
ax[0][0].set_title("Choropleth resolution 10")
im1 = pilim.open('images/4_choropleth_multiresol_9.png', 'r')
ax[0][1].imshow(np.asarray(im1))
ax[0][1].set_title("Choropleth resolution 9")
im1 = pilim.open('images/4_choropleth_multiresol_8.png', 'r')
ax[1][0].imshow(np.asarray(im1))
ax[1][0].set_title("Choropleth resolution 8")
im1 = pilim.open('images/4_choropleth_multiresol_7.png', 'r')
ax[1][1].imshow(np.asarray(im1))
ax[1][1].set_title("Choropleth resolution 7")

for i in [0, 1]:
    for j in [0, 1]:
        ax[i][j].set_axis_off()
fig.tight_layout()

# %% [markdown]
# Depending on the resolution at which we computed the aggregates, we sometimes got a sparse spatial distribution of H3 cells with busstops.  
# Next we want to include all the H3 cells that cover the city's area and thus put these aggregates in a better perspective.

# %% [markdown]
# ## III.3. Study aggregates in the context of the city's hexagons coverage set

# %%
input_file_subzones = "datasets_demo/subzones_Toulouse.geojson"
gdf_subzones = load_and_prepare_districts(filepath = input_file_subzones) 
 
print(gdf_subzones.shape)
print("\n--------------------------------------------------------\n")
list_subzones = list(gdf_subzones["libcom"].unique())
list_subzones.sort()
print(columnize(list_subzones, displaywidth=100))
print("\n--------------------------------------------------------\n")

gdf_subzones[["libcom", "geometry", 
              "geom_swap", "geom_swap_geojson"]].head()

# %% [markdown]
# There are 37 subzones that form Toulouse metropolitan territory, here we'll focus on the central subzone: 

# %%
# we select the main subzone of the city
selected_subzone = "TOULOUSE"
gdf_subzone_sel = gdf_subzones[gdf_subzones["libcom"] == "TOULOUSE"]
gdf_subzone_sel


# %% [markdown]
# Fill the subzone's geometry with H3 cells (as we've done before with districts, but without compacting this time)

# %%
geom_to_fill = gdf_subzone_sel.iloc[0]["geom_swap_geojson"]

dict_fillings = {}
msg_ = "the subzone was filled with {} hexagons at resolution {}"

for res in [8, 9, 10]:
    # lat/lon in geometry_swap_geojson -> flag_reverse_geojson = False
    df_fill_hex = fill_hexagons(geom_geojson = geom_to_fill,
                                res = res,
                                flag_return_df = True)
    print(msg_.format(df_fill_hex.shape[0], res))

    # add entry in dict_fillings
    dict_fillings[res] = df_fill_hex

# --------------------------
dict_fillings[8].head()

# %% [markdown]
# **Merge (by left outer join) two H3 spatially indexed datasets at the same H3 index resolution**

# %%
dict_filled_aggreg = {}

for res in dict_fillings:
    col_hex_id = "hex_id_{}".format(res)
    df_outer = pd.merge(left = dict_fillings[res][["hex_id", "geometry"]],
                        right = dict_aggreg_hex[res][[col_hex_id, "value"]],
                        left_on = "hex_id",
                        right_on = col_hex_id,
                        how = "left")
    df_outer.drop(columns = [col_hex_id], inplace = True)
    df_outer["value"].fillna(value = 0, inplace = True)

    # add entry to dict
    dict_filled_aggreg[res] = df_outer

# -----------------------------
dict_filled_aggreg[8].sort_values(by="value", ascending=False).head()

# %% [markdown]
# <b>Visualize on map</b>

# %%
res_to_plot = 9
m_filled_aggreg = choropleth_map(df_aggreg = dict_filled_aggreg[res_to_plot],
                                 hex_id_field = "hex_id",
                                 value_field = "value",
                                 geometry_field = "geometry",
                                 initial_map=None,
                                 layer_name = "Polyfill aggreg",
                                 with_legend = True,
                                 kind = "filled_nulls")

m_filled_aggreg

# %%
fig, ax = plt.subplots(1, 2, figsize=(20, 14))

im1 = pilim.open('images/filled_aggreg_merged_res8.png', 'r')
ax[0].imshow(np.asarray(im1))
ax[0].set_title("Polyfill resolution 8")
im1 = pilim.open('images/filled_aggreg_merged_res9.png', 'r')
ax[1].imshow(np.asarray(im1))
ax[1].set_title("Polyfill resolution 9")

ax[0].set_axis_off()
ax[1].set_axis_off()
fig.tight_layout()


# %%
fig, ax = plt.subplots(1, 1, figsize=(14, 14))

im1 = pilim.open('images/filled_aggreg_merged_res10.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("Polyfill resolution 10 - detailed view")
ax.set_axis_off()

# %%
# percentage of cells with value zero at varius index resolutions

msg_ = "Percentage of cells with value zero at resolution {}: {} %"
for res in dict_filled_aggreg:
    df_outer = dict_filled_aggreg[res]
    perc_hexes_zeros = 100 * df_outer[df_outer["value"] == 0].shape[0] / df_outer.shape[0]
    print(msg_.format(res, round(perc_hexes_zeros, 2)))


# %% [markdown]
# **See the corresponding 3d visualization with Deck.gl in section V.2 at the end of this notebook.**

# %%
df_aux = dict_filled_aggreg[9].drop(columns = ["geometry"])
df_aux.to_json("datasets_demo/counts_res9.json", orient = "records", indent = 4)

# %%
!ls -alh datasets_demo/counts_res9.json

# %%
!head -n 20 datasets_demo/counts_res9.json

# %% [markdown]
# ---------------------------------

# %% [markdown]
# ------------------------------

# %% [markdown]
# # IV. Global Spatial Autocorrelation

# %% [markdown]
# ## IV.1 Background

# %% [markdown]
# Global spatial autocorrelation is a measure of the relationship between the values of a variable across space. When a spatial pattern exists, it may be of clustering (positive spatial autocorrelation, similar values are in proximity of each other) or of competition (negative spatial autocorrelation, dissimilarity among neighbors, high values repel other high values).
# 
# **Global Moran's I** is the most commonly used measure of spatial autocorrelation.
# 
# Its formula is usually written as:
# 
# $$I = \frac{N}{\sum_{i}{\sum_{j}{w_{ij}}}} * \frac{ \sum_{i}{\sum_{j}{w_{ij} * (X_i - \bar X) * (X_j - \bar X) }} }{\sum_{i} (X_i - \bar X)^2 }   \tag{1}$$
# 
# and it takes values $I \in [-1,1]$ 
# 
# 
# However, we can replace the variance identified in the formula above, which leads to:
# 
# $$I = \frac{1}{\sum_{i}{\sum_{j}{w_{ij}}}} * \frac{ \sum_{i}{\sum_{j}{w_{ij} * (X_i - \bar X) * (X_j - \bar X) }} }{ \sigma _X ^2 }    \tag{2}$$
# 
# Further on, we can distribute the standard deviation to the factors of the cross-product:
# 
# $$I = \frac{1}{\sum_{i}{\sum_{j}{w_{ij}}}} *  \sum_{i}{\sum_{j}{w_{ij} * \frac{X_i - \bar X}{\sigma _X} * \frac{X_j - \bar X}{\sigma _X} }}    \tag{3}$$
# 
# And finally re-write the formula using z-scores:
# 
# $$I = \frac{1}{\sum_{i}{\sum_{j}{w_{ij}}}} *  \sum_{i}{\sum_{j}{w_{ij} * z_i * z_j }}    \tag{4}$$  
# 
# 
# For our case, weights are computed using Queen contiguity of first order, which means that $w_{ij} = 1$ if geometries i and j touch on their boundary. Weights are usually arranged in a row-standardized (row-stochastic) weights matrix (i.e. sum on each row is 1). While the binary matrix of weights is symmetric, the row-standardized matrix of weights is asymmetric.   
# Applying this row-standardization, we obtain: $\sum_{i}{\sum_{j}{w_{ij}}} = N $
# 
# Formula of Global Moran's I becomes:  
# $$I = \frac{ \sum_{i}{ z_i * \sum_{j}{ w_{ij} * z_j }} }{N} \tag{5}$$
# 
#   
# A first indication about the existance (or absence) of a spatial pattern in the data is obtained by comparing the observed value of I with the expected value of I under the null hypothesis of spatial randomness $\frac{-1}{N-1}$   . 
#   
# <br/>
# 
# Statistical test of global spatial autocorrelation:
# 
# ```
# H0:  complete spatial randomness (values are randomly distributed on the geometries)
# 
# H1 (for the two-tailed test):  global spatial autocorrelation
# H1 (for a one-tailed test):    clustered pattern (resp. dispersed pattern)  
# ```
# 
# The method of choice is Permutation inference, which builds an empirical distribution for Global Moran's I,  randomly reshuffling the data among the geometries (for 999 times in our case).  
# Relative to this distribution, we can assess how likely is to obtain the observed value of Global Moran's I under the null hypothesis.  
# For the computation of the pseudo p-value we can use the empirical CDF, and depending on the H1 use either $1 - ECDF(I_{obs})$ for the right tail or $ECDF(I_{obs})$ for the left tail. The pseudo p-value is compared to the significance level $\alpha$ to decide if we can reject H0.
# 
# 
# Readings:   
# [1] https://www.sciencedirect.com/topics/computer-science/spatial-autocorrelation  
# [2] https://www.insee.fr/en/statistiques/fichier/3635545/imet131-g-chapitre-3.pdf  
# [3] https://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/h-how-spatial-autocorrelation-moran-s-i-spatial-st.htm   

# %% [markdown]
# **Prepare the dataframes with precomputed z-scores and first hollow ring, at various resolutions**

# %%
def prepare_geodataframe_GMI(df_aggreg, num_rings = 2, 
                             flag_debug = False, flag_return_gdf = True):
    """Prepares dataframe for Global Moran's I computation, namely by
       computing z-score and geometry object for each row of the input df_aggreg"""

    df_aux = df_aggreg.copy()

    # get resolution from the hex_id of the first row (assume all the same in df_aggreg)
    res = h3.h3_get_resolution(df_aux.iloc[0]["hex_id"])

    mean_busstops_cell = df_aux["value"].mean()
    stddev_busstops_cell = df_aux["value"].std(ddof = 0)

    if flag_debug is True:
        msg_ = "Average number of busstops per H3 cell at resolution {} : {}"
        print(msg_.format(res, mean_busstops_cell))

    # z_score column
    df_aux["z_score"] = (df_aux["value"] - mean_busstops_cell) / stddev_busstops_cell

    # list of cell ids on hollow rings
    for i in range(1, num_rings + 1):
        df_aux["ring{}".format(i)] = df_aux["hex_id"].apply(lambda x:
                                                            list(h3.hex_ring(h = x,
                                                                             k = i)))

    if flag_return_gdf is True:
        # make shapely geometry objects out of geojson
        df_aux["geometry_shp"] = df_aux["geometry"].apply(
                                              lambda x:
                                              geometry.Polygon(geometry.shape(x)))
        df_aux.rename(columns={"geometry": "geometry_geojson"}, inplace=True)

        geom = df_aux["geometry_shp"]
        df_aux.drop(columns=["geometry_shp"], inplace = True)
        gdf_aux = gpd.GeoDataFrame(df_aux, crs="EPSG:4326", geometry=geom)

        return gdf_aux
    else:
        return df_aux

# %%
dict_prepared_GMI = {}

for res in dict_filled_aggreg:
    gdf_gmi_prepared = prepare_geodataframe_GMI(dict_filled_aggreg[res],
                                                num_rings = 1,
                                                flag_debug = True)
    dict_prepared_GMI[res] = gdf_gmi_prepared

# -----------------------
dict_prepared_GMI[8].head()

# %% [markdown]
# When we look in the Global Moran'I numerator in (5), the sum $\sum_{j}{ w_{ij} * z_j }$ is in fact the spatial lag of cell $i$ .  
# 
# Moran's diagram is a scatterplot that visualizes the relationship between the spatial lag and the z-score of each geometry. The slope of the fitted regression line is quite the value of the Global Moran's I.

# %%
def compute_spatial_lags_using_H3(gdf_prepared, variable_col = "z_score"):
    """Computes spatial lags for an input dataframe which was prepared with method
       prepare_geodataframe_GMI"""

    gdf_aux = gdf_prepared.copy()
    gdf_aux["spatial_lag"] = np.nan

    # for better performance on lookup
    dict_z = dict(zip(gdf_prepared["hex_id"], gdf_prepared[variable_col]))
    dict_ring1 = dict(zip(gdf_prepared["hex_id"], gdf_prepared["ring1"]))

    # in step 2, for each hexagon get its hollow ring 1
    for hex_id in dict_z.keys():
        list_hexes_ring = dict_ring1[hex_id]

        # filter and keep only the hexagons of this ring that have a value in our dataset
        hexes_ring_with_value = [item for item in list_hexes_ring if item in dict_z]
        num_hexes_ring_with_value = len(hexes_ring_with_value)

        # ensure row-standardized weights
        wij_adjusted = 1 / num_hexes_ring_with_value

        if num_hexes_ring_with_value > 0:
            sum_neighbors = sum([dict_z[k] for k in hexes_ring_with_value])
            # spatial lag
            spatial_lag = wij_adjusted * sum_neighbors

            gdf_aux.loc[gdf_aux["hex_id"] == hex_id, "spatial_lag"] = spatial_lag

    return gdf_aux

# %%
gdf_spatial_lags_8 = compute_spatial_lags_using_H3(gdf_prepared = dict_prepared_GMI[8],
                                                   variable_col = "z_score")

gdf_spatial_lags_8.head()

# %% [markdown]
# **The Linear Regression:**

# %%
result = sm_formula.ols(formula = "spatial_lag ~ z_score", 
                        data = gdf_spatial_lags_8).fit()

params = result.params.to_dict()
print(params, "\n")
slope = params["z_score"]
print("Global Moran'I approximated by slope of the regression line:", slope)
print("\n----------------------------------------------------------------\n")

print(result.summary())

# %%
# plot
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
sns.regplot(x = "z_score", y = "spatial_lag", data = gdf_spatial_lags_8, ax = ax)
ax.axhline(0.0)
ax.axvline(0.0)

x_min = math.floor(gdf_spatial_lags_8["z_score"].min())
x_max = math.ceil(gdf_spatial_lags_8["z_score"].max())

ax.set_xlim(x_min, x_max)
ax.set_xlabel("z_score")
ax.set_ylabel("spatially lagged z_score");

# %% [markdown]
# ## IV.2. The PySAL baseline 

# %% [markdown]
# Read docs at: https://splot.readthedocs.io/en/stable/users/tutorials/autocorrelation.html
# 
# Based on our column of geometries (Shapely objects), PySAL will build its own weights matrix.

# %%
help(esda.moran.Moran.__init__)

# %%
def wrapper_over_esda_Global_Moran_I(gdf_prepared, geometry_field, value_field):

    # weights
    wq = pys.weights.Queen.from_dataframe(df = gdf_prepared,
                                          geom_col = "geometry")
    y = gdf_prepared[value_field].values

    # transformation="r" performs row-standardization of weights matrix
    mi = esda.moran.Moran(y = y, w = wq, transformation="r",
                          permutations=999, two_tailed=True)
    return mi

# %%
mi = wrapper_over_esda_Global_Moran_I(gdf_prepared = dict_prepared_GMI[8],
                                      geometry_field = "geometry",
                                      value_field = "value")

print("\nGlobal Moran I:", mi.I, "   p_sim =", mi.p_sim)

# %%
%%capture 
# we used capture to prevent displaying lots of warnings of island geometries, such as:
# ('WARNING: ', 208, ' is an island (no neighbors)')

mi = wrapper_over_esda_Global_Moran_I(gdf_prepared = dict_prepared_GMI[9],
                                      geometry_field = "geometry",
                                      value_field = "value")

# %%
print("\nGlobal Moran I:", mi.I, "   p_sim =", mi.p_sim)

# %%
%%capture 
# we used capture to prevent displaying lots of warnings of island geometries
mi = wrapper_over_esda_Global_Moran_I(gdf_prepared = dict_prepared_GMI[10],
                                      geometry_field = "geometry",
                                      value_field = "value")

# %%
print("\nGlobal Moran I:", mi.I, "   p_sim =", mi.p_sim)

# %% [markdown]
# Interpretation: while at resolution 10, we fail to reject H0 of spatial randomness, at resolution 8 and at resolution 9 we can reject H0 and conclude that there is positive global spatial autocorrelation (clustering) in the dataset.

# %% [markdown]
# ## IV.3. Implementation of Global Moran's I formula from scratch using H3 

# %% [markdown]
# This time we manage the whole computation and use the ring1 column, instead of geometries.

# %%
def compute_Global_Moran_I_using_H3(gdf_prepared, variable_col = "z_score"):
    """Computes Global Moran I for an input dataframe which was prepared with method
       prepare_geodataframe_GMI"""

    S_wijzizj = 0
    S_wij = gdf_prepared.shape[0]

    # for better performance on lookup
    dict_z = dict(zip(gdf_prepared["hex_id"], gdf_prepared[variable_col]))
    dict_ring1 = dict(zip(gdf_prepared["hex_id"], gdf_prepared["ring1"]))

    # now, in step 2, for each hexagon get its hollow ring 1
    for hex_id in dict_z.keys():
        zi = dict_z[hex_id]
        list_hexes_ring = dict_ring1[hex_id]

        # filter and keep only the hexagons of this ring that have a value in our dataset
        hexes_ring_with_value = [item for item in list_hexes_ring if item in dict_z]
        num_hexes_ring_with_value = len(hexes_ring_with_value)

        # ensure row-standardized weights
        wij_adjusted = 1 / num_hexes_ring_with_value

        if num_hexes_ring_with_value > 0:
            # update sum
            sum_neighbors = sum([dict_z[k] for k in hexes_ring_with_value])
            S_wijzizj += wij_adjusted * zi * sum_neighbors

    GMI = S_wijzizj / S_wij
    return GMI


# %%
def reshuffle_and_recompute_GMI(gdf_prepared, variable_col = "z_score",
                                num_permut = 999, I_observed = None,
                                alpha = 0.005, alternative = "greater",
                                flag_plot = True, flag_verdict = True):
    """Permutation inference with number of permutations given by num_permut and
       pseudo significance level specified by alpha"""

    gdf_aggreg_reshuff = gdf_prepared.copy()
    list_reshuff_I = []

    for i in range(num_permut):
        # simulate by reshuffling column
        gdf_aggreg_reshuff[variable_col] = np.random.permutation(
                                             gdf_aggreg_reshuff[variable_col].values)

        I_reshuff = compute_Global_Moran_I_using_H3(gdf_prepared = gdf_aggreg_reshuff)
        list_reshuff_I.append(I_reshuff)

    # for hypothesis testing
    list_reshuff_I.append(I_observed)

    # empirical CDF
    ecdf_GMI = sm.distributions.empirical_distribution.ECDF(list_reshuff_I, side = "left")

    percentile_observedI = stats.percentileofscore(list_reshuff_I,
                                                   I_observed,
                                                   kind='strict')
    # note: use decimal to avoid 99.9 / 100 = 0.9990000000000001
    percentile_observedI_ = float(str(decimal.Decimal(str(percentile_observedI)) / 100))

    try:
        assert(ecdf_GMI(I_observed) == percentile_observedI_)
    except Exception:
        pass
        # print(ecdf_GMI(I_observed), " vs ", percentile_observedI_)

    msg_reject_H0 = "P_sim = {:3f} , we can reject H0"
    msg_failtoreject_H0 = "P_sim = {:3f} , we fail to reject H0 under alternative {}"
        
    if alternative == "greater":
        pseudo_p_value = 1 - ecdf_GMI(I_observed)
        if flag_verdict is True:
            if pseudo_p_value < alpha:
                print(msg_reject_H0.format(pseudo_p_value))
            else:
                print(msg_failtoreject_H0.format(pseudo_p_value, alternative))
    elif alternative == "less":
        pseudo_p_value = ecdf_GMI(I_observed)
        if flag_verdict is True:
            if pseudo_p_value < alpha:
                print(msg_reject_H0.format(pseudo_p_value))
            else:
                print(msg_failtoreject_H0.format(pseudo_p_value, alternative))
    elif alternative == "two-tailed":
        pseudo_p_value_greater = 1 - ecdf_GMI(I_observed)
        pseudo_p_value_less = ecdf_GMI(I_observed)
        pseudo_p_value = min(pseudo_p_value_greater, pseudo_p_value_less)
        
        if flag_verdict is True:
            if (pseudo_p_value_greater < alpha/2):
                print(msg_reject_H0.format(pseudo_p_value_greater))
            elif (pseudo_p_value_less < alpha/2):
                print(msg_reject_H0.format(pseudo_p_value_less))
            else:
                pseudo_p_value = min(pseudo_p_value_greater, pseudo_p_value_less)
                print(msg_failtoreject_H0.format(pseudo_p_value, alternative))
        
    if flag_plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(20, 7),
                               gridspec_kw={'width_ratios': [2, 3]})
        gdf_prepared.plot(column=variable_col, cmap= "viridis", ax=ax[0], legend=False)

        ax[1].hist(list_reshuff_I, density=True, bins=50)
        ax[1].axvline(I_observed, color = 'red', linestyle = '--', linewidth = 3)
        fig.tight_layout()
        
    return pseudo_p_value 

# %% [markdown]
# **Compute at various index resolutions**

# %%
%%time

I_8 = compute_Global_Moran_I_using_H3(gdf_prepared = dict_prepared_GMI[8])
print("I =", I_8)

# %%
%%time
p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[8],   
                                    num_permut = 999,                            
                                    I_observed = I_8,
                                    alternative = "two-tailed",
                                    flag_plot = True)

# %%
%%time
p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[8],   
                                    num_permut = 999,                            
                                    I_observed = I_8,
                                    alternative = "greater",
                                    flag_plot = False)

# %%
%%time
I_9 = compute_Global_Moran_I_using_H3(gdf_prepared = dict_prepared_GMI[9])
print("I =",I_9)

# %%
%%time
p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[9], 
                                    num_permut = 999,                            
                                    I_observed = I_9,
                                    alternative = "two-tailed",
                                    flag_plot = True)

# %%
%%time
p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[9], 
                                    num_permut = 999,                            
                                    I_observed = I_9,
                                    alternative = "greater",
                                    flag_plot = False)

# %%
%%time
I_10 = compute_Global_Moran_I_using_H3(gdf_prepared = dict_prepared_GMI[10])

print("I =",I_10)

# %%
%%time
p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[10],  
                                    num_permut = 999,                            
                                    I_observed = I_10,
                                    alternative = "two-tailed",
                                    flag_plot = True)

# %%
%%time
p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[10],  
                                    num_permut = 999,                            
                                    I_observed = I_10,
                                    alternative = "less",
                                    flag_plot = False)

# %% [markdown]
# ## IV.4. Spatial Autocorrelation Prediction with Tensorflow

# %% [markdown]
# We build a Convolutional Neural Network with Tensorflow, able to classify an input spatial distribution of points (over the central subzone of Toulouse), bucketed into H3 cells at resolution 9 and converted to a matrix using H3 IJ coordinates system, into one of the following 2 classes: 
#  * complete spatial randomness
#  * global spatial autocorrelation (clustered)
#  
# Note: the IJ coordinate system was overviewed in the preliminaries section I.3. of this notebook.

# %% [markdown]
# Having chosen to prototype for resolution 9 of the H3 index, let's first see the matrix size corresponding to the central subzone of Toulouse: 

# %%
df_test = dict_prepared_GMI[9][["hex_id", "z_score"]]
df_test.head()

# %% [markdown]
# ### IV.4.1. Dataframe to matrix

# %%
def df_to_matrix(df):
    
    """Given a dataframe with columns hex_id and value, with the set of all rows' hex_id 
       covering the geometry under study (a district, a subzone, any custom polygon),
       create the marix with values in ij coordinate system"""

    # take first row's hex_id as local origin
    # (it doesn't matter this choice, as we'll post-process the resulted ij)
    dict_ij = {}
    dict_values = {}

    local_origin = df.iloc[0]["hex_id"]

    for i, row in df.iterrows():
        ij_ex = h3.experimental_h3_to_local_ij(origin = local_origin,
                                               h = row["hex_id"])
        dict_ij[row["hex_id"]] = ij_ex
        dict_values[row["hex_id"]] = row["z_score"]

    # post-process
    min_i = min([dict_ij[h][0] for h in dict_ij])
    min_j = min([dict_ij[h][1] for h in dict_ij])

    max_i = max([dict_ij[h][0] for h in dict_ij])
    max_j = max([dict_ij[h][1] for h in dict_ij])

    # rescale
    dict_ij_rescaled = {}
    for h in dict_ij:
        dict_ij_rescaled[h] = [dict_ij[h][0] - min_i, dict_ij[h][1] - min_j]

    num_rows = max_i - min_i + 1
    num_cols = max_j - min_j + 1

    arr_ij = np.zeros(shape=(num_rows, num_cols), dtype = np.float32)

    for h in dict_ij_rescaled:
        arr_ij[dict_ij_rescaled[h][0]][dict_ij_rescaled[h][1]] = dict_values[h]

    return arr_ij

# %%
arr_ij_busstops = df_to_matrix(df = df_test)
print(arr_ij_busstops.shape)



# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(arr_ij_busstops, cmap='coolwarm', interpolation = None)
ax.set_axis_off()
fig.savefig("images/matrix_city_busstops.png");

# %% [markdown]
# ### IV.4.2. Generate dataset for training:

# %% [markdown]
# For this, we'll use PySAL's Pointpats library:

# %%
help(pp.PoissonPointProcess.__init__)

# %%
help(pp.PoissonClusterPointProcess.__init__)

# %%
# create spatial window for generating points
geom_subzone = gdf_subzone_sel["geometry"].values[0]
xs = geom_subzone.exterior.coords.xy[0]
ys = geom_subzone.exterior.coords.xy[1]
vertices = [(xs[i], ys[i]) for i in range(len(xs))]
print(vertices[0:10])
print(" ------------------------------------------------------------------- ")

window = pp.Window(vertices)
print("Window's bbox:", window.bbox)

# %%
# demo a CSR and a clustered point pattern generated with PySAL
np.random.seed(13)
num_points_to_gen = 500
num_parents = 50

samples_csr = pp.PoissonPointProcess(window = window, 
                                     n = num_points_to_gen, 
                                     samples = 1, 
                                     conditioning = False,
                                     asPP = False)
pp_csr = pp.PointPattern(samples_csr.realizations[0])
print(samples_csr.realizations[0][0:3], "\n")
df_csr = pd.DataFrame(samples_csr.realizations[0], 
                      columns= ["longitude", "latitude"])
print(df_csr.head(3))
print(" ----------------------------------------------------- ")

samples_clustered = pp.PoissonClusterPointProcess(window = window, 
                                                  n = num_points_to_gen, 
                                                  parents = num_parents, 
                                                  radius = 0.01, 
                                                  samples = 1, 
                                                  asPP = False, 
                                                  conditioning = False)
pp_clustered = pp.PointPattern(samples_clustered.realizations[0])
print(samples_clustered.realizations[0][0:3], "\n")
df_clustered = pd.DataFrame(samples_clustered.realizations[0], 
                            columns= ["longitude", "latitude"])
print(df_clustered.head(3))
# --------------------------------------------------------

# %%
fig, ax = plt.subplots(1, 2, figsize = (20, 10))

ax[0].fill(xs, ys, alpha=0.1, fc='r', ec='none')
ax[0].scatter(df_csr[["longitude"]], df_csr[["latitude"]], 
              fc = "blue", marker=".", s = 35)
ax[0].set_title("Random")

ax[1].fill(xs, ys, alpha=0.1, fc='r', ec='none')
ax[1].scatter(df_clustered[["longitude"]], df_clustered[["latitude"]], 
              fc = "blue", marker=".", s = 35)
ax[1].set_title("Clustered");

# %% [markdown]
# Note: varying the radius of the clustered point process we get various degrees of spatial clustering.
# 
# What we have to do next is to index the realizations into H3 at resolution 9, outer join with the set of cells of resolution 9 covering this city subzone and compute Global Moran's I.

# %%
def generate_training_sample(window, flag_clustered = True, 
                             num_points_to_generate = 500,
                             num_parents = 50, radius_offsprings = 0.01):
    
    # generate points in space
    if flag_clustered is True:
        samples_generated = pp.PoissonClusterPointProcess(
                                 window = window, 
                                 n = num_points_to_generate, 
                                 parents = num_parents, 
                                 radius = radius_offsprings, 
                                 samples = 1, 
                                 asPP = False, 
                                 conditioning = False)
    else:
        samples_generated = pp.PoissonPointProcess(
                                 window = window, 
                                 n = num_points_to_generate, 
                                 samples = 1, 
                                 conditioning = False,
                                 asPP = False)
        
    # make dataframe with their lon/lat
    df_generated = pd.DataFrame(samples_generated.realizations[0], 
                                columns= ["longitude", "latitude"])
    # index in H3
    df_generated["hex_id_9"] = df_generated.apply(
                                  lambda row: h3.geo_to_h3(
                                              lat = row["latitude"],
                                              lng = row["longitude"],
                                              resolution = 9),
                                  axis = 1)
    
    # counts groupped by cell
    df_aggreg = df_generated.groupby(by = "hex_id_9").agg({"latitude": "count"})
    df_aggreg.reset_index(inplace = True)
    df_aggreg.rename(columns={"latitude": "value"}, inplace = True)
    
    # outer join with set of cells covering the city's subzone
    df_outer = pd.merge(left = dict_fillings[9][["hex_id", "geometry"]],
                        right = df_aggreg[["hex_id_9", "value"]],
                        left_on = "hex_id",
                        right_on = "hex_id_9",
                        how = "left")
    df_outer.drop(columns = ["hex_id_9"], inplace = True)
    df_outer["value"].fillna(value = 0, inplace = True)
    
    # compute Global Moran's I
    df_GMI_prepared = prepare_geodataframe_GMI(df_outer,
                                               num_rings = 1,
                                               flag_debug = False,
                                               flag_return_gdf = False)
    
    I_9 = compute_Global_Moran_I_using_H3(gdf_prepared = df_GMI_prepared)
    
    # assert the hypothesis testing is consistent with the manner we generated points
    p_sim = reshuffle_and_recompute_GMI(gdf_prepared = df_GMI_prepared, 
                                        num_permut = 999,                            
                                        I_observed = I_9,
                                        alternative = "two-tailed",
                                        flag_plot = False, flag_verdict = False)
    
    result_valid = True
    alpha = 0.005
    if (p_sim > alpha) and (flag_clustered is True):
        msg_ = "Failed to produce clustered point pattern with params {},{},{} (failed to reject H0)" 
        print(msg_.format(num_points_to_generate, num_parents, radius_offsprings))
        result_valid = False
    elif (p_sim < alpha) and (flag_clustered is False):
        print("Failed to produce random point pattern (H0 was rejected)")
        result_valid = False
    
    # create matrix
    arr_ij = df_to_matrix(df = df_GMI_prepared)
    
    if result_valid is True:
        # return the matrix and the computed Moran's I
        return arr_ij, I_9, samples_generated.realizations[0]
    else:
        return None, None, None
    

# %% [markdown]
# Distributions from which to draw the num_points_to_generate,num_parents, radius_offsprings

# %%
# sidenote: how it works
list_multiples_of_100 = [random.randrange(100, 1000, 100) for _ in range(50)]
print(Counter(list_multiples_of_100))

list_choice = [random.choice([0.01, 0.02, 0.03, 0.05]) for _ in range(50)]
print(Counter(list_choice))

# %% [markdown]
# **Generate a small batch of samples with randomly distributed points**

# %%
%%time
arr_matrices = None
arr_GMI = np.array([])
arr_labels = np.array([])
arr_points = []

k = 0
while k < 10:
    arr_ij, GMI, points = generate_training_sample(
                            window = window,
                            flag_clustered = False, 
                            num_points_to_generate = random.randrange(100, 1000, 100),
                            num_parents = None, 
                            radius_offsprings = None)
    if GMI is not None:
        if arr_matrices is None:
            arr_matrices = np.array([arr_ij])
        else:
            arr_matrices = np.vstack((arr_matrices, [arr_ij]))
        arr_GMI = np.append(arr_GMI, GMI)
        arr_labels = np.append(arr_labels, 0)
        arr_points.append(points)
        k = k + 1

np.save("datasets_demo/smallbatch.npy", arr_matrices)

# %%
fig, ax = plt.subplots(3, 3, figsize = (15, 15))

arr_restored = np.load("datasets_demo/smallbatch.npy")

for k1 in range(3):
    for k2 in range(3):
        arr_ij = arr_restored[3 * k1 + k2]
        GMI = arr_GMI[3 * k1 + k2]
        ax[k1][k2].imshow(arr_ij, cmap='coolwarm', interpolation = None)
        ax[k1][k2].set_title("GMI = {}".format(round(GMI, 3)))

# %% [markdown]
# **Generate a small batch of samples with spatially clustered points:**

# %%
%%time
arr_matrices = None
arr_GMI = np.array([])
arr_labels = np.array([])
arr_points = []

k = 0
while k < 10:
    arr_ij, GMI, points  = generate_training_sample(
                            window = window,
                            flag_clustered = True, 
                            num_points_to_generate = random.randrange(100, 1000, 100),
                            num_parents = random.randrange(10, 100, 10), 
                            radius_offsprings = random.choice([0.01, 0.02, 0.03, 0.05]))
    if GMI is not None:
        if arr_matrices is None:
            arr_matrices = np.array([arr_ij])
        else:
            arr_matrices = np.vstack((arr_matrices, [arr_ij]))
        arr_GMI = np.append(arr_GMI, GMI)
        arr_labels = np.append(arr_labels, 0)
        arr_points.append(points)
        k = k + 1

np.save("datasets_demo/smallbatch2.npy", arr_matrices)

# %%
fig, ax = plt.subplots(3, 3, figsize = (15, 15))

arr_restored = np.load("datasets_demo/smallbatch2.npy")

for k1 in range(3):
    for k2 in range(3):
        arr_ij = arr_restored[3 * k1 + k2]
        GMI = arr_GMI[3 * k1 + k2]
        ax[k1][k2].imshow(arr_ij, cmap='coolwarm', interpolation = None)
        ax[k1][k2].set_title("GMI = {}".format(round(GMI, 3)))

# %% [markdown]
# Correspond to the following clustered point process realizations:

# %%
fig, ax = plt.subplots(3, 3, figsize = (15, 15))

for k1 in range(3):
    for k2 in range(3):
        points = arr_points[3 * k1 + k2]
        GMI = arr_GMI[3 * k1 + k2]
        # the shape of the subzone in pale pink
        ax[k1][k2].fill(xs, ys, alpha=0.1, fc='r', ec='none')
        # the points generated
        df_clust = pd.DataFrame(points, 
                                columns= ["longitude", "latitude"])
        ax[k1][k2].scatter(
              df_clust[["longitude"]], df_clust[["latitude"]], 
              fc = "blue", marker=".", s = 35)
        ax[k1][k2].set_title("GMI = {}".format(round(GMI, 3)))


# %% [markdown]
# **Generate the actual training set:**

# %%
%%capture
arr_matrices = None
arr_GMI = np.array([])
arr_labels = np.array([])
list_points = []

k = 0
while k < 600:
    arr_ij, GMI, points = generate_training_sample(
                            window = window,
                            flag_clustered = False, 
                            num_points_to_generate = random.randrange(100, 1000, 100),
                            num_parents = None, 
                            radius_offsprings = None)
    if GMI is not None:
        if arr_matrices is None:
            arr_matrices = np.array([arr_ij])
        else:
            arr_matrices = np.vstack((arr_matrices, [arr_ij]))
        arr_GMI = np.append(arr_GMI, GMI)
        arr_labels = np.append(arr_labels, 0)
        arr_points = np.concatenate((arr_points,))
        list_points.append(points)
        k = k + 1

np.save("datasets_demo/csr_matrices.npy", arr_matrices)
np.save("datasets_demo/csr_GMI.npy", arr_GMI)

arr_points = np.array(list_points)
np.save("datasets_demo/csr_points.npy", arr_points)

# %%
%%capture

arr_matrices = None
arr_GMI = np.array([])
arr_labels = np.array([])
list_points = []

k = 0
while k < 600:
    arr_ij, GMI, points = generate_training_sample(
                            window = window,
                            flag_clustered = True, 
                            num_points_to_generate = random.randrange(100, 1000, 100),
                            num_parents = random.randrange(10, 100, 10), 
                            radius_offsprings = random.choice([0.01, 0.02, 0.03, 0.05]))
    if GMI is not None:
        if arr_matrices is None:
            arr_matrices = np.array([arr_ij])
        else:
            arr_matrices = np.vstack((arr_matrices, [arr_ij]))
        arr_GMI = np.append(arr_GMI, GMI)
        arr_labels = np.append(arr_labels, 0)
        list_points.append(points)
        k = k + 1

np.save("datasets_demo/clustered_matrices.npy", arr_matrices)
np.save("datasets_demo/clustered_GMI.npy", arr_GMI)

arr_points = np.array(list_points)
np.save("datasets_demo/clustered_points.npy", arr_points)

# %%
!rm datasets_demo/a.npy
!ls -al datasets_demo/*.npy

# %% [markdown]
# ### IV.4.3 Prepare Tensorflow Dataset:

# %%
help(tf.data.Dataset.from_tensor_slices)

# %%
def prepare_datasets():
    
    batch_size = 4
    
    arr_ij_csr = np.load("datasets_demo/csr_matrices.npy")
    arr_ij_clustered = np.load("datasets_demo/clustered_matrices.npy")
    arr_ij_combined = np.concatenate((arr_ij_csr, arr_ij_clustered), axis = 0)
    assert(arr_ij_combined.shape[0] == (arr_ij_csr.shape[0] + arr_ij_clustered.shape[0]))
            
    #labels are 0 (for csr) and 1 (for clustered)
    labels_csr = np.zeros(arr_ij_csr.shape[0])
    labels_clustered = np.ones(arr_ij_clustered.shape[0])
    labels_combined = np.concatenate((labels_csr, labels_clustered), axis = 0)
    assert(labels_combined.shape[0] == arr_ij_combined.shape[0])

    with tf.device('/cpu:0'):
        dataset_matrices = tf.data.Dataset.from_tensor_slices(arr_ij_combined)
        dataset_labels = tf.data.Dataset.from_tensor_slices(labels_combined)
        
        dataset = tf.data.Dataset.zip((dataset_matrices, dataset_labels))
        dataset = dataset.shuffle(buffer_size=2000)
        print(dataset)
        print(" ------------------------------------------------------------ ")

        train_dataset = dataset.take(1000)
        validation_dataset = dataset.skip(1000)

        # we need repeat() otherwise it will later complain that:
        # tensorflow:Your input ran out of data; interrupting training.
        train_dataset = train_dataset.repeat().batch(batch_size)
        validation_dataset = validation_dataset.repeat().batch(batch_size)

        train_dataset = train_dataset.prefetch(1)
        validation_dataset = validation_dataset.prefetch(1)

        print(train_dataset)
        print(validation_dataset)
        
        return train_dataset, validation_dataset

# %%
train_dataset, validation_dataset = prepare_datasets()

# %%
# get a batch of samples

# note: make_one_shot_iterator was deprecated in tf v2
iterator = train_dataset.__iter__() 
x_batch = next(iterator)

print(type(x_batch[0]), x_batch[0].dtype, x_batch[0].shape)
print(type(x_batch[1]), x_batch[1].dtype, x_batch[1].shape)

# %%
batch_size = 4
nr = batch_size // 2

fig = plt.figure(figsize = (8, 8))

for i in range(0, nr * nr):
    ax = fig.add_subplot(nr, nr, i+1)
    image = x_batch[0][i]
    if i == 0:
        print(image.shape)
    ax.imshow(image, cmap="coolwarm", interpolation = None)
    ax.set_title(str(x_batch[1][i].numpy()))
    ax.set_axis_off()

fig.tight_layout()

# %% [markdown]
# ### IV.4.4. Build the Tensorflow model:

# %% [markdown]
# The first convolution layer has a specified kernel and is not trainable, its role being of computing the spatial lag 

# %%
def get_fixed_kernel(shape = (3, 3, 1, 1), dtype=np.float32):
    kernel_fixed = np.array([[1/6, 1/6, 0],
                            [1/6, 1, 1/6],
                            [0, 1/6, 1/6]])
    kernel = np.zeros(shape)
    kernel[:, :, 0, 0] = kernel_fixed
    return kernel


get_fixed_kernel()

# %%
help(layers.Conv2D.__init__)

# %%
help(layers.Dense.__init__)

# %%
def build_classifier():

    # build a sequential model using the functional API
    tf.keras.backend.clear_session()

    # theuse input shape (None,None,1) to allow variable size inputs; 1 channel
    model_inputs = tf.keras.Input(shape=(48, 52, 1), name = "ClassifInput")

    # first is a hexagonal convolution with the specified kernel (non-trainable)
    conv1 = layers.Conv2D(filters = 1, 
                          kernel_size = [3, 3], 
                          kernel_initializer = get_fixed_kernel, 
                          input_shape = (None, None, 1), 
                          padding = "valid",                       # no padding
                          use_bias = False,
                          activation='relu',
                          name='HexConv')
    conv1.trainable = False
    x = conv1(model_inputs)

    # other usual convolutional layers layer
    x = layers.Convolution2D(128, 3, 3, activation='relu', name='Conv2')(x)
    x = layers.Convolution2D(64, 3, 3, activation='relu', name='Conv3')(x)

    # here use GlobalAveragePooling2D; we cannot use Flatten because we have no fix inputshape
    x = layers.GlobalAveragePooling2D(data_format='channels_last', name='GlobalAvgPool')(x)

    x = layers.Dense(16, activation='relu', name='Dense1')(x)

    # the output for binary classifier
    model_outputs = layers.Dense(2, activation='softmax', name = "ClassifOutput")(x)

    model = tf.keras.Model(inputs = model_inputs, 
                           outputs = model_outputs, 
                           name="global_spatial_assoc_classifier")

    model.compile(loss = "sparse_categorical_crossentropy",
                  optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001),
                  metrics = ["accuracy"])
    
    return model


# %%
model_classif = build_classifier()
model_classif.summary()

# %% [markdown]
# **Automatically generate diagram of the Tensorflow model in LaTex:**

# %%
!rm -r -f latex_files
!mkdir -p latex_files

# %%
# sidenote: how it works
print(to_head( '../PlotNeuralNet' ))
print("-----------------------------------------")
print(to_cor())
print("-----------------------------------------")
print(to_begin())
print("-----------------------------------------")
print(to_end())

# %%
list_info_layers = []


# note: every path should be relative to folder latex_files
arch = [
    to_head( '../PlotNeuralNet' ),
    """\\usepackage{geometry}
       \\geometry{
            paperwidth=6cm,
            paperheight=4cm,
            margin=0.5cm
        }
    """,
    to_cor(),
    to_begin()
]

last_lay = None
prev_lay_pos = "(0,0,0)"


for lay in list(model_classif.layers):
    list_info_layers.append((lay.name, type(lay), 
                             lay.input_shape, lay.output_shape))
    
    #for the latex diagram    
    # where to position the current layer in the diagram
    if last_lay is not None:
        output_dim = lay.output_shape
        if last_lay != "ClassifInput":
            prev_lay_pos = "({}-east)".format(last_lay)  
        else:
            prev_lay_pos = "(0,0,0)"
    else:
        output_dim = lay.output_shape[0]
        prev_lay_pos = "(-1,0,0)"
    print(str(type(lay)).ljust(50), output_dim)

    if isinstance(lay, layers.InputLayer) is True:
        arch.append(to_input(name = lay.name, 
                             pathfile = '../images/matrix_city_busstops.png', 
                             to = prev_lay_pos, 
                             width = 14, height = 14)
                   )
    
    elif isinstance(lay, layers.Conv2D) is True:
        size_kernel = lay.kernel_size[0]
        num_filters = lay.filters
            
        arch.append(to_Conv(name = lay.name,
                            s_filer = output_dim[2], 
                            n_filer = num_filters, 
                            offset = "(1,1,2)", 
                            to = prev_lay_pos,
                            depth = output_dim[1], height = output_dim[2], 
                            width = num_filters // 4,   # divide by 4 for design
                            caption=lay.name)
                   )
        
    elif isinstance(lay, layers.GlobalAveragePooling2D) is True:
        arch.append(to_Pool(name = lay.name, 
                            offset="(1,1,3)", 
                            to = prev_lay_pos, 
                            depth = 1, height = 1, 
                            width = output_dim[1]// 4,   # divide by 4 for design
                            caption = lay.name)
                   )
        
    elif isinstance(lay, layers.Dense) is True:
        num_units = lay.units
        
        arch.append(to_SoftMax(name = lay.name, 
                               s_filer = num_units,
                               offset="(2,1,3)", 
                               to = prev_lay_pos,
                               depth = output_dim[1], height = 1, 
                               width = 1, 
                               caption=lay.name)
                   )
        
    #prepare for next  
    last_lay = lay.name

arch.append(to_end())

pd.DataFrame(list_info_layers, columns = ["name", "type",
                                          "input_shape", "output_shape"])

# %%
%%capture
to_generate(arch, "latex_files/demovis_nn.tex")

# %%
%%sh
cd latex_files
pdflatex demovis_nn.tex  >/dev/null 2>&1

# %%
!ls -alh latex_files

# %%
fig, ax = plt.subplots(1, 1, figsize=(14, 14))

im1 = pilim.open('images/cnn_arch.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("Diagram generated above (LaTex) for the architecture of our CNN classifier")
ax.set_axis_off()

# %% [markdown]
# <br/><br/>

# %% [markdown]
# ### IV.4.5 Train the model

# %%
batch_size = 4
num_iter_per_epoch_train = 1000//batch_size
num_iter_per_epoch_valid = 200//batch_size

print("Iterations per epoch: training {}  validation {}".format(
                                             num_iter_per_epoch_train, 
                                             num_iter_per_epoch_valid))
print("Num_batches samples trained on per epoch = ", 
      batch_size * num_iter_per_epoch_train)

# %%
!rm -r -f tf_models/checkpoint_classif
!mkdir -p tf_models/checkpoint_classif

# %%
checkpoint_filepath = 'tf_models/checkpoint_classif'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath,
                                save_weights_only=False,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True,
                                verbose=0)


# custom callback for printing metrics only on certain epochs 
class SelectiveProgress(tf.keras.callbacks.Callback):
    """ inspired by tfdocs.EpochDots """

    def __init__(self, report_every=10):
        self.report_every = report_every

    def on_epoch_end(self, epoch, logs):
        if epoch % self.report_every == 0:
            print('Epoch: {:d}, '.format(epoch), end='')
            for name, value in sorted(logs.items()):
                print('{}:{:0.4f}'.format(name, value), end=',  ')
            print()


# %%
history = model_classif.fit(x = train_dataset, 
                            steps_per_epoch = num_iter_per_epoch_train,
                            validation_data = validation_dataset, 
                            validation_steps = num_iter_per_epoch_valid,
                            epochs = 50, 
                            shuffle = False, 
                            workers = 1,
                            verbose=0,
                            callbacks = [checkpoint_callback, 
                                         SelectiveProgress()])

# %%
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# %%
print(model_classif.output_names)

fig, ax = plt.subplots(1, 1, figsize = (15, 7))
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.set_title('model accuracy')
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.legend(['train', 'val'], loc='upper left')

# %%
!ls -alh tf_models/checkpoint_classif

# %%
#sidenote: we can verify that the hexagonal convolution preserved the kernel specified by us

print(model_classif.get_layer("HexConv").get_weights())

# %% [markdown]
# <br/>

# %% [markdown]
# ### IV.4.6. Load the best iteration model and make predictions

# %%
loaded_classifier = tf.keras.models.load_model("tf_models/checkpoint_classif")

# ------------------- 

print(list(loaded_classifier.signatures.keys())) 

infer = loaded_classifier.signatures["serving_default"]
print(infer.structured_outputs)


# %%
def predicted_label(arr_ij):   
    
    # reshape input from (m,n) to (1,m,n)
    reshaped_input = arr_ij[np.newaxis, :, :]
    
    #the result from the binary classifier
    prediction_logits = loaded_classifier.predict([reshaped_input])
    top_prediction = tf.argmax(prediction_logits, 1)
    the_label = top_prediction.numpy()[0]    
   
    return prediction_logits[0], the_label

# %%
pred_logits, the_label  = predicted_label(arr_ij_busstops)

dict_decode = {0: "CSR", 1: "Clustered"}

print(pred_logits, 
      " ---> PREDICTED CLASS:", the_label, 
      " ---> DECODED AS: ", dict_decode[the_label])


# %% [markdown]
# **Confusion matrix and confused samples**

# %%
# here N = CSR, P = CLUSTERED

TP = 0
TN = 0
FP = 0
FN = 0

list_misclassified = []
list_misclassified_realizations = []

arr_ij_csr = np.load("datasets_demo/csr_matrices.npy")
points_csr = np.load("datasets_demo/csr_points.npy", allow_pickle = True)
 
print(arr_ij_csr.shape)
for k in range(arr_ij_csr.shape[0]):
    sample_csr = arr_ij_csr[k]
    pred_logits, the_label = predicted_label(sample_csr)

    if the_label == 0:
        TN += 1
    else:
        FP += 1
        list_misclassified.append((0, arr_ij_csr[k], pred_logits))
        list_misclassified_realizations.append((0, points_csr[k], pred_logits))

arr_ij_clustered = np.load("datasets_demo/clustered_matrices.npy")
points_clustered = np.load("datasets_demo/clustered_points.npy", allow_pickle = True)

print(arr_ij_clustered.shape)
for k in range(arr_ij_clustered.shape[0]):
    sample_clustered = arr_ij_clustered[k]
    pred_logits, the_label = predicted_label(sample_clustered)

    if the_label == 1:
        TP += 1
    else:
        FN += 1  
        list_misclassified.append((1, arr_ij_csr[k], pred_logits))
        list_misclassified_realizations.append((1, points_clustered[k], pred_logits))

confusion_matrix = [[TN, FP], [FN, TP]]

assert(len(list_misclassified) == (FP + FN))

ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu")

# %% [markdown]
# **Confused samples:**

# %%
nr = min(8, len(list_misclassified))

fig = plt.figure(figsize = (24, 12))

for i in range(0, nr):
    ax = fig.add_subplot(2, 4, i+1)
    
    sample_chosen = list_misclassified[i]
    image = sample_chosen[1]

    ax.imshow(image, cmap="coolwarm", interpolation = None)
    ax.set_title("Real: {} / Logits {}".format(sample_chosen[0], sample_chosen[2]))
    ax.set_axis_off()

fig.tight_layout()

# %% [markdown]
# The point processes realizations of these were:

# %%
nr = min(8, len(list_misclassified))

fig = plt.figure(figsize = (24, 12))

for i in range(0, nr):
    ax = fig.add_subplot(2, 4, i+1)
    
    sample_chosen = list_misclassified_realizations[i]

    realizations = sample_chosen[1]
    df_points = pd.DataFrame(realizations, 
                             columns= ["longitude", "latitude"])

    # the shape of the subzone in pale pink
    ax.fill(xs, ys, alpha=0.1, fc='r', ec='none')
    
    ax.scatter(
      df_points[["longitude"]], df_points[["latitude"]], 
      fc = "blue", marker=".", s = 35)
    
    ax.set_title("Real: {} / Logits {}".format(sample_chosen[0], sample_chosen[2]))
    ax.set_axis_off()

fig.tight_layout()

# %% [markdown]
# ### IV.4.7. Attemp to find similar point process realizations using embeddings
# 
# Based on the embeddings extracted from the trained CNN, we seek to retrieve training instances which are most similar to a given query pattern.

# %% [markdown]
# Extract embeddings at a specified layer of the CNN:

# %%
list_layers = loaded_classifier.layers
assert(list_layers[-1].name == list(infer.structured_outputs.keys())[0])
print(list_layers[-2].name)

# %%
embeddings_extractor = tf.keras.Model(loaded_classifier.inputs,
                                      loaded_classifier.get_layer(list_layers[-2].name).output)
embeddings_extractor.summary()

# %%
reshaped_input = arr_ij_busstops[np.newaxis, :, :]

embedd_vector_busstops = embeddings_extractor.predict([reshaped_input])
embedd_vector_busstops[0]

# %% [markdown]
# Extract embeddings of all training samples and build an index of them in Annoy.
# 
# Annoy is an open source project by Spotify, aimed at fast nearest neighbor search (see https://github.com/spotify/annoy#readme)

# %%
%%time

embedd_length = 16
annoy_idx = AnnoyIndex(embedd_length, 'angular') 

for k in range(arr_ij_csr.shape[0]):
    sample_csr = arr_ij_csr[k]
    reshaped_input = sample_csr[np.newaxis, :, :]
    embedd_vect = embeddings_extractor.predict([reshaped_input])
    
    # note: ids must be integers in seq starting from 0 
    annoy_idx.add_item(k, embedd_vect[0])
    

for k2 in range(arr_ij_clustered.shape[0]):
    sample_clustered = arr_ij_clustered[k2]
    reshaped_input = sample_clustered[np.newaxis, :, :]
    embedd_vect = embeddings_extractor.predict([reshaped_input])
    
    # note: ids must be integers in seq starting from 0 
    annoy_idx.add_item(k + k2, embedd_vect[0])


num_trees = 10
annoy_idx.build(num_trees)
annoy_idx.save("datasets_demo/embeddings_index.ann")

# %% [markdown]
# Now load it and query the index for the 8 most similar point patterns compared to the busstops pattern:

# %%
help(annoy_idx.get_nns_by_vector)

# %%
%%time
loaded_annoy_idx = AnnoyIndex(embedd_length, 'angular')

#loading is fast, will just mmap the file
loaded_annoy_idx.load("datasets_demo/embeddings_index.ann")

similar = loaded_annoy_idx.get_nns_by_vector(
               vector = embedd_vector_busstops[0],
               n = 8, 
               search_k = -1, 
               include_distances = True)

# %%
instances_similar =  similar[0]
print(instances_similar)

distances_similar = similar[1]
distances_similar

# %% [markdown]
# Visualize:

# %%
gmi_csr = np.load("datasets_demo/csr_GMI.npy")
gmi_clustered = np.load("datasets_demo/clustered_GMI.npy")

list_labels_similar = []

fig = plt.figure(figsize=(30,15), constrained_layout=True) 
gs = fig.add_gridspec(3, 6) 

ax = fig.add_subplot(gs[0:2, 0:2])
ax.imshow(arr_ij_busstops, cmap="coolwarm", interpolation = None)
ax.set_title("Busstops")
ax.set_axis_off()

ii = 0
for k in range(len(similar[0])):
               
    idx_pos = similar[0][k]
    if idx_pos < arr_ij_csr.shape[0]: 
        similar_arr_ij = arr_ij_csr[idx_pos]
        gmi = gmi_crs[idx_pos]
        list_labels_similar.append(0)
    else:
        idx_pos = idx_pos - arr_ij_csr.shape[0]
        similar_arr_ij = arr_ij_clustered[idx_pos]
        gmi = gmi_clustered[idx_pos]
        list_labels_similar.append(1)
    
    i = int(ii /4)
    j = 2 + int (ii  % 4)
    ax = fig.add_subplot(gs[i:i+1, j:j+1])

    ax.imshow(similar_arr_ij, cmap="coolwarm", interpolation = None)
    ax.set_title("GMI = {}".format(gmi))
    ax.set_axis_off()
    ii = ii + 1
    
print(Counter(list_labels_similar))
fig.tight_layout()

# %% [markdown]
# All of them were clustered pattern, as is the busstops pattern. However, the spatial distribution differs.

# %% [markdown]
# ------------------

# %% [markdown]
# -------------------

# %% [markdown]
# # V. 3D visualizations in JavaScript with deck.gl

# %%
def repr_html(html_data, height = 500):
    """Build the HTML representation for Jupyter."""
    srcdoc = html_data.replace('"', "'")
    
    ifr = '''<iframe srcdoc="{srcdoc}" style="width: 100%; height: {h}px; border: none">
             </iframe>'''
    return (ifr.format(srcdoc = srcdoc, h = height))


# %% [markdown]
# The following resources were guidelines for this part:
# - https://www.mapbox.com/mapbox-gl-js/example/3d-buildings/
# - http://deck.gl/showcases/gallery/hexagon-layer
# - https://github.com/uber/deck.gl/blob/master/docs/layers/hexagon-layer.md
# - https://github.com/uber/deck.gl/blob/master/docs/layers/geojson-layer.md
# - https://github.com/uber/deck.gl/blob/master/docs/layers/arc-layer.md
# 

# %%
# MAPBOX_TOKEN = '<THE_MAPBOX_API_TOKEN_HERE>';

# %%
%%bash

mkdir -p js/lib
cd js/lib
wget https://unpkg.com/s2-geometry@1.2.10/src/s2geometry.js
mv s2geometry.js s2Geometry.js
ls -alh

# %% [markdown]
# ## V.1. deck.gl Arc, Scatterplot and GeoJSON layers for the route of bus 14

# %%
srcall = """

<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no' />    
    <link href='https://api.tiles.mapbox.com/mapbox-gl-js/v0.51.0/mapbox-gl.css' 
          rel='stylesheet' />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.js">
    </script>
    <style>
        body { margin:0; padding:0; }
        #map { position:absolute; top:0; bottom:0; width:100%; }
    </style>
</head>
<body>

<div id="container">
   <div id="map"></div>
   <canvas id="deck-canvas"></canvas>
</div>

<script>

requirejs.config({"baseUrl": "js/lib",
                  "paths": {
    "my_mapboxgl" : 'https://api.tiles.mapbox.com/mapbox-gl-js/v0.53.1/mapbox-gl', 
    "h3" :       'https://cdn.jsdelivr.net/npm/h3-js@3.6.4/dist/h3-js.umd', 
    "my_deck" :  'https://unpkg.com/deck.gl@~8.0.2/dist.min', 
    "my_d3" :    'https://d3js.org/d3.v5.min' 
    } 
 });


require(['my_mapboxgl', 'my_deck', 'my_d3'], function(mapboxgl,deck,d3) {


  // --- mapboxgl ----------------------------------------------------------
  const INITIAL_VIEW_STATE = {
    latitude: 43.600378,
    longitude: 1.445478,
    zoom: 12,
    bearing: 30,
    pitch: 60
  };  

  mapboxgl.accessToken = '""" + MAPBOX_TOKEN + """';
  var mymap = new mapboxgl.Map({
                container: 'map', 
                style: 'mapbox://styles/mapbox/streets-v9', 
                center: [INITIAL_VIEW_STATE.longitude, INITIAL_VIEW_STATE.latitude],
                zoom: INITIAL_VIEW_STATE.zoom,
                bearing: INITIAL_VIEW_STATE.bearing, 
                pitch: INITIAL_VIEW_STATE.pitch,
                interactive: false 
              });

  mymap.on('load', () => {
    var layers = mymap.getStyle().layers;
    var labelLayerId;
    for (var i = 0; i < layers.length; i++) {
      if (layers[i].type === 'symbol' && layers[i].layout['text-field']) {
        labelLayerId = layers[i].id;
        break;
      }
    }


    mymap.addLayer({
      'id': '3d-buildings',
      'source': 'composite',
      'source-layer': 'building',
      'filter': ['==', 'extrude', 'true'],
      'type': 'fill-extrusion',
      'minzoom': 15,
      'paint': {
        'fill-extrusion-color': '#aaa',
        // use an 'interpolate' expression to add a smooth transition effect to the
        // buildings as the user zooms in
        'fill-extrusion-height': ["interpolate", ["linear"], ["zoom"], 15, 0,
                                   15.05, ["get", "height"] ],
        'fill-extrusion-base': ["interpolate", ["linear"], ["zoom"], 15, 0,
                                  15.05, ["get", "min_height"] ],
        'fill-extrusion-opacity': .6
       }
    }, labelLayerId);
  });  

  // ---  -------------------------------------------------------------
  function color_arc(x){
    if (x == 0){
      return [0,160,0];
    }
    else{
      return [250,0,0];
    }
  };
 

  // The positions of lights specified as [x, y, z], in a flattened array.
  // The length should be 3 x numberOfLights
  const LIGHT_SETTINGS = {
    lightsPosition: [1.288984920469113, 43.5615971219998, 2000, 
                     1.563934056342489, 43.52658309103259, 4000],
    ambientRatio: 0.4,
    diffuseRatio: 0.6,
    specularRatio: 0.2,
    lightsStrength: [0.8, 0.0, 0.8, 0.0],
    numberOfLights: 2
  };
  
  
  //add also the geometries of the traversed districts, in pale beige color 
  geo_layer_border = new deck.GeoJsonLayer({
    id: 'traversed_districts_border',
    data: d3.json('datasets_demo/bus_14_districts.geojson'),
    elevationRange: [0, 10],
    elevationScale: 1,
    extruded: false,
    stroked: true,
    filled: true,
    lightSettings: LIGHT_SETTINGS,
    opacity: 0.2,
    getElevation: 10,
    getLineColor: f => [194, 122, 66],
    getLineWidth: 50,
    getFillColor: f => [245, 198, 144],
  });
  

  // scatterplots of busstops points
  scatter_layer =  new deck.ScatterplotLayer({
    id: 'busstops_1',
    pickable: true,
    data: d3.json('datasets_demo/bus_14_route.json'),
    getPosition: d => [d.longitude, d.latitude,10],
    getColor: [0,0,0],
    radiusScale: 30
  });

  scatter_layer2 =  new deck.ScatterplotLayer({
    id: 'busstops_2',
    pickable: true,
    data: d3.json('datasets_demo/bus_14_route.json'),
    getPosition: d => [d.next_longitude, d.next_latitude,10],
    getColor: [0,0,0],
    radiusScale: 20
  });

  arcs_layer = new deck.ArcLayer({
    id: 'busroute',
    data: d3.json('datasets_demo/bus_14_route.json'),
    pickable: false,
    getWidth: 12,
    getSourcePosition: d => [d.longitude, d.latitude],
    getTargetPosition: d => [d.next_longitude, d.next_latitude],
    getSourceColor:  d => color_arc(d.sens),
    getTargetColor:  d => color_arc(d.next_sens)
  });
  
  const mydeck = new deck.Deck({
    canvas: 'deck-canvas',
    width: '100%',
    height: '100%',
    initialViewState: INITIAL_VIEW_STATE,
    controller: true,
    layers: [geo_layer_border,scatter_layer,scatter_layer2,arcs_layer],
    onViewStateChange: ({viewState}) => {
      mymap.jumpTo({
        center: [viewState.longitude, viewState.latitude],
        zoom: viewState.zoom,
        bearing: viewState.bearing,
        pitch: viewState.pitch
     });
    }
  });

});
</script>


</body>
</html>


"""

# Load the map into an iframe
map4_html = repr_html(srcall, height = 900)

# Display the map
display(HTML(map4_html))


# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/busline14_img1.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("3D visualization of busline 14 route and the districts it traverses")
ax.set_axis_off()


# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/busline14_img3.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("3D visualization of busline 14 route and the districts it traverses")
ax.set_axis_off()

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/busline14_img2.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("3D visualization of busline 14 stops, extruded buildings on zoom")
ax.set_axis_off()


# %% [markdown]
# ## V.2. deck.gl H3Hexagon layer for aggregated counts of busstops

# %%
!head -n 20 datasets_demo/counts_res9.json

# %%
srcall = """

<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='initial-scale=1,maximum-scale=1,user-scalable=no' />    
    <link href='https://api.tiles.mapbox.com/mapbox-gl-js/v0.51.0/mapbox-gl.css' 
          rel='stylesheet' />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.js">
    </script>
    <style>
        body { margin:0; padding:0; }
        #map { position:absolute; top:0; bottom:0; width:100%; }
    </style>
</head>
<body>

<div id="container">
   <div id="map"></div>
   <canvas id="deck-canvas"></canvas>
</div>

<script>



requirejs.config({"baseUrl": "js/lib",
                  "paths": {
    "my_mapboxgl" : 'https://api.tiles.mapbox.com/mapbox-gl-js/v0.53.1/mapbox-gl', 
    "h3" :       'https://cdn.jsdelivr.net/npm/h3-js@3.6.4/dist/h3-js.umd', 
    "my_deck" :  'https://unpkg.com/deck.gl@~8.0.2/dist.min', 
    "my_d3" :    'https://d3js.org/d3.v5.min' 
    } 
 });


require(['h3', 'my_mapboxgl', 'my_deck', 'my_d3'], function(h3,mapboxgl,deck,d3) {


  // --- mapboxgl ----------------------------------------------------------
  const INITIAL_VIEW_STATE = {
    latitude: 43.600378,
    longitude: 1.445478,
    zoom: 12,
    bearing: 30,
    pitch: 60
  };  

  mapboxgl.accessToken = '""" + MAPBOX_TOKEN + """';
  var mymap = new mapboxgl.Map({
                container: 'map', 
                style: 'mapbox://styles/mapbox/light-v9', 
                center: [INITIAL_VIEW_STATE.longitude, INITIAL_VIEW_STATE.latitude],
                zoom: INITIAL_VIEW_STATE.zoom,
                bearing: INITIAL_VIEW_STATE.bearing, 
                pitch: INITIAL_VIEW_STATE.pitch,
                interactive: false 
              });

  mymap.on('load', () => {
    var layers = mymap.getStyle().layers;
    var labelLayerId;
    for (var i = 0; i < layers.length; i++) {
      if (layers[i].type === 'symbol' && layers[i].layout['text-field']) {
        labelLayerId = layers[i].id;
        break;
      }
    }
  });  

  //---deckgl -------------------------------------------------------------
  const COLOR_RANGE = [
      [243, 240, 247],  //gray for counts = 0
      [0, 200, 0],
      [250, 250, 0],
      [250, 170, 90],
      [250, 70, 70]
    ];
  
  function colorScale(x) {
    list_thresholds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    for(var i = 0; i < list_thresholds.length; i++){
      if(x <= list_thresholds[i]){
        return COLOR_RANGE[i];
      }  
    }
    return COLOR_RANGE[COLOR_RANGE.length - 1];
  };
  
  function defaultcolorScale(x) {
    return [255, (1 - d.value / 3) * 255, 100];
  }

  hexes_layer = new deck.H3HexagonLayer({
    id: 'hexes_counts',
    data: d3.json('datasets_demo/counts_res9.json'),
    pickable: true,
    wireframe: false,
    filled: true,
    extruded: true,
    elevationScale: 30,
    elevationRange: [0, 100],
    getHexagon: d => d.hex_id,
    getElevation: d => d.value * 10,
    getFillColor: d => colorScale(d.value),
    opacity: 0.8
  });

  // lights
  const cameraLight = new deck._CameraLight({
    color: [255, 255, 255],
    intensity: 2.0
  });
  
  const pointLight1 = new deck.PointLight({
    color: [255, 255, 255],
    intensity: 2.0,
    position: [1.288984920469113, 43.5615971219998, 2000]
  });
  
  const pointLight2 = new deck.PointLight({
    color: [255, 255, 255],
    intensity: 2.0,
    position: [1.563934056342489, 43.52658309103259, 4000]
  });
  

  const mydeck = new deck.Deck({
    canvas: 'deck-canvas',
    width: '100%',
    height: '100%',
    initialViewState: INITIAL_VIEW_STATE,
    controller: true,
    layers: [hexes_layer],
    effects: [ new deck.LightingEffect({cameraLight}, pointLight1, pointLight2)],
    onViewStateChange: ({viewState}) => {
      mymap.jumpTo({
        center: [viewState.longitude, viewState.latitude],
        zoom: viewState.zoom,
        bearing: viewState.bearing,
        pitch: viewState.pitch
     });
    }
  });

});
</script>

</body>
</html>

"""

# Load the map into an iframe
map_html = repr_html(srcall, height=900)

# Display the map
display(HTML(map_html))


# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/vis_aggreg_img1.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("3D visualization of busstops aggregated by H3 cells at resolution 9")
ax.set_axis_off()

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

im1 = pilim.open('images/vis_aggreg_img2.png', 'r')
ax.imshow(np.asarray(im1))
ax.set_title("3D visualization of busstops aggregated by H3 cells at resolution 9")
ax.set_axis_off()

# %% [markdown]
# ### The end.

# %% [markdown]
# ----


