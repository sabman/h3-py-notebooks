#!/usr/bin/env python
# coding: utf-8

# <h1> Uber H3 API examples on Urban Analytics in the city of Toulouse (France)</h1>
# 
# <br/><br/>
# <font size="3"><b> Author: Camelia Ciolac </b></font><br/>
# ciolac_c@inria-alumni.fr

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

# In[12]:


# %%sh
# mkdir -p datasets_demo


# In[13]:


# %%sh
# wget -O datasets_demo/busstops_Toulouse.geojson --content-disposition -q \
#     "https://data.toulouse-metropole.fr/explore/dataset/arrets-de-bus0/download/?format=geojson&timezone=Europe/Helsinki"

# # the link is not working: 
# # possible alternative: https://data.toulouse-metropole.fr/explore/dataset/arrets-de-bus/download?format=geojson&timezone=Europe/Berlin&use_labels_for_header=false
# # taken from: https://data.europa.eu/data/datasets/5f76a6971606e3a0437ce91e?locale=en


# In[14]:


# %%sh
# ls -alh datasets_demo/busstops_*.geojson


# In[15]:


# %%sh
# wget -O datasets_demo/subzones_Toulouse.geojson --content-disposition -q \
#     "https://data.toulouse-metropole.fr/explore/dataset/communes/download/?format=geojson&timezone=Europe/Helsinki"


# In[16]:


# %%sh
# ls -alh datasets_demo/subzones_*.geojson


# In[17]:


# %%sh
# wget -O datasets_demo/districts_Toulouse.geojson --content-disposition -q \
#     "https://data.toulouse-metropole.fr/explore/dataset/recensement-population-2015-grands-quartiers-logement/download/?format=geojson&timezone=Europe/Helsinki"


# In[18]:


# %%sh
# ls -alh datasets_demo/districts_*.geojson


# ---

# ## Imports

# In[19]:


import json
import pandas as pd
# from pandas.io.json import json_normalize
from pandas import json_normalize
import numpy as np

import statistics
import statsmodels as sm
import statsmodels.formula.api as sm_formula
from scipy import stats

import tensorflow as tf
from tensorflow.keras import layers, models

print(tf.__version__)


# In[20]:


import warnings
warnings.filterwarnings('ignore')


# don't use scientific notation
np.set_printoptions(suppress=True) 
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[21]:


import h3

import geopandas as gpd

from shapely import geometry, ops
import libpysal as pys
import esda
import pointpats as pp

from geojson.feature import *


# In[22]:


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


# In[23]:


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

get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


import sys
sys.path.append('./PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks  import *


# In[25]:


get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')


# See https://www.flake8rules.com/ for codes

# In[26]:


get_ipython().run_line_magic('flake8_on', '--ignore E251,E703,W293,W291 --max_line_length 90')
get_ipython().run_line_magic('flake8_off', '')


# --------------------

# # II. Use H3 indexing for spatial operations

# ## II.1. Prepare data - GeoJSON file of bus stops

# In[27]:


gdf_raw = gpd.read_file("datasets_demo/busstops_Toulouse.geojson")
gdf_raw


# In[28]:


def load_and_prepare_busstops(filepath):
    """Loads a geojson files of point geometries and features,
    extracts the latitude and longitude into separate columns,
    deduplicates busstops (since multiple buslines share them)"""

    gdf_raw = gpd.read_file(filepath, driver="GeoJSON")
    print("Total number of bus stops in original dataset", gdf_raw.shape[0]) 

    gdf_raw["latitude"] = gdf_raw["geometry"].apply(lambda p: p.centroid.y)
    gdf_raw["longitude"] = gdf_raw["geometry"].apply(lambda p: p.centroid.x)

    # reset index to store it in a column
    gdf_raw.reset_index(inplace=True, drop = False)
    
    return gdf_raw


# In[29]:


input_file_busstops = "datasets_demo/busstops_Toulouse.geojson"
gdf_raw = load_and_prepare_busstops(filepath = input_file_busstops)

# display first 5 rows of the geodataframe, transposed
gdf_raw.head().T


# In[30]:


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


# In[31]:


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


# Better yet, we can plot a heatmap with pydeck (Docs at https://pydeck.gl/index.html):

# In[32]:


# print(dir(cm.linear))

steps = 5
color_map = cm.linear.RdYlGn_10.scale(0, 1).to_step(steps)

# in reverse order (green to red)
for i in range(steps-1, -1, -1):
    # would be fractional values, but we need them as RGB in [0,255]
    # also drop the alpha (4th element)
    print([int(255 * x) for x in color_map.colors[i][:-1]])

color_map


# In[33]:


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

MAPBOX_TOKEN = '<THE_MAPBOX_API_TOKEN_HERE>';

r = pydeck.Deck(
    layers=[busstops_layer],
    initial_view_state = view,
    # mapbox_key = MAPBOX_TOKEN,
    map_style='mapbox://styles/mapbox/light-v9'
)


# In[34]:


r.to_html('busstops.html', notebook_display=True)


# **Create a new dataframe to work with throughout the notebook:**

# In[35]:


gdf_raw_cpy = gdf_raw.reset_index(inplace = False, drop = False)
df_stops_to_buslines = gdf_raw_cpy.groupby(by=["longitude", "latitude"]).agg(
                                    {"index": list, "conc_ligne": set, "nom_log": "first"})

df_stops_to_buslines["info"] = df_stops_to_buslines[["nom_log", "conc_ligne"]].apply(
                                  lambda x: "{} ({})".format(x[0], ",".join(list(x[1]))), 
                                  axis = 1)
df_stops_to_buslines.reset_index(inplace = True, drop = False)
df_stops_to_buslines.head()


# ## II.2. Index data spatially with H3

# In[36]:


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


# ## II.3 Compute K Nearest Neighbors (spatial search) using the H3 index

# Create an inverted index hex_id_9 to list of row indices in df_stops_to_buslines:

# In[37]:


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


# For a given GPS location, we index it and then iterate over its hollow rings until we collect the candidates. Last step for computing result in descending distance, is to compute the actual Haversine distance: 

# In[38]:


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


# In[39]:


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


# In[40]:


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


# In[41]:


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
    


# Note: there exist bus stops on the 2nd hollow ring that are nearer to the source (which is marked by red circle) than some of the bus stops on the 1st hollow ring.  
# So it is adviseabale to always include one additional hollow ring of candidates before computing Haversine distance.

# -------------

# ## II.4. Compute Point in Polygon (spatial join) using the H3 index

# For this demo, we use the set of districts of Toulouse:

# In[42]:


def load_and_prepare_districts(filepath):
    """Loads a geojson files of polygon geometries and features,
    swaps the latitude and longitude andstores geojson"""

    gdf_districts = gpd.read_file(filepath, driver="GeoJSON")
    
    gdf_districts["geom_geojson"] = gdf_districts["geometry"].apply(
                                              lambda x: geometry.mapping(x))

    gdf_districts["geom_swap"] = gdf_districts["geometry"].map(
                                              lambda polygon: ops.transform(
                                                  lambda x, y: (y, x), polygon))

    gdf_districts["geom_swap_geojson"] = gdf_districts["geom_swap"].apply(
                                              lambda x: geometry.mapping(x))
    
    return gdf_districts


# In[43]:


input_file_districts = "datasets_demo/districts_Toulouse.geojson"
gdf_districts = load_and_prepare_districts(filepath = input_file_districts) 
 
print(gdf_districts.shape)
print("\n--------------------------------------------------------\n")
list_districts = list(gdf_districts["libelle_du_grand_quartier"].unique())
list_districts.sort()
print(columnize(list_districts, displaywidth=100))
print("\n--------------------------------------------------------\n")

gdf_districts[["libelle_du_grand_quartier", "geometry", 
               "geom_swap", "geom_swap_geojson"]].head()


# The approach is to fill each district geometry with hexgons at resolution 13 and then compact them.
# 
# **Initial fill:**

# In[44]:


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


# In[45]:


gdf_districts["hex_fill_initial"] = gdf_districts["geom_swap_geojson"].apply(
                                         lambda x: list(fill_hexagons(geom_geojson = x, 
                                                                      res = 13))
                                          )
gdf_districts["num_hex_fill_initial"] = gdf_districts["hex_fill_initial"].apply(len)

total_num_hex_initial = gdf_districts["num_hex_fill_initial"].sum()
print("Until here, we'd have to search over {} hexagons".format(total_num_hex_initial))

gdf_districts[["libelle_du_grand_quartier", "geometry", "num_hex_fill_initial"]].head()


# To reduce the number of hexagons we can benefit from H3 cells compacting.
# 
# **Compacted fill:**

# In[46]:


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


# In[47]:


# this column of empty lists is a placeholder, will be used further in this section
gdf_districts["compacted_novoids"] = [[] for _ in range(gdf_districts.shape[0])]


# In[48]:


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


# In[49]:


list_districts_names = ["MIRAIL-UNIVERSITE", "BAGATELLE", "PAPUS",
                        "FAOURETTE", "CROIX-DE-PIERRE"]
visualize_district_filled_compact(gdf_districts = gdf_districts,
                                  list_districts_names = list_districts_names)


# In the detail zoom that follows, we can observe that some small areas remained uncovered after compacting the set of hexagons used for filling districts geometries.   
# These small voids occur at the juxtaposition of hexagon cells of different H3 resolutions. 
# As explained in section I.2 of the preliminaries, the parent's polygon does not overlap completely with the multipolygon of its children union.
# 
# A consequence of this, for our spatial join, is that any point that would fall exactly in such a void would  be wrongly labelled as outside the district.
# 

# So far, how many hexagons belonged to more than one district (i.e were on the border between districts)?

# In[50]:


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


# In[51]:


_ = check_hexes_on_multiple_districts(gdf_districts, hexes_column = "hex_fill_compact")


# **Fill the voids**

# In[52]:


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
    


# In[53]:


boundary_hexes_res11 = get_hexes_traversed_by_borders(gdf_districts, res = 11)
boundary_hexes_res12 = get_hexes_traversed_by_borders(gdf_districts, res = 12)
boundary_hexes_res13 = get_hexes_traversed_by_borders(gdf_districts, res = 13)

print("{} hexes on boundary at res {}".format(len(boundary_hexes_res11), 11))
print("{} hexes on boundary at res {}".format(len(boundary_hexes_res12), 12))
print("{} hexes on boundary at res {}".format(len(boundary_hexes_res13), 13))


# In[54]:


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


# In[55]:


get_ipython().run_cell_magic('time', '', 'gdf_districts["compacted_novoids"] = gdf_districts.apply(lambda r: fill_voids(r), axis = 1)\n')


# In[56]:


_ = check_hexes_on_multiple_districts(
                          gdf_districts, 
                          hexes_column = "compacted_novoids")


# In[57]:


list_districts_names = ["MIRAIL-UNIVERSITE", "BAGATELLE", "PAPUS",
                        "FAOURETTE", "CROIX-DE-PIERRE"]
visualize_district_filled_compact(gdf_districts = gdf_districts,
                                  list_districts_names = list_districts_names)


# In[58]:


# sidenote - how it works itertools.chain.from_iterable
l1 = ["a", "b"]
l2 = ["a", "c"]
list(itertools.chain.from_iterable([l1, l2]))


# In[59]:


gdf_districts["union_compacted_novoids"] = \
             gdf_districts[["compacted_novoids", "hex_fill_compact"]].apply(
             lambda x: list(itertools.chain.from_iterable([x[0], x[1]])), axis = 1)
gdf_districts["union_compacted_novoids"] = gdf_districts["union_compacted_novoids"].apply(
             lambda x: list(set(x)))
gdf_districts["num_final"] = gdf_districts["union_compacted_novoids"].apply(
             lambda x: len(x))

gdf_districts["num_final"].sum()


# Note: these 282148 multi-resolution H3 cells seem as a good trade-off compared with the former 2 extremes: the initial dense filling at resolution 13 with 2851449 hexagons versus the 94287 hexagons after compacting which left uncovered areas(voids) 

# In[60]:


dict_hex_districts = check_hexes_on_multiple_districts(
                          gdf_districts, 
                          hexes_column = "union_compacted_novoids")


# Now, for a given point, index it at all resolutions between 6 and 12 and search starting from coarser resolution towards finer resolutions:

# In[61]:


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


# In[62]:


list_res_after_compact_novoids = [h3.h3_get_resolution(x) for x in dict_hex_districts]
finest_res = max(list_res_after_compact_novoids)
coarsest_res = min(list_res_after_compact_novoids)
print("Resolution between {} and {}".format(coarsest_res, finest_res))


# In[63]:


get_ipython().run_cell_magic('time', '', '\ndf_sjoin_h3 = df_stops_to_buslines.copy()\n\ndf_sjoin_h3["district"] = df_sjoin_h3.apply(spatial_join_districts, \n                                            args=(dict_hex_districts,\n                                                  coarsest_res,\n                                                  finest_res), \n                                            axis = 1)\n')


# In[64]:


counts_by_district = pd.DataFrame(df_sjoin_h3["district"].value_counts())
counts_by_district.columns = ["num_busstops"]
counts_by_district.head()


# Note: the N/A category includes all busstops that are outside the districts (but in the wider metropolitan area of Toulouse)

# In[65]:


# the number of bus stops that were found inside the districts
counts_by_district[counts_by_district.index != "N/A"]["num_busstops"].sum()


# In[66]:


# bus stops situated on the border of 2 districts
counts_by_district[counts_by_district.index.str.contains(",")]


# In[67]:


special_map = visualize_district_filled_compact(
                     gdf_districts = gdf_districts,
                     list_districts_names =["AMIDONNIERS", "CASSELARDIT"],
                     fillcolor="pink")

df_on_border = df_sjoin_h3[df_sjoin_h3["district"] == "AMIDONNIERS,CASSELARDIT"]

for i, row in df_on_border.iterrows():
    mk = Marker(location=[row["latitude"], row["longitude"]],
                icon = folium.Icon(icon='circle', color='darkgreen'),
                popup=str(row["info"]))
    mk.add_to(special_map)
    
special_map


# **After having computed the spatial join, we can use the results for identifying which are the districts served by each bus line**

# ### Doesn't work

# Rest of the section <font color="red">DOES NOT WORK BECAUSE OF DATA SHIFT</font>

# Comparison of the columns we have and what we should have but don't
# 
# | Old         | New         |
# | ----------- | ----------- |
# | index       | index       |
# | gir         |             |
# | code_expl   | code_expl   |
# | pty         |             |
# | code_log    | code_log    |
# | nom_ligne   |             |
# | ligne       | conc_ligne  |
# | ordre       |             |
# | nom_log     | nom_log     |
# | id_ligne    |             |
# | nom_expl    | nom_expl    |
# | nom_iti     |             |
# | sens        |             |
# | id          | id          |
# | mode        | conc_mode   |
# | geometry    | geometry    |
# | latitude    | latitude    |
# | longitude   | longitude   |

# In[69]:


# selected_busline = "14"
# print(gdf_raw[gdf_raw["conc_ligne"] == selected_busline]["pty"].unique())
# print(gdf_raw[gdf_raw["conc_ligne"] == selected_busline].groupby(by="pty")["sens"].apply(set))

# df_route_busline = pd.merge(left = gdf_raw[gdf_raw["pty"].isin(['14/106', '14/13'])], 
#                             right = df_sjoin_h3,
#                             left_on = ["latitude", "longitude"],
#                             right_on = ["latitude", "longitude"],
#                             how = "left")

# df_route_busline.sort_values(by = ["pty", "sens", "ordre"], inplace = True)
# df_route_busline[["pty", "sens", "ordre", "info", "district"]]


# In[70]:


# direction_0 = df_route_busline[df_route_busline["sens"] == 0]["district"]
# list(unique_everseen(direction_0))


# In[71]:


# list_aux = list(unique_everseen(df_route_busline["district"].values))
# list_distr = []
# for s in list_aux:
#     if "," in s:
#         # if on border, add both districts 
#         list_distr.extend(s.split(","))
#     else:
#         list_distr.append(s)
        
# gdf_bus_traversed_districts = gdf_districts[
#                           gdf_districts["libelle_du_grand_quartier"].isin(list_distr)]
# gdf_bus_traversed_districts = gdf_bus_traversed_districts[
#                                             ["geometry", "libelle_du_grand_quartier"]]
# gdf_bus_traversed_districts.to_file("datasets_demo/bus_14_districts.geojson",
#                                     driver = "GeoJSON")


# In[72]:


# !ls -alh datasets_demo/bus_14_districts.geojson


# Recall that we have comma in district when the point was found on the border between 2 districts.
# 
# Prepare files for the section V.1

# In[73]:


# list_projected_columns = ["latitude", "longitude", "sens", "ordre", "info", "district"]
# df_route_busline_cpy = df_route_busline[list_projected_columns]
# df_route_busline_cpy.sort_values(by = ["sens", "ordre"], inplace = True)

# # shift
# df_route_busline_cpy['next_longitude'] = df_route_busline_cpy["longitude"].shift(-1)
# df_route_busline_cpy['next_latitude'] = df_route_busline_cpy["latitude"].shift(-1)
# df_route_busline_cpy['next_sens'] = df_route_busline_cpy["sens"].shift(-1)
# df_route_busline_cpy['next_ordre'] = df_route_busline_cpy["ordre"].shift(-1)

# # the last row will have next_{} all none, we manually match it to the start of the route
# df_route_busline_cpy["next_latitude"].fillna(df_route_busline_cpy.iloc[0]["latitude"],
#                                              inplace=True)
# df_route_busline_cpy["next_longitude"].fillna(df_route_busline_cpy.iloc[0]["longitude"],
#                                               inplace=True)
# df_route_busline_cpy["next_sens"].fillna(0, inplace=True)
# df_route_busline_cpy["next_ordre"].fillna(1, inplace=True)

# df_route_busline_cpy


# In[74]:


# json_rep = df_route_busline_cpy.to_dict(orient='record')

# with open("datasets_demo/bus_14_route.json", mode="w") as f:
#     json.dump(json_rep, f)


# In[75]:


# %%sh
# ls -alh datasets_demo/bus_14_route.json


# **See the corresponding 3d visualization with Deck.gl in section V.1 at the end of this notebook.**

# -------------------

# # III. Use H3 spatial index for aggregated analytics

# ## III.1. Count busstops groupped by H3 cell

# In[76]:


df_aggreg = df_stops_to_buslines.groupby(by = f"hex_id_{8}").agg({f"geometry_{8}": "first",
                                                "latitude": "count"})
df_aggreg.reset_index(inplace = True)
df_aggreg.rename(columns={"latitude": "value"}, inplace = True)


# In[77]:


df_aggreg.sort_values(by = "value", ascending = False, inplace = True)
df_aggreg


# In[78]:


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


# In[79]:


# demo at resolution 8
df_aggreg_8 = counts_by_hexagon(df = df_stops_to_buslines, res = 8)
print(df_aggreg_8.shape)
df_aggreg_8.head(5)


# ## III.2. Visualization with choropleth map

# In[80]:


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


# In[81]:


m_hex = choropleth_map(df_aggreg = df_aggreg_8,
                       hex_id_field = "hex_id_8",
                       geometry_field = "geometry_8",
                       value_field = "value",
                       layer_name = "Choropleth 8",
                       with_legend = True)
m_hex


# Better yet, plot it 3d with pydeck:

# In[82]:


norm = mpl.colors.Normalize(vmin = df_aggreg_8["value"].min(), 
                            vmax = df_aggreg_8["value"].max())
f2rgb = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.get_cmap('RdYlGn_r'))


def get_color(value):
    return [int(255 * x) for x in f2rgb.to_rgba(value)[:-1]]


get_color(value = 10)


# In[83]:


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
    # mapbox_key = MAPBOX_TOKEN,
    map_style='mapbox://styles/mapbox/light-v9',
    tooltip={"text": "Count: {value}"}
)

r.to_html('aggregate.html', notebook_display=True)


# **Aggregate at  coarser and at finer resolutions:**
# 

# In[84]:


# coarser resolutions than 8
df_aggreg_7 = counts_by_hexagon(df = df_stops_to_buslines, res = 7)

# finer resolutions than 8
df_aggreg_9 = counts_by_hexagon(df = df_stops_to_buslines, res = 9)
df_aggreg_10 = counts_by_hexagon(df = df_stops_to_buslines, res = 10)


# In[85]:


# make a dictionary of mappings resolution -> dataframes, for future use
dict_aggreg_hex = {7: df_aggreg_7,
                   8: df_aggreg_8,
                   9: df_aggreg_9,
                   10: df_aggreg_10}

msg_ = "At resolution {} we used {} H3 cells for indexing the bus stops"
for res in dict_aggreg_hex:
    print(msg_.format(res, dict_aggreg_hex[res].shape[0]))


# In[86]:


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


# First we focus (zoom in) on the city center and display H3 cells covering the same zone at various resolutions:
# 

# Depending on the resolution at which we computed the aggregates, we sometimes got a sparse spatial distribution of H3 cells with busstops.  
# Next we want to include all the H3 cells that cover the city's area and thus put these aggregates in a better perspective.

# ## III.3. Study aggregates in the context of the city's hexagons coverage set

# In[87]:


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


# There are 37 subzones that form Toulouse metropolitan territory, here we'll focus on the central subzone: 

# In[88]:


# we select the main subzone of the city
selected_subzone = "TOULOUSE"
gdf_subzone_sel = gdf_subzones[gdf_subzones["libcom"] == "TOULOUSE"]
gdf_subzone_sel


# Fill the subzone's geometry with H3 cells (as we've done before with districts, but without compacting this time)

# In[93]:


import shapely
from h3ronpy.pandas.vector import geodataframe_to_cells


# In[94]:


def fill_multipolygon_with_hexagons(gdf_subzone_sel, res, flag_swap = False, flag_return_df = False):
    """Fills a geometry given in geojson format with H3 hexagons at specified
    resolution. The flag_reverse_geojson allows to specify whether the geometry
    is lon/lat or swapped"""
    cells = geodataframe_to_cells(gdf_subzone_sel[['geometry']], res)['cell']
    cells = cells.apply(h3.h3_to_string)
    set_hexagons = set(cells)
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


# In[95]:


gdf_subzone_sel


# In[96]:


geom_to_fill = gdf_subzone_sel.explode()['geometry'].values[0]

dict_fillings = {}
msg_ = "the subzone was filled with {} hexagons at resolution {}"

for res in [8, 9, 10]:
    # lat/lon in geometry_swap_geojson -> flag_reverse_geojson = False
    df_fill_hex = fill_multipolygon_with_hexagons(gdf_subzone_sel,
                                res = res,
                                flag_return_df = True)
    print(msg_.format(df_fill_hex.shape[0], res))

    # add entry in dict_fillings
    dict_fillings[res] = df_fill_hex

# --------------------------
dict_fillings[8].head()


# **Merge (by left outer join) two H3 spatially indexed datasets at the same H3 index resolution**

# In[97]:


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


# <b>Visualize on map</b>

# In[99]:


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


# In[100]:


# percentage of cells with value zero at varius index resolutions

msg_ = "Percentage of cells with value zero at resolution {}: {} %"
for res in dict_filled_aggreg:
    df_outer = dict_filled_aggreg[res]
    perc_hexes_zeros = 100 * df_outer[df_outer["value"] == 0].shape[0] / df_outer.shape[0]
    print(msg_.format(res, round(perc_hexes_zeros, 2)))


# **See the corresponding 3d visualization with Deck.gl in section V.2 at the end of this notebook.**

# In[101]:


df_aux = dict_filled_aggreg[9].drop(columns = ["geometry"])
df_aux.to_json("datasets_demo/counts_res9.json", orient = "records", indent = 4)


# In[102]:


get_ipython().system('ls -alh datasets_demo/counts_res9.json')


# In[103]:


get_ipython().system('head -n 20 datasets_demo/counts_res9.json')


# ---------------------------------

# # IV. Global Spatial Autocorrelation

# ## IV.1 Background

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

# **Prepare the dataframes with precomputed z-scores and first hollow ring, at various resolutions**

# In[104]:


from shapely import Polygon


# In[105]:


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
        df_aux["geometry_shp"] = df_aux["hex_id"].apply(lambda h3_cell: Polygon(h3.h3_to_geo_boundary(h3_cell, geo_json=True)))
                                              # lambda x:
                                              # geometry.Polygon(geometry.shape(x)))
        df_aux.rename(columns={"geometry": "geometry_geojson"}, inplace=True)

        geom = df_aux["geometry_shp"]
        df_aux.drop(columns=["geometry_shp"], inplace = True)
        gdf_aux = gpd.GeoDataFrame(df_aux, crs="EPSG:4326", geometry=geom)

        return gdf_aux
    else:
        return df_aux


# In[106]:


dict_filled_aggreg[8]


# In[107]:


dict_prepared_GMI = {}

for res in dict_filled_aggreg:
    gdf_gmi_prepared = prepare_geodataframe_GMI(dict_filled_aggreg[res],
                                                num_rings = 1,
                                                flag_debug = True)
    dict_prepared_GMI[res] = gdf_gmi_prepared

# -----------------------
dict_prepared_GMI[8].head()


# When we look in the Global Moran'I numerator in (5), the sum $\sum_{j}{ w_{ij} * z_j }$ is in fact the spatial lag of cell $i$ .  
# 
# Moran's diagram is a scatterplot that visualizes the relationship between the spatial lag and the z-score of each geometry. The slope of the fitted regression line is quite the value of the Global Moran's I.

# In[108]:


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


# In[109]:


gdf_spatial_lags_8 = compute_spatial_lags_using_H3(gdf_prepared = dict_prepared_GMI[8],
                                                   variable_col = "z_score")

gdf_spatial_lags_8.head()


# **The Linear Regression:**

# In[110]:


result = sm_formula.ols(formula = "spatial_lag ~ z_score", 
                        data = gdf_spatial_lags_8).fit()

params = result.params.to_dict()
print(params, "\n")
slope = params["z_score"]
print("Global Moran'I approximated by slope of the regression line:", slope)
print("\n----------------------------------------------------------------\n")

print(result.summary())


# In[111]:


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


# ## IV.2. The PySAL baseline 

# Read docs at: https://splot.readthedocs.io/en/stable/users/tutorials/autocorrelation.html
# 
# Based on our column of geometries (Shapely objects), PySAL will build its own weights matrix.

# In[112]:


def wrapper_over_esda_Global_Moran_I(gdf_prepared, geometry_field, value_field):

    # weights
    wq = pys.weights.Queen.from_dataframe(df = gdf_prepared,
                                          geom_col = "geometry")
    y = gdf_prepared[value_field].values

    # transformation="r" performs row-standardization of weights matrix
    mi = esda.moran.Moran(y = y, w = wq, transformation="r",
                          permutations=999, two_tailed=True)
    return mi


# In[113]:


mi = wrapper_over_esda_Global_Moran_I(gdf_prepared = dict_prepared_GMI[8],
                                      geometry_field = "geometry",
                                      value_field = "value")

print("\nGlobal Moran I:", mi.I, "   p_sim =", mi.p_sim)


# In[114]:


get_ipython().run_cell_magic('capture', '', '# we used capture to prevent displaying lots of warnings of island geometries, such as:\n# (\'WARNING: \', 208, \' is an island (no neighbors)\')\n\nmi = wrapper_over_esda_Global_Moran_I(gdf_prepared = dict_prepared_GMI[9],\n                                      geometry_field = "geometry",\n                                      value_field = "value")\n')


# In[115]:


print("\nGlobal Moran I:", mi.I, "   p_sim =", mi.p_sim)


# In[116]:


get_ipython().run_cell_magic('capture', '', '# we used capture to prevent displaying lots of warnings of island geometries\nmi = wrapper_over_esda_Global_Moran_I(gdf_prepared = dict_prepared_GMI[10],\n                                      geometry_field = "geometry",\n                                      value_field = "value")\n')


# In[117]:


print("\nGlobal Moran I:", mi.I, "   p_sim =", mi.p_sim)


# Interpretation: while at resolution 10, we fail to reject H0 of spatial randomness, at resolution 8 and at resolution 9 we can reject H0 and conclude that there is positive global spatial autocorrelation (clustering) in the dataset.

# ## IV.3. Implementation of Global Moran's I formula from scratch using H3 

# This time we manage the whole computation and use the ring1 column, instead of geometries.

# In[118]:


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


# In[119]:


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


# **Compute at various index resolutions**

# In[120]:


get_ipython().run_cell_magic('time', '', '\nI_8 = compute_Global_Moran_I_using_H3(gdf_prepared = dict_prepared_GMI[8])\nprint("I =", I_8)\n')


# In[121]:


get_ipython().run_cell_magic('time', '', 'p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[8],   \n                                    num_permut = 999,                            \n                                    I_observed = I_8,\n                                    alternative = "two-tailed",\n                                    flag_plot = True)\n')


# In[122]:


get_ipython().run_cell_magic('time', '', 'p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[8],   \n                                    num_permut = 999,                            \n                                    I_observed = I_8,\n                                    alternative = "greater",\n                                    flag_plot = False)\n')


# In[123]:


get_ipython().run_cell_magic('time', '', 'I_9 = compute_Global_Moran_I_using_H3(gdf_prepared = dict_prepared_GMI[9])\nprint("I =",I_9)\n')


# In[124]:


get_ipython().run_cell_magic('time', '', 'p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[9], \n                                    num_permut = 999,                            \n                                    I_observed = I_9,\n                                    alternative = "two-tailed",\n                                    flag_plot = True)\n')


# In[125]:


get_ipython().run_cell_magic('time', '', 'p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[9], \n                                    num_permut = 999,                            \n                                    I_observed = I_9,\n                                    alternative = "greater",\n                                    flag_plot = False)\n')


# In[126]:


get_ipython().run_cell_magic('time', '', 'I_10 = compute_Global_Moran_I_using_H3(gdf_prepared = dict_prepared_GMI[10])\n\nprint("I =",I_10)\n')


# In[127]:


get_ipython().run_cell_magic('time', '', 'p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[10],  \n                                    num_permut = 999,                            \n                                    I_observed = I_10,\n                                    alternative = "two-tailed",\n                                    flag_plot = True)\n')


# In[128]:


get_ipython().run_cell_magic('time', '', 'p_sim = reshuffle_and_recompute_GMI(gdf_prepared = dict_prepared_GMI[10],  \n                                    num_permut = 999,                            \n                                    I_observed = I_10,\n                                    alternative = "less",\n                                    flag_plot = False)\n')


# ## IV.4. Spatial Autocorrelation Prediction with Tensorflow

# We build a Convolutional Neural Network with Tensorflow, able to classify an input spatial distribution of points (over the central subzone of Toulouse), bucketed into H3 cells at resolution 9 and converted to a matrix using H3 IJ coordinates system, into one of the following 2 classes: 
#  * complete spatial randomness
#  * global spatial autocorrelation (clustered)
#  
# Note: the IJ coordinate system was overviewed in the preliminaries section I.3. of this notebook.

# Having chosen to prototype for resolution 9 of the H3 index, let's first see the matrix size corresponding to the central subzone of Toulouse: 

# In[129]:


df_test = dict_prepared_GMI[9][["hex_id", "z_score"]]
df_test.head()


# ### IV.4.1. Dataframe to matrix

# In[130]:


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


# In[131]:


arr_ij_busstops = df_to_matrix(df = df_test)
print(arr_ij_busstops.shape)



# In[132]:


# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(arr_ij_busstops, cmap='coolwarm', interpolation = None)
# ax.set_axis_off()
# fig.savefig("images/matrix_city_busstops.png");


# ### IV.4.2. Generate dataset for training:

# For this, we'll use PySAL's Pointpats library:

# In[133]:


# pp.PoissonPointProcess
# pp.PoissonClusterPointProcess


# In[134]:


# create spatial window for generating points
geom_subzone = gdf_subzone_sel["geometry"].explode().values[0]
xs = geom_subzone.exterior.coords.xy[0]
ys = geom_subzone.exterior.coords.xy[1]
vertices = [(xs[i], ys[i]) for i in range(len(xs))]
print(vertices[0:10])
print(" ------------------------------------------------------------------- ")

window = pp.Window(vertices)
print("Window's bbox:", window.bbox)


# In[135]:


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


# In[136]:


fig, ax = plt.subplots(1, 2, figsize = (20, 10))

ax[0].fill(xs, ys, alpha=0.1, fc='r', ec='none')
ax[0].scatter(df_csr[["longitude"]], df_csr[["latitude"]], 
              fc = "blue", marker=".", s = 35)
ax[0].set_title("Random")

ax[1].fill(xs, ys, alpha=0.1, fc='r', ec='none')
ax[1].scatter(df_clustered[["longitude"]], df_clustered[["latitude"]], 
              fc = "blue", marker=".", s = 35)
ax[1].set_title("Clustered");


# Note: varying the radius of the clustered point process we get various degrees of spatial clustering.
# 
# What we have to do next is to index the realizations into H3 at resolution 9, outer join with the set of cells of resolution 9 covering this city subzone and compute Global Moran's I.

# In[137]:


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
    


# Distributions from which to draw the num_points_to_generate,num_parents, radius_offsprings

# In[138]:


# sidenote: how it works
list_multiples_of_100 = [random.randrange(100, 1000, 100) for _ in range(50)]
print(Counter(list_multiples_of_100))

list_choice = [random.choice([0.01, 0.02, 0.03, 0.05]) for _ in range(50)]
print(Counter(list_choice))


# **Generate a small batch of samples with randomly distributed points**

# In[139]:


get_ipython().run_cell_magic('time', '', 'arr_matrices = None\narr_GMI = np.array([])\narr_labels = np.array([])\narr_points = []\n\nk = 0\nwhile k < 10:\n    arr_ij, GMI, points = generate_training_sample(\n                            window = window,\n                            flag_clustered = False, \n                            num_points_to_generate = random.randrange(100, 1000, 100),\n                            num_parents = None, \n                            radius_offsprings = None)\n    if GMI is not None:\n        if arr_matrices is None:\n            arr_matrices = np.array([arr_ij])\n        else:\n            arr_matrices = np.vstack((arr_matrices, [arr_ij]))\n        arr_GMI = np.append(arr_GMI, GMI)\n        arr_labels = np.append(arr_labels, 0)\n        arr_points.append(points)\n        k = k + 1\n\nnp.save("datasets_demo/smallbatch.npy", arr_matrices)\n')


# In[140]:


fig, ax = plt.subplots(3, 3, figsize = (15, 15))

arr_restored = np.load("datasets_demo/smallbatch.npy")

for k1 in range(3):
    for k2 in range(3):
        arr_ij = arr_restored[3 * k1 + k2]
        GMI = arr_GMI[3 * k1 + k2]
        ax[k1][k2].imshow(arr_ij, cmap='coolwarm', interpolation = None)
        ax[k1][k2].set_title("GMI = {}".format(round(GMI, 3)))


# **Generate a small batch of samples with spatially clustered points:**

# In[141]:


get_ipython().run_cell_magic('time', '', 'arr_matrices = None\narr_GMI = np.array([])\narr_labels = np.array([])\narr_points = []\n\nk = 0\nwhile k < 10:\n    arr_ij, GMI, points  = generate_training_sample(\n                            window = window,\n                            flag_clustered = True, \n                            num_points_to_generate = random.randrange(100, 1000, 100),\n                            num_parents = random.randrange(10, 100, 10), \n                            radius_offsprings = random.choice([0.01, 0.02, 0.03, 0.05]))\n    if GMI is not None:\n        if arr_matrices is None:\n            arr_matrices = np.array([arr_ij])\n        else:\n            arr_matrices = np.vstack((arr_matrices, [arr_ij]))\n        arr_GMI = np.append(arr_GMI, GMI)\n        arr_labels = np.append(arr_labels, 0)\n        arr_points.append(points)\n        k = k + 1\n\nnp.save("datasets_demo/smallbatch2.npy", arr_matrices)\n')


# In[142]:


fig, ax = plt.subplots(3, 3, figsize = (15, 15))

arr_restored = np.load("datasets_demo/smallbatch2.npy")

for k1 in range(3):
    for k2 in range(3):
        arr_ij = arr_restored[3 * k1 + k2]
        GMI = arr_GMI[3 * k1 + k2]
        ax[k1][k2].imshow(arr_ij, cmap='coolwarm', interpolation = None)
        ax[k1][k2].set_title("GMI = {}".format(round(GMI, 3)))


# Correspond to the following clustered point process realizations:

# In[143]:


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


# **Generate the actual training set:**

# In[144]:


get_ipython().run_cell_magic('capture', '', 'arr_matrices = None\narr_GMI = np.array([])\narr_labels = np.array([])\nlist_points = []\n\nk = 0\nwhile k < 10:\n    arr_ij, GMI, points = generate_training_sample(\n                            window = window,\n                            flag_clustered = False, \n                            num_points_to_generate = 100, #random.randrange(100, 1000, 100),\n                            num_parents = None, \n                            radius_offsprings = None)\n    if GMI is not None:\n        if arr_matrices is None:\n            arr_matrices = np.array([arr_ij])\n        else:\n            arr_matrices = np.vstack((arr_matrices, [arr_ij]))\n        arr_GMI = np.append(arr_GMI, GMI)\n        arr_labels = np.append(arr_labels, 0)\n        # arr_points = np.concatenate((*arr_points, ))\n        list_points.append(points)\n        k = k + 1\n\nnp.save("datasets_demo/csr_matrices.npy", arr_matrices)\nnp.save("datasets_demo/csr_GMI.npy", arr_GMI)\n\narr_points = np.array(list_points)\nnp.save("datasets_demo/csr_points.npy", arr_points)\n')


# In[145]:


get_ipython().run_cell_magic('capture', '', '\narr_matrices = None\narr_GMI = np.array([])\narr_labels = np.array([])\nlist_points = []\n\nk = 0\nwhile k < 10:\n    arr_ij, GMI, points = generate_training_sample(\n                            window = window,\n                            flag_clustered = True, \n                            num_points_to_generate = 100,#random.randrange(100, 1000, 100),\n                            num_parents = random.randrange(10, 100, 10), \n                            radius_offsprings = random.choice([0.01, 0.02, 0.03, 0.05]))\n    if GMI is not None:\n        if arr_matrices is None:\n            arr_matrices = np.array([arr_ij])\n        else:\n            arr_matrices = np.vstack((arr_matrices, [arr_ij]))\n        arr_GMI = np.append(arr_GMI, GMI)\n        arr_labels = np.append(arr_labels, 0)\n        list_points.append(points)\n        k = k + 1\n\nnp.save("datasets_demo/clustered_matrices.npy", arr_matrices)\nnp.save("datasets_demo/clustered_GMI.npy", arr_GMI)\n\narr_points = np.array(list_points)\nnp.save("datasets_demo/clustered_points.npy", arr_points)\n')


# In[146]:


get_ipython().system('rm datasets_demo/a.npy')
get_ipython().system('ls -al datasets_demo/*.npy')


# ### IV.4.3 Prepare Tensorflow Dataset:

# In[147]:


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


# In[148]:


train_dataset, validation_dataset = prepare_datasets()


# In[149]:


# get a batch of samples

# note: make_one_shot_iterator was deprecated in tf v2
iterator = train_dataset.__iter__() 
x_batch = next(iterator)

print(type(x_batch[0]), x_batch[0].dtype, x_batch[0].shape)
print(type(x_batch[1]), x_batch[1].dtype, x_batch[1].shape)


# In[150]:


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


# ### IV.4.4. Build the Tensorflow model:

# The first convolution layer has a specified kernel and is not trainable, its role being of computing the spatial lag 

# In[151]:


def get_fixed_kernel(shape = (3, 3, 1, 1), dtype=np.float32):
    kernel_fixed = np.array([[1/6, 1/6, 0],
                            [1/6, 1, 1/6],
                            [0, 1/6, 1/6]])
    kernel = np.zeros(shape)
    kernel[:, :, 0, 0] = kernel_fixed
    return kernel


get_fixed_kernel()


# In[152]:


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


# In[153]:


model_classif = build_classifier()
model_classif.summary()


# ### IV.4.5 Train the model

# In[154]:


batch_size = 4
num_iter_per_epoch_train = 1000//batch_size
num_iter_per_epoch_valid = 200//batch_size

print("Iterations per epoch: training {}  validation {}".format(
                                             num_iter_per_epoch_train, 
                                             num_iter_per_epoch_valid))
print("Num_batches samples trained on per epoch = ", 
      batch_size * num_iter_per_epoch_train)


# In[155]:


get_ipython().system('rm -r -f tf_models/checkpoint_classif')
get_ipython().system('mkdir -p tf_models/checkpoint_classif')


# In[156]:


checkpoint_filepath = 'tf_models/checkpoint_classif'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath,
                                save_weights_only=False,
                                monitor='accuracy',
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


# In[157]:


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


# In[158]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[159]:


print(model_classif.output_names)

fig, ax = plt.subplots(1, 1, figsize = (15, 7))
ax.plot(history.history['accuracy'])
# ax.plot(history.history['val_accuracy'])
ax.set_title('model accuracy')
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.legend(['train', 'val'], loc='upper left')


# In[160]:


get_ipython().system('ls -alh tf_models/checkpoint_classif')


# In[161]:


#sidenote: we can verify that the hexagonal convolution preserved the kernel specified by us

print(model_classif.get_layer("HexConv").get_weights())


# <br/>

# ### IV.4.6. Load the best iteration model and make predictions

# In[162]:


loaded_classifier = tf.keras.models.load_model("tf_models/checkpoint_classif")

# ------------------- 

print(list(loaded_classifier.signatures.keys())) 

infer = loaded_classifier.signatures["serving_default"]
print(infer.structured_outputs)


# In[163]:


get_ipython().system('ls tf_models/checkpoint_classif')


# In[164]:


def predicted_label(arr_ij):   
    
    # reshape input from (m,n) to (1,m,n)
    reshaped_input = arr_ij[np.newaxis, :, :]
    
    #the result from the binary classifier
    prediction_logits = loaded_classifier.predict([reshaped_input])
    top_prediction = tf.argmax(prediction_logits, 1)
    the_label = top_prediction.numpy()[0]    
   
    return prediction_logits[0], the_label


# In[165]:


pred_logits, the_label  = predicted_label(arr_ij_busstops)

dict_decode = {0: "CSR", 1: "Clustered"}

print(pred_logits, 
      " ---> PREDICTED CLASS:", the_label, 
      " ---> DECODED AS: ", dict_decode[the_label])


# **Confusion matrix and confused samples**

# In[166]:


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


# **Confused samples:**

# In[167]:


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


# In[168]:


plt.show()


# The point processes realizations of these were:

# In[169]:


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


# ### IV.4.7. Attemp to find similar point process realizations using embeddings
# 
# Based on the embeddings extracted from the trained CNN, we seek to retrieve training instances which are most similar to a given query pattern.

# Extract embeddings at a specified layer of the CNN:

# In[170]:


list_layers = loaded_classifier.layers
assert(list_layers[-1].name == list(infer.structured_outputs.keys())[0])
print(list_layers[-2].name)


# In[171]:


embeddings_extractor = tf.keras.Model(loaded_classifier.inputs,
                                      loaded_classifier.get_layer(list_layers[-2].name).output)
embeddings_extractor.summary()


# In[172]:


reshaped_input = arr_ij_busstops[np.newaxis, :, :]

embedd_vector_busstops = embeddings_extractor.predict([reshaped_input])
embedd_vector_busstops[0]


# Extract embeddings of all training samples and build an index of them in Annoy.
# 
# Annoy is an open source project by Spotify, aimed at fast nearest neighbor search (see https://github.com/spotify/annoy#readme)

# In[173]:


get_ipython().run_cell_magic('time', '', '\nembedd_length = 16\nannoy_idx = AnnoyIndex(embedd_length, \'angular\') \n\nfor k in range(arr_ij_csr.shape[0]):\n    sample_csr = arr_ij_csr[k]\n    reshaped_input = sample_csr[np.newaxis, :, :]\n    embedd_vect = embeddings_extractor.predict([reshaped_input])\n    \n    # note: ids must be integers in seq starting from 0 \n    annoy_idx.add_item(k, embedd_vect[0])\n    \n\nfor k2 in range(arr_ij_clustered.shape[0]):\n    sample_clustered = arr_ij_clustered[k2]\n    reshaped_input = sample_clustered[np.newaxis, :, :]\n    embedd_vect = embeddings_extractor.predict([reshaped_input])\n    \n    # note: ids must be integers in seq starting from 0 \n    annoy_idx.add_item(k + k2, embedd_vect[0])\n\n\nnum_trees = 10\nannoy_idx.build(num_trees)\nannoy_idx.save("datasets_demo/embeddings_index.ann")\n')


# Now load it and query the index for the 8 most similar point patterns compared to the busstops pattern:

# In[174]:


help(annoy_idx.get_nns_by_vector)


# In[175]:


get_ipython().run_cell_magic('time', '', 'loaded_annoy_idx = AnnoyIndex(embedd_length, \'angular\')\n\n#loading is fast, will just mmap the file\nloaded_annoy_idx.load("datasets_demo/embeddings_index.ann")\n\nsimilar = loaded_annoy_idx.get_nns_by_vector(\n               vector = embedd_vector_busstops[0],\n               n = 8, \n               search_k = -1, \n               include_distances = True)\n')


# In[176]:


instances_similar =  similar[0]
print(instances_similar)

distances_similar = similar[1]
distances_similar


# Visualize:

# In[178]:


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
        gmi = gmi_csr[idx_pos]
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


# All of them were clustered pattern, as is the busstops pattern. However, the spatial distribution differs.

# ------------------

# # V. 3D visualizations in JavaScript with deck.gl

# In[179]:


def repr_html(html_data, height = 500):
    """Build the HTML representation for Jupyter."""
    srcdoc = html_data.replace('"', "'")
    
    ifr = '''<iframe srcdoc="{srcdoc}" style="width: 100%; height: {h}px; border: none">
             </iframe>'''
    return (ifr.format(srcdoc = srcdoc, h = height))


# The following resources were guidelines for this part:
# - https://www.mapbox.com/mapbox-gl-js/example/3d-buildings/
# - http://deck.gl/showcases/gallery/hexagon-layer
# - https://github.com/uber/deck.gl/blob/master/docs/layers/hexagon-layer.md
# - https://github.com/uber/deck.gl/blob/master/docs/layers/geojson-layer.md
# - https://github.com/uber/deck.gl/blob/master/docs/layers/arc-layer.md
# 

# In[180]:


# MAPBOX_TOKEN = '<THE_MAPBOX_API_TOKEN_HERE>';


# In[181]:


get_ipython().run_cell_magic('bash', '', '\nmkdir -p js/lib\ncd js/lib\nwget https://unpkg.com/s2-geometry@1.2.10/src/s2geometry.js\nmv s2geometry.js s2Geometry.js\nls -alh\n')


# ## V.1. deck.gl Arc, Scatterplot and GeoJSON layers for the route of bus 14

# In[182]:


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


# ## V.2. deck.gl H3Hexagon layer for aggregated counts of busstops

# In[183]:


get_ipython().system('head -n 20 datasets_demo/counts_res9.json')


# In[184]:


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


# ### The end.

# ----

# In[ ]:




