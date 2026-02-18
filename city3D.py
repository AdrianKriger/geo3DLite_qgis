#- geo3D_qgis: 2026
#- arkriger

import os
import json
import processing
from urllib.parse import quote
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkt

from qgis.core import (
    QgsField, QgsProject, QgsDistanceArea, QgsCoordinateTransform, QgsFeatureRequest,
    QgsCoordinateReferenceSystem, QgsGeometry, QgsVariantUtils, QgsVectorLayer, QgsVectorFileWriter,
    QgsLineSymbol, QgsSingleSymbolRenderer, QgsMapLayer, QgsCoordinateTransformContext,
    NULL, QgsField, QgsFeature
)
from qgis.PyQt.QtCore import QEventLoop, QUrl
from qgis.PyQt.QtNetwork import QNetworkAccessManager, QNetworkRequest
from qgis.PyQt.QtGui import QColor

from PyQt5.QtCore import QVariant

from pyproj import CRS

from osgeo import gdal, ogr, osr

def _remove_layer_by_name(name):
    """Finds and removes any existing layer with the same name to prevent duplicates."""
    existing_layers = QgsProject.instance().mapLayersByName(name)
    for layer in existing_layers:
        QgsProject.instance().removeMapLayer(layer.id())
        
def _fetch_overpass(query):
    """Synchronous network fetcher using QGIS-native QNetworkAccessManager."""
    manager = QNetworkAccessManager()
    loop = QEventLoop()
    manager.finished.connect(loop.quit)
    url = f'https://overpass-api.de/api/interpreter?data={quote(query)}'
    reply = manager.get(QNetworkRequest(QUrl(url)))
    loop.exec_()
    
    if reply.error() != 0:
        raise RuntimeError(f"Overpass request failed: {reply.errorString()}")
    return json.loads(reply.readAll().data().decode())

def _parse_to_geojson(raw_data, geom_type="Polygon"):
    """Generic Overpass JSON to GeoJSON converter."""
    geojson = {"type": "FeatureCollection", "features": []}
    
    for el in raw_data.get("elements", []):
        tags = el.get("tags", {})
        
        # --- Handle Ways (Simple Lines or Polygons) ---
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) < 2: continue
            
            if geom_type == "MultiLineString":
                geometry = {"type": "MultiLineString", "coordinates": [coords]}
            elif geom_type == "Polygon":
                if len(coords) < 4: continue
                geometry = {"type": "Polygon", "coordinates": [coords]}
            else: # Default LineString
                geometry = {"type": "LineString", "coordinates": coords}
                
            geojson["features"].append({"type": "Feature", "properties": tags, "geometry": geometry})

        # --- Handle Relations (The Bus Routes) ---
        elif el.get("type") == "relation" and "members" in el:
            if geom_type == "MultiLineString":
                # For bus routes, we want to combine all 'way' members into one MultiLineString
                line_segments = []
                for m in el["members"]:
                    if m.get("type") == "way" and "geometry" in m:
                        line_segments.append([(pt["lon"], pt["lat"]) for pt in m["geometry"]])
                
                if line_segments:
                    geojson["features"].append({
                        "type": "Feature",
                        "properties": tags,
                        "geometry": {"type": "MultiLineString", "coordinates": line_segments}
                    })
            
            elif geom_type == "Polygon":
                # Your existing Multipolygon logic for buildings/parks
                outers = [[(pt["lon"], pt["lat"]) for pt in m["geometry"]] 
                          for m in el["members"] if m.get("role") == "outer" and "geometry" in m]
                inners = [[(pt["lon"], pt["lat"]) for pt in m["geometry"]] 
                          for m in el["members"] if m.get("role") == "inner" and "geometry" in m]
                if outers:
                    geojson["features"].append({
                        "type": "Feature", "properties": tags,
                        "geometry": {"type": "Polygon", "coordinates": outers + inners}
                    })
                    
    return geojson

def overpass2qgis(large, focus, zoom=True):
    """Harvest buildings and add to project."""
    name = f"buildings_{focus}"
    query = (f'[out:json][timeout:180];area[name="{large}"]->.L;area[name="{focus}"](area.L)->.a;(way["building"](area.a);relation["building"]["type"="multipolygon"](area.a););out geom;')
    data = _fetch_overpass(query)
    vlayer = QgsVectorLayer(json.dumps(_parse_to_geojson(data, "Polygon")), name, "ogr")
    final = vlayer.materialize(QgsFeatureRequest())
    
    _remove_layer_by_name(name)
    QgsProject.instance().addMapLayer(final)
    
    if zoom:
        from qgis.utils import iface
        iface.setActiveLayer(final)
        iface.zoomToActiveLayer()
    return final

def get_rgb_color(bld):
    """Returns RGB list as a string to match the original notebook format."""
    if bld in ['house', 'semidetached_house', 'terrace']:
        rgb = [255, 255, 204]
    elif bld == 'apartments':
        rgb = [252, 194, 3]
    elif bld in ['residential', 'dormitory', 'cabin']:
        rgb = [119, 3, 252]
    elif bld in ['garage', 'parking']:
        rgb = [3, 132, 252]
    elif bld in ['retail', 'supermarket']:
        rgb = [253, 141, 60]
    elif bld in ['office', 'commercial']:
        rgb = [185, 206, 37]
    elif bld in ['school', 'kindergarten', 'university', 'college']:
        rgb = [128, 0, 38]
    elif bld in ['clinic', 'doctors', 'hospital']:
        rgb = [89, 182, 178]
    elif bld in ['community_centre', 'service', 'post_office', 'hall', 'civic', 
                  'townhall', 'police', 'library', 'fire_station']:
        rgb = [181, 182, 89]
    elif bld in ['warehouse', 'industrial']:
        rgb = [193, 255, 193]
    elif bld == 'hotel':
        rgb = [139, 117, 0]
    elif bld in ['church', 'mosque', 'synagogue']:
        rgb = [225, 225, 51]
    else:
        rgb = [255, 255, 204]
    return str(rgb)

def get_homebaked_plus_code(lat, lon):
    """Computes 11-digit Plus Code based on the Base20 offset formula."""
    alphabet = "23456789CFGHJMPQRVWX"
    lat_val, lon_val = lat + 90.0, lon + 180.0
    lat_digits, lon_digits = [], []
    l_rem, n_rem = lat_val / 20.0, lon_val / 20.0
    for _ in range(5):
        l_idx = int(l_rem)
        l_rem = (l_rem - l_idx) * 20.0
        lat_digits.append(alphabet[max(0, min(19, l_idx))])
        n_idx = int(n_rem)
        n_rem = (n_rem - n_idx) * 20.0
        lon_digits.append(alphabet[max(0, min(19, n_idx))])
    row, col = int(l_rem * 5 / 20), int(n_rem * 4 / 20)
    grid_idx = (row * 4) + col
    last_digit = alphabet[max(0, min(19, grid_idx))]
    return (f"{lat_digits[0]}{lon_digits[0]}{lat_digits[1]}{lon_digits[1]}"
            f"{lat_digits[2]}{lon_digits[2]}{lat_digits[3]}{lon_digits[3]}+"
            f"{lat_digits[4]}{lon_digits[4]}{last_digit}")

def process3D(layer):
    if not layer or not layer.isValid():
        return None

    # SCHEMA: All numeric outputs are set to String to force the "." decimal
    schema = [
        ('osm_id', QVariant.String), ('address', QVariant.String), 
        ('building', QVariant.String), ('building:levels', QVariant.String), 
        ('building:use', QVariant.String), ('building:flats', QVariant.String), 
        ('building:units', QVariant.String), ('beds', QVariant.String), 
        ('rooms', QVariant.String), ('residential', QVariant.String),
        ('amenity', QVariant.String), ('social_facility', QVariant.String), 
        ('operator', QVariant.String), 
        ('building_height', QVariant.String), 
        ('roof_height', QVariant.String), 
        ('ground_height', QVariant.String), 
        ('bottom_bridge_height', QVariant.String), 
        ('bottom_roof_height', QVariant.String),
        ('plus_code', QVariant.String), ('footprint', QVariant.String), 
        ('geometry_wkt', QVariant.String), ('fill_color', QVariant.String)
    ]

    layer.startEditing()
    existing_names = layer.fields().names()
    
    # Add missing columns
    to_add = [QgsField(n, t) for n, t in schema if n not in existing_names]
    if to_add:
        layer.dataProvider().addAttributes(to_add)
        layer.updateFields()
    
    fields = layer.fields()
    
    # Coordinate Transformer for Plus Codes (WGS84)
    xform = QgsCoordinateTransform(layer.crs(), QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance())

    for feat in layer.getFeatures():
        geom = feat.geometry()
        if not geom or geom.isEmpty(): continue
        if not geom.isGeosValid(): geom = geom.makeValid()

        # --- 1. RAW INPUT SANITIZATION ---
        def to_f(val):
            if QgsVariantUtils.isNull(val) or val == "": return 0.0
            try: return float(str(val).replace(',', '.'))
            except: return 0.0

        b_type = str(feat['building']) if 'building' in existing_names and not QgsVariantUtils.isNull(feat['building']) else 'house'
        levels = to_f(feat['building:levels']) if 'building:levels' in existing_names else 1.0
        ground_h_num = to_f(feat['mean']) if 'mean' in existing_names else 0.0
        min_h = to_f(feat['min_height']) if 'min_height' in existing_names else 0.0

        # --- 2. HEIGHT CALCULATIONS (Floating Point) ---
        storey_h = 2.8
        b_h = round(levels * storey_h + 1.3, 2)
        r_h = round(b_h + ground_h_num, 2)
        bb_h = None
        br_h = None

        if b_type == 'cabin':
            b_h = round(levels * storey_h, 2)
            r_h = round(b_h + ground_h_num, 2)
        elif b_type == 'bridge':
            bb_h = round(min_h + ground_h_num, 2)
        elif b_type == 'roof':
            br_h = round(levels * storey_h + ground_h_num, 2)
            r_h = round(br_h + 1.3, 2)
            b_h = None

        # --- 3. THE FIX: FORCE DOT FORMATTING ---
        def force_dot(val):
            if val is None: return None
            # Converts the number to string and ensures dot is used
            return "{:.2f}".format(float(val)).replace(',', '.')

        # --- 4. ADDRESS & PLUS CODE LOGIC ---
        address_keys = ['name', 'addr:housename', 'addr:flats', 'addr:housenumber', 'addr:street', 'addr:suburb', 'addr:postcode', 'addr:city', 'addr:province']
        address_parts = [str(feat[k]).strip() for k in address_keys if k in existing_names and not QgsVariantUtils.isNull(feat[k])]
        
        pt_geom = geom.pointOnSurface()
        pt_geom.transform(xform)
        p_code = get_homebaked_plus_code(pt_geom.asPoint().y(), pt_geom.asPoint().x())

        # --- 5. APPLY UPDATES ---
        updates = {
            fields.indexFromName('address'): " ".join(address_parts) if address_parts else None,
            fields.indexFromName('plus_code'): p_code,
            fields.indexFromName('building_height'): force_dot(b_h),
            fields.indexFromName('roof_height'): force_dot(r_h),
            fields.indexFromName('ground_height'): force_dot(ground_h_num),
            fields.indexFromName('bottom_bridge_height'): force_dot(bb_h),
            fields.indexFromName('bottom_roof_height'): force_dot(br_h),
            fields.indexFromName('footprint'): json.dumps(json.loads(geom.asJson())['coordinates']),
            fields.indexFromName('geometry_wkt'): geom.asWkt(),
            fields.indexFromName('fill_color'): get_rgb_color(b_type)
        }
        layer.changeAttributeValues(feat.id(), updates)

    if layer.commitChanges():
        layer.triggerRepaint()
        return layer
    return None

def extract_bndrs(input_pbf, focus, zoom=True):
    gdal.UseExceptions()
    
    layer_name = f'aoi_{focus}'
    boundary_name = focus
    geojson_vsimem = "/vsimem/temp_boundary.geojson"

    place_types = ["neighbourhood", "suburb", "quarter", "borough", "village", "town", "city"]
    amenity_list = ["university", "research_institute"]

    def _run_gdal_translate(where_clause):
        try:
            gdal.VectorTranslate(
                geojson_vsimem,
                input_pbf,
                format="GeoJSON",
                layers=["multipolygons"],
                options=["-where", where_clause, "-makevalid"]
            )
            
            # Load the temporary OGR layer
            temp_layer = QgsVectorLayer(geojson_vsimem, "temp", "ogr")
            
            if temp_layer.isValid() and temp_layer.featureCount() > 0:
                # IMPORTANT: Copy features to a permanent Memory Layer
                # This prevents the layer from disappearing when vsimem is unlinked
                mem_layer = temp_layer.materialize(QgsFeatureRequest())
                mem_layer.setName(layer_name)
                return mem_layer
        except Exception as e:
            print(f"Error during extraction: {e}")
            return None
        finally:
            if gdal.VSIStatL(geojson_vsimem):
                gdal.Unlink(geojson_vsimem)
        return None

    _remove_layer_by_name(layer_name)

    # Strategy 1: Places
    place_filter = " OR ".join([f"place = '{p}'" for p in place_types])
    where_place = f"name = '{boundary_name}' AND ({place_filter})"
    final = _run_gdal_translate(where_place)

    # Strategy 2: Amenities
    if final is None:
        amenity_filter = " OR ".join([f"amenity = '{a}'" for a in amenity_list])
        where_amenity = f"name = '{boundary_name}' AND ({amenity_filter})"
        final = _run_gdal_translate(where_amenity)

    if final is None:
        raise RuntimeError(f"No boundary found for '{boundary_name}'")

    # Finalize
    QgsProject.instance().addMapLayer(final)

    if zoom:
        from qgis.utils import iface
        iface.setActiveLayer(final)
        # Ensure the canvas refreshes
        iface.mapCanvas().setExtent(final.extent())
        iface.mapCanvas().refresh()

    return final

def osm_key_to_field(key):
    return (
        key.replace(":", "_")
           .replace("-", "_")
           .lower()
    )

def process_osm_tags_and_ids(layer):
    if not layer:
        return None

    layer.startEditing()
    pr = layer.dataProvider()

    fields = layer.fields()
    field_names = {f.name() for f in fields}

    # Regex for "key"=>"value"
    tag_pattern = re.compile(r'"(.*?)"=>"(.*?)"')

    # ---- discover missing OSM keys ----
    missing_keys = set()

    for f in layer.getFeatures():
        tags = f['other_tags']
        if not tags:
            continue

        for k, v in tag_pattern.findall(tags):
            if k not in field_names:
                missing_keys.add(k)

    # ---- add missing fields (with colons preserved) ----
    if missing_keys:
        pr.addAttributes(
            [QgsField(k, QVariant.String) for k in sorted(missing_keys)]
        )
        layer.updateFields()
        fields = layer.fields()

    # ---- populate fields ----
    for f in layer.getFeatures():
        attrs = {}
        tags = f['other_tags']

        if tags:
            for k, v in tag_pattern.findall(tags):
                idx = fields.indexFromName(k)
                if idx != -1:
                    # only fill if empty
                    if f[k] in (None, "", NULL):
                        attrs[idx] = v

        # ---- osm_id fallback (safe) ----
        if 'osm_id' in field_names and 'osm_way_id' in field_names:
            if not f['osm_id']:
                idx = fields.indexFromName('osm_id')
                attrs[idx] = f['osm_way_id']

        if attrs:
            layer.changeAttributeValues(f.id(), attrs)

    layer.commitChanges()
    return layer

def extract_blds(input_pbf, focus, aoi_layer):
    """
    Extracts buildings within the bounding box, then clips them 
    to the irregular geometry of the aoi_layer.
    """
    gdal.UseExceptions()

    layer_name = f"Buildings_{focus}"
    geojson_vsimem = "/vsimem/temp_buildings.geojson"
    
    # 1. Get extent for the initial GDAL harvesting (fast)
    extent = aoi_layer.extent()
    minx, miny = extent.xMinimum(), extent.yMinimum()
    maxx, maxy = extent.xMaximum(), extent.yMaximum()

    def _run_gdal_translate():
        try:
            gdal.VectorTranslate(
                geojson_vsimem,
                input_pbf,
                format="GeoJSON",
                layers=["multipolygons"],
                options=[
                    "-where", "building IS NOT NULL", 
                    "-makevalid", 
                    "-spat", str(minx), str(miny), str(maxx), str(maxy)
                ]
            )
            
            temp_layer = QgsVectorLayer(geojson_vsimem, "raw_harvest", "ogr")
            
            if temp_layer.isValid() and temp_layer.featureCount() > 0:
                # 2. Clip the harvested buildings to the irregular AOI
                # This removes buildings in the "corners" of the bounding box
                clipped_result = processing.run(
                    "native:clip",
                    {
                        'INPUT': temp_layer,
                        'OVERLAY': aoi_layer,
                        'OUTPUT': 'memory:'
                    }
                )
                
                final_layer = clipped_result['OUTPUT']
                final_layer.setName(layer_name)
                return final_layer
                
        except Exception as e:
            print(f"Error during building extraction/clip: {e}")
            return None
        finally:
            if gdal.VSIStatL(geojson_vsimem):
                gdal.Unlink(geojson_vsimem)
        return None

    # ---- clean up existing layer to prevent duplicates ----
    _remove_layer_by_name(layer_name)

    # ---- Execute extraction and clipping ----
    final_blds = _run_gdal_translate()
    final_blds = process_osm_tags_and_ids(final_blds)

    if final_blds is None or final_blds.featureCount() == 0:
        print(f"No buildings found inside the irregular AOI for '{focus}'.")
        return None

    # ---- add to QGIS ----
    QgsProject.instance().addMapLayer(final_blds)

    return final_blds

def harvest_osm_to_qgis(input_pbf, layer_type, sql_where, aoi_layer, base_name, focus):
    """
    Generic QGIS Harvester for geo3DLite.
    Names layers as: BaseName_Focus (e.g., Buildings_CapeTown)
    """
    gdal.UseExceptions()
    
    # 1. Standardized Naming
    target_layer_name = f"{base_name}_{focus}" 
    geojson_vsimem = f"/vsimem/temp_{target_layer_name}.geojson"
    
    # Clean up existing layer to prevent duplicates
    _remove_layer_by_name(target_layer_name)

    # 2. Extent for GDAL
    ext = aoi_layer.extent()
    options = [
        "-where", sql_where,
        "-makevalid",
        "-spat", str(ext.xMinimum()), str(ext.yMinimum()), 
        str(ext.xMaximum()), str(ext.yMaximum())
    ]

    try:
        # 3. GDAL Extract
        gdal.VectorTranslate(geojson_vsimem, input_pbf, format="GeoJSON", 
                             layers=[layer_type], options=options)
        
        raw_layer = QgsVectorLayer(geojson_vsimem, "raw_harvest", "ogr")
        
        if not raw_layer.isValid() or raw_layer.featureCount() == 0:
            return None

        # 4. Irregular Clip
        clipped_result = processing.run("native:clip", {
            'INPUT': raw_layer,
            'OVERLAY': aoi_layer,
            'OUTPUT': 'memory:'
        })
        
        final_layer = clipped_result['OUTPUT']
        final_layer.setName(target_layer_name)

        # 5. Tag/ID Parsing
        final_layer = process_osm_tags_and_ids(final_layer)

        # 6. Add to Project
        QgsProject.instance().addMapLayer(final_layer)
        return final_layer

    except Exception as e:
        print(f"Error harvesting {target_layer_name}: {e}")
        return None
    finally:
        if gdal.VSIStatL(geojson_vsimem):
            gdal.Unlink(geojson_vsimem)


def harvest_water_to_qgis(input_pbf, aoi_layer, focus):
    """
    Merges Multipolygons and Lines into one true Temporary Scratch 'Water' layer.
    """
    target_name = f"Water_{focus}"
    _remove_layer_by_name(target_name)

    # 1. Harvest silently (No project addition)
    # Ensure harvest_osm_to_qgis is updated with the add_to_project=False flag
    poly_layer = harvest_osm_to_qgis(
        input_pbf, "multipolygons", "natural='water' OR landuse='reservoir'", 
        aoi_layer, "temp_poly", focus, add_to_project=False
    )

    line_layer = harvest_osm_to_qgis(
        input_pbf, "lines", "waterway IS NOT NULL", 
        aoi_layer, "temp_line", focus, add_to_project=False
    )

    layers_to_merge = [l for l in [poly_layer, line_layer] if l is not None]
    if not layers_to_merge:
        return None

    # 2. Perform the Merge
    # We merge to memory, but then we "Cast" it to a fresh Scratch layer
    merge_result = processing.run("native:mergevectorlayers", {
        'LAYERS': layers_to_merge,
        'OUTPUT': 'memory:'
    })
    merged_data = merge_result['OUTPUT']

    # 3. Create the ACTUAL Temporary Scratch Layer
    # This URI format is what QGIS uses for 'Scratch' layers
    uri = f"Polygon?crs={aoi_layer.crs().authid()}"
    water_scratch = QgsVectorLayer(uri, target_name, "memory")
    provider = water_scratch.dataProvider()

    # Copy fields and features from the merge result
    provider.addAttributes(merged_data.fields())
    water_scratch.updateFields()
    
    # Efficiently add features
    features = [f for f in merged_data.getFeatures()]
    provider.addFeatures(features)

    # 4. Add to Project
    QgsProject.instance().addMapLayer(water_scratch)
    
    return water_scratch

def q_farmland(large, focus):
    """Harvest landuse=farmland and add to project."""
    name = f"farmland_{focus}"
    query = (f'[out:json][timeout:180];area[name="{large}"]->.L;area[name="{focus}"](area.L)->.a;(way["landuse"="farmland"](area.a);relation["landuse"="farmland"]["type"="multipolygon"](area.a););out geom;')
    data = _fetch_overpass(query)
    vlayer = QgsVectorLayer(json.dumps(_parse_to_geojson(data, "Polygon")), name, "ogr")
    final = vlayer.materialize(QgsFeatureRequest())
    
    _remove_layer_by_name(name)
    # Check if dataset is not empty before adding
    if final.featureCount() > 0:
        QgsProject.instance().addMapLayer(final)
        return final
    else:
        print(f"Skipped {name}: No features found.")
        return None
        
def q_green_spaces(large, focus):
    """Harvest leisure areas and add to project."""
    name = f"greenSpace_{focus}"
    query = (f'[out:json][timeout:180];area[name="{large}"]->.L;area[name="{focus}"](area.L)->.a;(way["leisure"~"park|track|pitch"](area.a);relation["leisure"~"park|track|pitch"]["type"="multipolygon"](area.a););out geom;')
    data = _fetch_overpass(query)
    vlayer = QgsVectorLayer(json.dumps(_parse_to_geojson(data, "Polygon")), name, "ogr")
    final = vlayer.materialize(QgsFeatureRequest())
    
    _remove_layer_by_name(name)
    # Check if dataset is not empty before adding
    if final.featureCount() > 0:
        QgsProject.instance().addMapLayer(final)
        return final
    else:
        print(f"Skipped {name}: No features found.")
        return None

def q_water(large, focus):
    """Harvest water features, buffer lines, and return a single Polygon scratch layer."""
    name = f"water_{focus}"
    _remove_layer_by_name(name)

    query = (f'[out:json][timeout:180];area[name="{large}"]->.L;'
             f'area[name="{focus}"](area.L)->.a;'
             f'(way["water"](area.a);'
             f'way["waterway"](area.a);'
             f'relation["water"]["type"="multipolygon"](area.a););out geom;')
    
    data = _fetch_overpass(query)
    if not data or not data.get('elements'):
        return None

    # 1. Parse everything as raw features first
    # Use 'LineString' to capture the actual geometry of the ways
    raw_geojson = _parse_to_geojson(data, "LineString")
    gdf_raw = gpd.GeoDataFrame.from_features(raw_geojson['features'], crs="EPSG:4326")

    # 2. Separate by Geometry Type
    # This ensures we don't treat a river as a polygon unless it's a closed loop
    gdf_line = gdf_raw[gdf_raw.geometry.type == 'LineString'].copy()
    gdf_poly = gdf_raw[gdf_raw.geometry.type == 'Polygon'].copy()

    # 3. Buffer ONLY the lines
    # This turns the 32 lines into real water surfaces without 'snapping' start to end
    if not gdf_line.empty:
        # 0.00005 degrees is approx 1m width
        gdf_line['geometry'] = gdf_line.geometry.buffer(0.00001)

    # 4. Merge them back together
    # Now both are Polygons, but the rivers are the correct shape
    gdf_water = pd.concat([gdf_poly, gdf_line], ignore_index=True)

    if gdf_water.empty:
        return None

    # 5. Create the TRUE Temporary Scratch Layer in QGIS
    uri = f"Polygon?crs=EPSG:4326"
    water_layer = QgsVectorLayer(uri, name, "memory")
    provider = water_layer.dataProvider()

    # Add attributes from GDF (excluding geometry)
    cols = [c for c in gdf_water.columns if c != 'geometry']
    provider.addAttributes([QgsField(str(n), QVariant.String) for n in cols])
    water_layer.updateFields()

    # Convert features
    features = []
    for _, row in gdf_water.iterrows():
        if row.geometry and not row.geometry.is_empty:
            fet = QgsFeature()
            fet.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
            fet.setAttributes([str(row[n]) if pd.notnull(row[n]) else None for n in cols])
            features.append(fet)

    provider.addFeatures(features)
    QgsProject.instance().addMapLayer(water_layer)
    
    return water_layer

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

def q_Troutes(large, operator='MyCiTi'):
    """Harvest bus routes and apply original RGB tuple conversion."""
    name = f"transit{operator}"
    query = (f'[out:json][timeout:180];area[name="{large}"];(relation["type"="route"]["route"="bus"]["operator"="{operator}"]["colour"](area););out geom;')
    
    data = _fetch_overpass(query)
    geojson = _parse_to_geojson(data, "MultiLineString")

    # Apply your hex_to_rgb conversion logic directly
    for feat in geojson["features"]:
        if "colour" in feat["properties"]:
            try:
                # Store the tuple directly as requested
                feat["properties"]["colour"] = hex_to_rgb(feat["properties"]["colour"])
            except Exception:
                feat["properties"]["colour"] = (255, 0, 0) # Fallback
            
    vlayer = QgsVectorLayer(json.dumps(geojson), name, "ogr")
    final = vlayer.materialize(QgsFeatureRequest())
    
    _remove_layer_by_name(name)
    # Check if dataset is not empty before adding
    if final.featureCount() > 0:
        QgsProject.instance().addMapLayer(final)
        return final
    else:
        print(f"Skipped {name}: No features found.")
        return None

def layer_to_geojson_dict(layer):
    """Converts a QGIS layer to a GeoJSON-style dictionary in WGS84."""
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GeoJSON"
    # Ensure coordinates are in WGS84 for MapLibre
    dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    
    # Create a temporary file path
    temp_path = os.path.join(os.path.expanduser("~"), "temp_layer.geojson")
    QgsVectorFileWriter.writeAsVectorFormatV3(layer, temp_path, QgsProject.instance().transformContext(), options)
    
    with open(temp_path, 'r') as f:
        data = json.load(f)
    
    os.remove(temp_path)
    return data


def create_3Dviz(
    result_dir, 
    buildings_layer, 
    roads_layer=None, 
    green_layer=None, 
    water_layer=None, 
    bus_layer=None, 
    offline=False,
    local_js_path="./data/maplibre-gl.js", 
    local_css_path="./data/maplibre-gl.css"
):
    """
    Creates a 3D MapLibre visualization. 
    If offline=True, it embeds local JS/CSS and uses a solid dark background.
    """
    if offline == False:
        html_path = os.path.join(result_dir, "interactiveOnly.html")
    else:
        html_path = os.path.join(result_dir, "interactiveAlt.html")
    #html_path = os.path.join(result_dir, f"geo3D_{focus}.html")
    
    # 1. Convert QGIS Layers to GeoJSON Dicts
    # Assumes layer_to_geojson_dict is available in your namespace
    building_data = layer_to_geojson_dict(buildings_layer)
    road_data = layer_to_geojson_dict(roads_layer) if roads_layer else {"type": "FeatureCollection", "features": []}
    green_data = layer_to_geojson_dict(green_layer) if green_layer else {"type": "FeatureCollection", "features": []}
    water_data = layer_to_geojson_dict(water_layer) if water_layer else {"type": "FeatureCollection", "features": []}
    bus_data = layer_to_geojson_dict(bus_layer) if bus_layer else {"type": "FeatureCollection", "features": []}

    # 2. Map View Settings
    extent = buildings_layer.extent()
    center_coords = [(extent.xMinimum() + extent.xMaximum()) / 2, (extent.yMinimum() + extent.yMaximum()) / 2]

    # 3. Asset & Style Logic
    if offline:
        try:
            with open(local_js_path, 'r', encoding='utf-8') as f:
                js_content = f"<script>{f.read()}</script>"
            with open(local_css_path, 'r', encoding='utf-8') as f:
                css_content = f"<style>{f.read()}</style>"
        except FileNotFoundError:
            js_content = '<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>'
            css_content = '<link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />'

        style_js = json.dumps({
            "version": 8,
            "sources": {},
            "layers": [{"id":"bg","type":"background","paint":{"background-color":"#0e0e0e"}}]
        })
    else:
        js_content = '<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>'
        css_content = '<link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />'
        style_js = "'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'"
        
    # 4. Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>geo3DLite (QGIS). 3D City Models for Geography and Sustainable Development Education</title>
    {css_content}
    {js_content}
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; background: #0e0e0e; }}
        .popup-content {{ font-family: sans-serif; font-size: 12px; line-height: 1.5; color: #333; }}
    </style>
</head>
<body>
<div id="map"></div>
<script>
const map = new maplibregl.Map({{
    container: 'map',
    style: {style_js},
    center: {json.dumps(center_coords)},
    zoom: 16,
    pitch: 60,
    antialias: true
}});

map.on('load', () => {{
    map.addControl(new maplibregl.NavigationControl({{ visualizePitch: true }}), 'top-right');

    // Add Sources
    map.addSource('roads', {{ type: 'geojson', data: {json.dumps(road_data)} }});
    map.addSource('water', {{ type: 'geojson', data: {json.dumps(water_data)} }});
    map.addSource('green', {{ type: 'geojson', data: {json.dumps(green_data)} }});
    map.addSource('bus', {{ type: 'geojson', data: {json.dumps(bus_data)} }});
    map.addSource('buildings', {{ type: 'geojson', data: {json.dumps(building_data)} }});

    // 1. Water
    map.addLayer({{
        id: 'water-layer', type: 'fill', source: 'water',
        paint: {{ 'fill-color': '#01579b', 'fill-opacity': 0.8 }}
    }});

    // 2. Green Space
    map.addLayer({{
        id: 'green-layer', type: 'fill', source: 'green',
        paint: {{ 'fill-color': '#2e7d32', 'fill-opacity': 0.4 }}
    }});

    // 3. Road Hierarchy (Dark Matter Style)
    const roadLayers = [
        {{ id: 'road-motorway', filter: ['==', 'highway', 'motorway'], color: '#666666', width: [8, 1, 14, 6] }},
        {{ id: 'road-primary', filter: ['==', 'highway', 'primary'], color: '#8b949e', width: [8, 0.75, 14, 4] }},
        {{ id: 'road-secondary', filter: ['==', 'highway', 'secondary'], color: '#6e7681', width: [9, 0.5, 14, 3] }},
        {{ id: 'road-tertiary', filter: ['==', 'highway', 'tertiary'], color: '#5a5f66', width: [10, 0.4, 14, 2.5] }},
        {{ id: 'road-residential', filter: ['==', 'highway', 'residential'], color: '#444c56', width: [11, 0.3, 16, 2] }},
        {{ id: 'road-service', filter: ['in', 'highway', 'service', 'track', 'minor', 'motorway_link'], color: '#363b42', width: [12, 0.25, 16, 1] }}
    ];

    roadLayers.forEach(layer => {{
        map.addLayer({{
            'id': layer.id, 'type': 'line', 'source': 'roads',
            'filter': layer.filter,
            'paint': {{
                'line-color': layer.color,
                'line-width': ['interpolate', ['linear'], ['zoom'], ...layer.width]
            }}
        }});
    }});

    // 4. Bus/BRT Routes (Focus Layer)
    map.addLayer({{
        id: 'bus-layer', type: 'line', source: 'bus',
        layout: {{ 'line-join': 'round', 'line-cap': 'round' }},
        paint: {{
            'line-color': ['case', ['has', 'colour'], ['get', 'colour'], '#FF4500'],
            'line-width': 4
        }}
    }});

    // 5. 3D Buildings
    map.addLayer({{
        id: '3d-buildings', type: 'fill-extrusion', source: 'buildings',
        paint: {{
            'fill-extrusion-color': ['case', ['has', 'fill_color'], ['get', 'fill_color'], '#aaaaaa'],
            'fill-extrusion-height': ['coalesce', ['to-number', ['get', 'building_height']], 10],
            'fill-extrusion-base': 0,
            'fill-extrusion-opacity': 0.8
        }}
    }});

    // Click Interaction
    map.on('click', '3d-buildings', (e) => {{
        const p = e.features[0].properties;
        const content = '<div class="popup-content">' +
                '<strong>Building:</strong> ' + (p.building || 'N/A') + '<br><hr>' +
                '<strong>Address:</strong> ' + (p.address || 'N/A') + '<br>' +
                '<strong>Height:</strong> ' + (p.building_height || 0) + 'm<br>' +
                '<strong>Plus Code:</strong> ' + (p.plus_code || 'N/A') +
            '</div>';
        new maplibregl.Popup().setLngLat(e.lngLat).setHTML(content).addTo(map);
    }});
}});
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return html_path
    
def get_utm_crs(gdf):
    # 1. Calculate center point to determine UTM zone
    # We use WGS84 coordinates for the calculation
    wgs84_gdf = gdf.to_crs("EPSG:4326")
    avg_lon = wgs84_gdf.geometry.centroid.x.mean()
    avg_lat = wgs84_gdf.geometry.centroid.y.mean()
    
    utm_zone = int((avg_lon + 180) / 6) + 1
    # 326xx for North, 327xx for South
    epsg_prefix = 32600 if avg_lat >= 0 else 32700
    epsg_code = epsg_prefix + utm_zone
    
    # 2. Initialize the CRS object from the EPSG
    utm_crs = CRS.from_epsg(epsg_code)
    
    # 3. Print the full detailed report
    # Using 'print(utm_crs)' in some environments only shows the code.
    # To force the full report, we can use the __repr__ or specifically format it.
    #print("-" * 30)
    print(utm_crs.to_string()) # This usually gives the detailed block
    #print("-" * 30)
    
    # If the above is still short, this is the guaranteed full metadata:
    # (Matches the exact printout you requested)
    return epsg_code

def q_solar(large, focus):
    """Harvest solar (power=generator) and add to project."""
    name = f"solar_{focus}"
    query = (f'[out:json][timeout:180];area[name="{large}"]->.L;area[name="{focus}"](area.L)->.a;(way["power"="generator"]["generator:source"="solar"](area.a););out geom;')
    data = _fetch_overpass(query)
    vlayer = QgsVectorLayer(json.dumps(_parse_to_geojson(data, "Polygon")), name, "ogr")
    final = vlayer.materialize(QgsFeatureRequest())
    
    _remove_layer_by_name(name)
    QgsProject.instance().addMapLayer(final)
    return final

def read_vsimem_geojson(vsimem_path):
    """
    Read a GDAL /vsimem GeoJSON into a standard GeoPandas GeoDataFrame.
    """
    gdf = gpd.read_file(vsimem_path)
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf

def _harvestSolar(input_pbf, focus, aoi_layer, epsg):
    """
    Harvests solar data and CLIPS it to the AOI.
    Returns: GeoDataFrame in projected CRS (epsg).
    Displays: QGIS Memory Layer in WGS84 (EPSG:4326).
    """
    gdal.UseExceptions()
    gdal.SetConfigOption("OGR_GEOMETRY_ACCEPT_UNCLOSED_RING", "NO")
    gdal.SetConfigOption("OGR_INTERLEAVED_READING", "YES")

    sql_where_solar_generator = """
        other_tags LIKE '%"power"=>"generator"%'
        AND other_tags LIKE '%"generator:source"=>"solar"%'
    """

    # Get extent for GDAL harvesting
    extent = aoi_layer.extent()
    minx, miny = extent.xMinimum(), extent.yMinimum()
    maxx, maxy = extent.xMaximum(), extent.yMaximum()
    
    all_solar_gdfs = []

    # --- 1. Process Multipolygons ---
    geojson_poly = "/vsimem/solar_multipolygons.geojson"
    gdal.VectorTranslate(
        geojson_poly, input_pbf, format="GeoJSON", layers=["multipolygons"],
        options=["-where", sql_where_solar_generator, "-makevalid",
                 "-spat", str(minx), str(miny), str(maxx), str(maxy)]
    )
    gdf_poly = read_vsimem_geojson(geojson_poly)
    if not gdf_poly.empty: all_solar_gdfs.append(gdf_poly)

    # --- 2. Process Lines ---
    geojson_lines = "/vsimem/solar_lines.geojson"
    gdal.VectorTranslate(
        geojson_lines, input_pbf, format="GeoJSON", layers=["lines"],
        options=["-where", sql_where_solar_generator, "-makevalid",
                 "-spat", str(minx), str(miny), str(maxx), str(maxy), "-nlt", "POLYGON"]
    )
    gdf_lines = read_vsimem_geojson(geojson_lines)
    if not gdf_lines.empty: all_solar_gdfs.append(gdf_lines)

    # --- 3. Combine, Clip, and Cleanup ---
    if not all_solar_gdfs:
        gdal.Unlink(geojson_poly)
        gdal.Unlink(geojson_lines)
        return gpd.GeoDataFrame(geometry=[], crs=epsg)

    # Base data in WGS84
    gdf_wgs84_raw = gpd.GeoDataFrame(pd.concat(all_solar_gdfs, ignore_index=True), crs="EPSG:4326")

    # SPATIAL CLIP LOGIC
    # Extract QGIS AOI geometry as Shapely object for GeoPandas
    # We use the union of all features in the AOI layer just in case it has multiple parts
    aoi_geom_wkt = QgsGeometry.unaryUnion([f.geometry() for f in aoi_layer.getFeatures()]).asWkt()
    aoi_shapely = shapely.wkt.loads(aoi_geom_wkt)
    
    # Clip the harvested data to the AOI boundary
    gdf_wgs84 = gpd.clip(gdf_wgs84_raw, aoi_shapely).reset_index(drop=True)

    gdal.Unlink(geojson_poly)
    gdal.Unlink(geojson_lines)

    # If clipping removed everything, return empty
    if gdf_wgs84.empty:
        return gpd.GeoDataFrame(geometry=[], crs=epsg)

    # --- 4. Add to QGIS Map View (WGS84) ---
    layer_name = f"solar_{focus}"
    _remove_layer_by_name(layer_name)

    uri = "Polygon?crs=EPSG:4326"
    temp_layer = QgsVectorLayer(uri, layer_name, "memory")
    provider = temp_layer.dataProvider()

    cols = [column for column in gdf_wgs84.columns if column != 'geometry']
    provider.addAttributes([QgsField(str(name), QVariant.String) for name in cols])
    temp_layer.updateFields()

    features = []
    for _, row in gdf_wgs84.iterrows():
        if row.geometry:
            fet = QgsFeature()
            fet.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
            # Handle QVariant-like issues by ensuring standard Python types
            fet.setAttributes([str(row[name]) if pd.notnull(row[name]) else None for name in cols])
            features.append(fet)

    provider.addFeatures(features)
    QgsProject.instance().addMapLayer(temp_layer)

    # --- 5. Return projected GeoDataFrame (UTM/EPSG) ---
    gdf_projected = gdf_wgs84.to_crs(epsg)
    
    return gdf_projected

def calculate_azimuth_from_geometry(polygon):
    """
    Calculates the azimuth (angle from North, clockwise, 0-180) 
    of the minimum rotated bounding box for a given Shapely Polygon.
    """
    if not polygon or polygon.geom_type not in ['Polygon', 'MultiPolygon']:
        return 0.0

    min_rect = polygon.minimum_rotated_rectangle
    
    if min_rect.geom_type != 'Polygon':
        return 0.0
        
    coords = np.array(min_rect.exterior.coords)
    
    segment1 = coords[1] - coords[0]
    segment2 = coords[2] - coords[1]
    
    len1 = np.linalg.norm(segment1)
    len2 = np.linalg.norm(segment2)

    long_segment = segment1 if len1 >= len2 else segment2
        
    if np.linalg.norm(long_segment) == 0:
        return 0.0

    angle_rad = np.arctan2(long_segment[1], long_segment[0])
    angle_deg = np.degrees(angle_rad)
    
    # Convert angle (from X-axis CCW) to Azimuth (from North CW)
    azimuth = 90.0 - angle_deg
    azimuth = azimuth % 360.0
        
    # Constrain to 0-180 range
    if azimuth > 180.0:
        azimuth -= 180.0

    return azimuth

def _with_solar(gdf_buildings, gdf_solar):
    """
    Efficient Dual Join: Performs both building-centric and solar-centric joins
    in one loop.

    Returns: (gdf_buildings_modified, gdf_solar_modified)
    """

    n_bld = len(gdf_buildings["geometry"])
    n_sol = len(gdf_solar["geometry"])
    
    # CRITICAL: IDs for both DataFrames
    BLD_ID_COLUMN = "osm_id"  # Assuming 'osm_id' is the unique ID in the building layer
    SOLAR_ID_COLUMN = "osm_id"
    
    # --- BUILDING-CENTRIC OUTPUT (for blds.df) ---
    solar_id_lists = [[] for _ in range(n_bld)]  # List of solar IDs for each building
    solar_m = [[] for _ in range(n_bld)]
    has_solar = [False] * n_bld

    # --- SOLAR-CENTRIC OUTPUT (for gdf_solar.df) ---
    bld_id_lists = [[] for _ in range(n_sol)]  # List of building IDs for each solar panel
    #bld_id_lists =[None] * n_sol
    
    # Brute-force intersection
    for i in range(n_bld):
        b_geom = gdf_buildings["geometry"].iloc[i]
        bld_id = gdf_buildings[BLD_ID_COLUMN].iloc[i] # Get the building ID

        for j in range(n_sol):
            s_geom = gdf_solar["geometry"].iloc[j]
            sol_id = gdf_solar[SOLAR_ID_COLUMN].iloc[j] # Get the solar ID
            s_m = gdf_solar['generator:method'].iloc[j] # Get the building ID

            if b_geom.contains(s_geom):
                # 1. Building-Centric Logic (Attaches solar ID to building)
                has_solar[i] = True
                solar_id_lists[i].append(sol_id)
                solar_m[i].append(s_m)

                # 2. Solar-Centric Logic (Attaches building ID to solar panel)
                bld_id_lists[j].append(bld_id)
                #if bld_id_lists[j] is None:
                #    bld_id_lists[j] = bld_id

    # --- Finalize Lists (Replace [] with None) ---
    for i in range(n_bld):
        if not solar_id_lists[i]:
            solar_id_lists[i] = None

    for j in range(n_sol):
        if not bld_id_lists[j]:
            bld_id_lists[j] = None

    # --- CREATE OUTPUT DataFrames ---

    # 1. Modified Building DataFrame (blds)
    #gdf_blds_out = dict(gdf_buildings)
    gdf_buildings["children"] = solar_id_lists 
    gdf_buildings["has_solar"] = has_solar
    gdf_buildings["method"] = solar_m
    #blds_out = GeoDataFrameLite(gdf_blds_out)
    #blds_out.crs = epsg

    # 2. Modified Solar DataFrame (gdf_solar)
    # We must use the solar DataFrame as the base to keep all geometry/attributes
    #gdf_solar_out = dict(gdf_solar)
    # The new column holds the list of intersecting building IDs
    gdf_solar["parent"] = bld_id_lists 
    #gdf_solar_out = GeoDataFrameLite(gdf_solar_out)
    #gdf_solar_out.crs = epsg
    gdf_solar = gdf_solar.rename(columns={'generator:method': 'method'})
    gdf_solar['area'] = gdf_solar['geometry'].apply(lambda geom: geom.area)
    gdf_solar['azimuth'] = gdf_solar['geometry'].apply(calculate_azimuth_from_geometry)

    return gdf_buildings, gdf_solar

def save_to_geopackage(gpkg_path, target_crs_string):
    if os.path.exists(gpkg_path):
        try:
            os.remove(gpkg_path)
            print(f"Cleared existing file: {gpkg_path}")
        except Exception as e:
            print(f"File locked: {e}")
            return

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.destCRS = QgsCoordinateReferenceSystem(target_crs_string)
    
    # Use the actual project context to handle CRS transformations correctly
    context = QgsProject.instance().transformContext()
    
    # Flag to track if the GPKG file has been created yet
    file_created = False
    
    for layer in QgsProject.instance().mapLayers().values():
        # Only process vector layers
        if layer.type() == QgsMapLayer.VectorLayer:
            options.layerName = layer.name().replace(" ", "_").lower()
            
            if not file_created:
                # The first valid vector layer creates the actual .gpkg file
                options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
                file_created = True
            else:
                # Every subsequent vector layer adds a new table (layer) to that file
                options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
            
            error, message = QgsVectorFileWriter.writeAsVectorFormatV2(
                layer, gpkg_path, context, options
            )
            
            if error == QgsVectorFileWriter.NoError:
                print(f" Exported: {layer.name()}")
            else:
                print(f" Failed {layer.name()}: {message}")