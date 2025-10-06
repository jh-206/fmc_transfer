# Executable module to retrieve and save dead FEMS data
# 
# Main tools fems_api.py modified from fems_wirc.py provided by Angel and Chase


import requests as requests
import pandas as pd
import yaml
import time
import sys
import os
import os.path as osp
import ast

# Set Project Paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Local Modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import fems_api as fems
from utils import read_yml, Dict

# Metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fuels = read_yml(osp.join(CONFIG_DIR, "fuels.yaml")) # Created with fems_api/get_fuel_types()
deads = [k for k in fuels if fuels[k]["category"] == "Dead"]
gaccs = read_yml(osp.join(CONFIG_DIR, "gaccs.yaml")) # BBox from NIFC, copied from wrfxpy rtma_cycler

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def parse_bbox(box_str):
    try:
        # Use ast.literal_eval to safely parse the string representation
        # This will only evaluate literals and avoids security risks associated with eval
        box = ast.literal_eval(box_str)
        # Check if the parsed box is a list and has four elements
        if isinstance(box, list) and len(box) == 4:
            return box
        else:
            raise ValueError("Invalid bounding box format")
    except (SyntaxError, ValueError) as e:
        print("Error parsing bounding box:", e)
        sys.exit(-1)
        return None


# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(f"Usage: {sys.argv[0]} <config_path> <output_path>")
        print("Example: python retrieve_fems_dead.py etc/test_dead.yaml data/test.csv")
        sys.exit(-1)  

    confpath = sys.argv[1]
    outpath = sys.argv[2]
    conf = Dict(read_yml(confpath))
    if not conf.fuelTypes == deads:
        print(f"Warning: fuelTypes in configuration ({conf.fuelTypes}) do not match expected values {deads}. Check for missing or extra fuel categories.")

    # Get sites in config bbox
    if "bbox" in conf:
        conf.update({"bbox": parse_bbox(conf.bbox)})
        stids = fems.get_sites_in_bbox(source="stash", bbox = conf.bbox, stash_path = "data/fems_sts.xlsx")
    else:
        stids = fems.get_all_sites(source="stash", stash_path = "data/fems_sts.xlsx")

    
    # Query over all sites
    conf["siteIds"] = stids.siteId.to_list()
    df = fems.get_fuel_data(
        conf, verbose=True, save_path = outpath
    )

    # Format output and save
    # Subsetting station data columns to join
    df = df.join(pd.json_normalize(df['fuel'])).drop(columns="fuel")
    stids = stids[['longitude', 'latitude', 'elevation', 'timeZone', "siteId", "siteName", "stateId", "slope", "aspect", "rawsId", "raws"]]
    df = df.merge(stids, left_on="site_id", right_on="siteId", how="left")
    print(f"Writing Data to {outpath}")
    os.makedirs(osp.dirname(outpath), exist_ok = True)
    if df.shape[0] > 0:
        df.to_csv(outpath)
    else:
        print(f"No Data retrieved for given input")
