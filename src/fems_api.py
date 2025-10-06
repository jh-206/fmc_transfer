# Set of tools to query FEMS API for fuel samples
# Modified from fems_wirc.py provided by Angel and Chase

import requests as requests
import pandas as pd
import yaml
import time
import os.path as osp

# General API 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
API_URL = "https://fems.fs2c.usda.gov/fuelmodel/apis/graphql"

SITES_QUERY_VARS = {
    "returnAll": "false",
    "sortBy": "site_name",
    "sortOrder": "desc",
    "page": 0,
    "perPage": 1000000,
}


def fuelQueryVars(page, fuel_params):
    startDate = fuel_params["startDate"]
    endDate = fuel_params["endDate"]
    siteId = fuel_params["siteId"]
    fuelType = fuel_params["fuelType"]
    return {
        "endDate": endDate,
        "filterByCategory": "All",
        "filterByMethod": "All",
        "filterBySampleType": fuelType,
        "filterByStatus": "All",
        "filterBySubCategory": "All",
        "page": page,
        "perPage": 100000,
        "returnAll": "false",
        "siteId": siteId,
        "sortBy": "date_time",
        "sortOrder": "desc",
        "startDate": startDate,
    }

def api_request_headers(access_token):
    headers = {"User-Agent": ""}
    if access_token != None:
        auth_header = f"Bearer {access_token}"
        headers.update({"Authorization": auth_header})
    return headers

FUELS_PAGES_QUERY = """
    query GetFuelSamples(
      $returnAll: Boolean!,
      $sampleId: Int,
      $siteId: String,
      $startDate: DateTime
      $endDate: DateTime
      $filterBySampleType: String,
      $filterByStatus: String,
      $filterByCategory: String,
      $filterBySubCategory: String,
      $filterByMethod: String,
      $sortBy: String,
      $sortOrder: String,
      $page: Int,
      $perPage: Int,
      ) {
           getFuelSamples(
                   returnAll: $returnAll
                   sampleId: $sampleId
                   siteId: $siteId
                   startDate: $startDate
                   endDate: $endDate
                   filterBySampleType: $filterBySampleType
                   filterByStatus: $filterByStatus
                   filterByCategory: $filterByCategory
                   filterBySubCategory: $filterBySubCategory
                   filterByMethod: $filterByMethod
                   sortBy: $sortBy
                   sortOrder: $sortOrder
                   page: $page
                   perPage: $perPage
                    
               ) {
                       pageInfo{
                               page
                               per_page
                               page_count
                               total_count
                           }
                       
                   }
             }
"""

# Query for the station and fuel moisture data from the FEMS API
FUELS_QUERY = """
    query GetFuelSamples(
      $returnAll: Boolean!,
      $sampleId: Int,
      $siteId: String,
      $startDate: DateTime
      $endDate: DateTime
      $filterBySampleType: String,
      $filterByStatus: String,
      $filterByCategory: String,
      $filterBySubCategory: String,
      $filterByMethod: String,
      $sortBy: String,
      $sortOrder: String,
      $page: Int,
      $perPage: Int,
      ) {
           getFuelSamples(
                   returnAll: $returnAll
                   sampleId: $sampleId
                   siteId: $siteId
                   startDate: $startDate
                   endDate: $endDate
                   filterBySampleType: $filterBySampleType
                   filterByStatus: $filterByStatus
                   filterByCategory: $filterByCategory
                   filterBySubCategory: $filterBySubCategory
                   filterByMethod: $filterByMethod
                   sortBy: $sortBy
                   sortOrder: $sortOrder
                   page: $page
                   perPage: $perPage
                    
               ) {
                       pageInfo{
                               page
                               per_page
                               page_count
                               total_count
                           }
                       fuelSamples {
                             fuel_sample_id
                             site_id
                             fuel{
                                   fuel_type 
                                   category
                                 }
                             sub_category
                             sample
                             subSampleCount
                             method_type
                             status
                             sample_average_value
                           }
                   }
             }
"""

# Site-Related Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_site_fields(api_url: str = API_URL, verbose=False) -> list[str]:
    """
    Retrieve all available field names from the SiteDTO type
    via GraphQL introspection.
    """
    introspection_query = """
    query {
      __type(name: "SiteDTO") {
        fields {
          name
        }
      }
    }
    """
    if verbose:
        print(f"Getting all available field names from url: {api_url}")
        print(f"Query: {introspection_query}")
    resp = requests.post(api_url, json={"query": introspection_query})
    resp.raise_for_status()
    fields = [f["name"] for f in resp.json()["data"]["__type"]["fields"]]
    fields.remove("fuelSampleMetadata")
    if verbose:
        print(f"Response: {resp}")
        print(f"Number of Fields: {len(fields)}")
    
    return fields

def build_sites_query(fields: list[str]) -> str:
    """
    Build a GraphQL query string for getSites using a dynamic list of fields.
    """
    fields_str = "\n                ".join(fields)
    return f"""
    query GetSites(
        $returnAll: Boolean!,
        $groupId: Int,
        $siteId: String,
        $sortBy: String,
        $sortOrder: String,
        $page: Int,
        $perPage: Int) {{
        getSites(
            returnAll: $returnAll,
            groupId: $groupId,
            siteId: $siteId,
            sortBy: $sortBy,
            sortOrder: $sortOrder,
            page: $page,
            perPage: $perPage
        ) {{
            pageInfo {{
                page
                per_page
                page_count
                total_count
            }}
            sites {{
                {fields_str}
            }}
        }}
    }}
    """


def get_all_sites(fields = None, access_token=None, verbose=False, stash_path = None):
    """
    Request all station data from the FEMS API

    Parameters
    ----------
    access_token: str
        Access token for the FEMS API (optional)

    Returns
    ----------
    Array of sites from FEMS API
    """
    print(f"Retrieving all FEMS sampling sites.")
    request_headers = api_request_headers(access_token)
    if fields is None:
        fields = get_site_fields(verbose=verbose)
    st_query = build_sites_query(fields)
    q_vars = {**SITES_QUERY_VARS, "returnAll": True}
    request_json = {
        "query": st_query,
        "variables": q_vars,
    }
    response = requests.post(url=API_URL, json=request_json, headers=request_headers)
    sites = pd.DataFrame(response.json()["data"]["getSites"]["sites"])

    if stash_path is not None:
        print(f"Saving stations to {stash_path}")
        sites.to_excel(stash_path)

    return sites

def get_single_site(siteId, fields=None, access_token=None, verbose=False):
    """
    Request all station data from the FEMS API

    Parameters
    ----------
    access_token: str
        Access token for the FEMS API (optional)

    Returns
    ----------
    Array of sites from FEMS API
    """
    request_headers = api_request_headers(access_token)
    if fields is None:
        fields = get_site_fields(verbose=verbose)
    st_query = build_sites_query(fields)
    q_vars = {**SITES_QUERY_VARS, "returnAll": False, "siteId": str(siteId)}
    request_json = {
        "query": st_query,
        "variables": q_vars,
    }
    response = requests.post(url=API_URL, json=request_json, headers=request_headers)
    site = pd.DataFrame(response.json()["data"]["getSites"]["sites"])
    
    return site


def get_sites_in_bbox(bbox, source="api", stash_path = None, access_token = None):
    """
    Retrieve sites within a bounding box.

    Parameters
    ----------
    bbox : list[float]
        Bounding box in the form [min_lat, min_lon, max_lat, max_lon].
    source : str
        Data source to use. Must be one of:
        - "api": Query data from the FEMS API.
        - "stash": Query data from a local stash/cache.
    stash_path : str, optional
        Path to an Excel file used as a local stash of site data. If provided
        during an API call, results will also be written to this path. If using
        "stash" as the source, this path will be read from instead. Defaults to
        "data/fems_sts.xlsx" if not provided when source="stash".
    access_token : str, optional
        Access token for the FEMS API. Required if the API endpoint enforces
        authentication. Ignored if source="stash".

    Returns
    -------
    list
        List of sites within the bounding box.

    Raises
    ------
    ValueError
        If 'source' is not one of the accepted values.
    """

    if source == "stash":
        if stash_path is None: stash_path = "data/fems_sts.xlsx"; print(f"Searching for stash path at: {stash_path}")
        if not osp.exists(stash_path):
            print(f"Getting all FEMS stations and building stash at {stash_path}")
            get_all_sites(stash_path = stash_path)
        print(f"Reading Local station data stash: {stash_path}")
        df = pd.read_excel(stash_path)
    elif source == "api":
        sts = get_all_sites(access_token = access_token, verbose=True, stash_path = stash_path)
        df = pd.DataFrame(sts)
    else:
        raise ValueError(f"Invalid source '{source}'. Must be 'api' or 'stash'.")

    # Filter by bbox
    print(f"Filtering Stations to Bounding Box: {bbox}")
    df = df[(df["latitude"] >= bbox[0]) & (df["latitude"] <= bbox[2]) & (df["longitude"] >= bbox[1]) & (df["longitude"] <= bbox[3])]
    print(f"Stations found within bbox: {df.shape[0]}")
    
    return df


# Fuel Sample Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_fuel_types(stash_path=None):
    """
    Retrieve all fuel types from the FEMS API and optionally write to a YAML file.

    Parameters
    ----------
    stash_path : str, optional
        If provided, the resulting dictionary will be written to this YAML file.

    Returns
    -------
    dict
        Dictionary of fuels keyed by fuel_type.
    """
    query = """
    query {
      getFuels(returnAll: true) {
        fuel_id
        fuel_type
        category
        scientific_name
        modified_time
        modified_by
        created_time
        created_by
      }
    }
    """

    resp = requests.post(API_URL, json={"query": query})
    resp.raise_for_status()
    fuels = resp.json()["data"]["getFuels"]

    fuels_by_type = {f["fuel_type"]: f for f in fuels}

    if stash_path is not None:
        with open(stash_path, "w") as f:
            yaml.safe_dump(fuels_by_type, f, sort_keys=False)

    return fuels_by_type

def get_fuel_request_page_count(fuel_params, access_token=None):
    """
    """
    request_headers = api_request_headers(access_token)
    request_json = {
        "query": FUELS_PAGES_QUERY,
        "variables": fuelQueryVars(0, fuel_params),
    }
    fuel_pages_response = requests.post(
        url=API_URL,
        json=request_json,
        headers=request_headers,
    )
    pages = fuel_pages_response.json()["data"]["getFuelSamples"]["pageInfo"][
        "page_count"
    ]
    return pages

def get_fuel_data(fuel_params, access_token=None, verbose=True,
            base_delay=0.2, max_delay=30, max_retries=5, save_path=None):
    """
    Request fuel sample data from the FEMS API.

    Parameters
    ----------
    fuel_params : dict
        {
            "startDate": datetime str of format "%Y-%m-%dT%H%M%S+00",
            "endDate": datetime str of format "%Y-%m-%dT%H%M%S+00",
            "fuelTypes": str or list[str],
            "siteIds": int or list[int]
        }
    access_token : str, optional
        Access token for the FEMS API.
    base_delay : float
        seconds to sleep between queries before any issues encountered
    max_delay : float
        max seconds to wait between blocked or failed queries
    max_retries : int
        max number of times to fail query before exiting
    save_path : 
        file path to location to save data. If provided, the code will incrementally save data as it goes. Assumes csv file format with a dataframe 

    Returns
    -------
    list
        Aggregated results for the given site(s) and fuel type(s).
    """
    request_headers = api_request_headers(access_token)

    # Normalize to lists
    site_ids = fuel_params["siteIds"]
    if not isinstance(site_ids, (list, tuple)):
        site_ids = [site_ids]

    fuel_types = fuel_params["fuelTypes"]
    if not isinstance(fuel_types, (list, tuple)):
        fuel_types = [fuel_types]

    results = []
    delay = base_delay
    if verbose:
        print(f"Querying FEMS from {fuel_params['startDate']}")
        print(f"Number of stations to query: {len(site_ids)}")
    for siteId in site_ids:
        print("~"*50)
        print(f"Collecting Data for {siteId}=")
        for fuelType in fuel_types:
            print(f"{fuelType=}")
            single_params = {
                "siteId": siteId,
                "fuelType": fuelType,
                "startDate": fuel_params["startDate"],
                "endDate": fuel_params["endDate"],
            }
            pages = get_fuel_request_page_count(single_params, access_token)
            for page in range(pages + 1):
                request_json = {
                    "query": FUELS_QUERY,
                    "variables": fuelQueryVars(page, single_params),
                }

                for attempt in range(1, max_retries + 1):
                    t0 = time.time()
                    response = requests.post(API_URL, json=request_json, headers=request_headers)
                    elapsed = time.time() - t0
                    if response.status_code == 200:
                        data = response.json()
                        new_rows = data["data"]["getFuelSamples"]["fuelSamples"]
                        results += new_rows
                        # Write output to target file, append rows and turn off header if file already exists
                        if save_path and len(new_rows) > 0:
                            exists = osp.exists(save_path)
                            pd.DataFrame(new_rows).to_csv(save_path, mode="a", header=not exists, index=False)
                        
                        # Reduce sleep time back if working
                        delay = max(base_delay, delay * 0.8)
                        if verbose:
                            print(f"  page {page}: success ({elapsed:.2f}s) sleeping {delay:.2f}")
                        time.sleep(delay)
                        
                        break                        
                        
                    elif response.status_code in (429, 503):
                        # check for Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            delay = float(retry_after)
                        else:
                            delay = min(max_delay, delay * 2 * random.uniform(0.5, 1.5))
        
                        print(f"  page {page}: throttled ({response.status_code}), retry in {delay:.2f}s")
                        time.sleep(delay)
        
                    else:
                        print(f"  page {page}: error {response.status_code}, attempt {attempt}/{max_retries}")
                        delay = min(max_delay, delay * 2)
                        time.sleep(delay)
                else:
                    print(f"  page {page}: failed after {max_retries} retries")

    

    return pd.DataFrame(results)


