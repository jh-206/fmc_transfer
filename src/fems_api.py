import requests as requests

API_URL = "https://fems.fs2c.usda.gov/fuelmodel/apis/graphql"

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


def get_fuel_data_of_sites(fuel_params, access_token=None):
    """
    Parameters
    ----------
    fuel_params: dict of form
        {
                "startDate": datetime str of format "%Y-%m-%dT%H%M%S+00",
                "endDate": datetime str of format "%Y-%m-%dT%H%M%S+00",
                "fuelTypes": Array[str] representing fuel from FEMS,
                "siteIds": Array[int] for the id of the site
        }
    access_token: str
        Access token for the FEMS API (optional)

    Returns
    ----------
        Array of results for provided fuel_params from FEMS API


    """
    results = []
    for siteId in fuel_params["siteIds"]:
        for fuelType in fuel_params["fuelTypes"]:
            fuel_param = {
                "fuelType": fuelType,
                "siteId": siteId,
                "startDate": fuel_params["startDate"],
                "endDate": fuel_params["endDate"],
            }
            results += get_fuel_data(fuel_param, access_token)

    return results


def get_fuel_data(fuel_params, access_token=None):
    """
    Parameters
    ----------
    fuel_params: dict of form
        {
                "startDate": datetime str of format "%Y-%m-%dT%H%M%S+00",
                "endDate": datetime str of format "%Y-%m-%dT%H%M%S+00",
                "fuelType": str representing fuel from FEMS,
                "siteId": int for the id of the site
        }
    access_token: str
        Access token for the FEMS API (optional)

    Returns
    ----------
        Array of results for provided fuel_params from FEMS API


    """
    breakpoint()
    request_headers = api_request_headers(access_token)
    pages = get_fuel_request_page_count(fuel_params, access_token)

    results = []
    for page in range(pages + 1):
        request_json = {
            "query": FUELS_QUERY,
            "variables": fuelQueryVars(page, fuel_params),
        }
        response = requests.post(
            url=API_URL,
            json=request_json,
            headers=request_headers,
        )
        results = results + response.json()["data"]["getFuelSamples"]["fuelSamples"]

    return results


def get_fuel_request_page_count(fuel_params, access_token=None):
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
