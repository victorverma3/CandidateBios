# Imports
from apiclient.discovery import build
from dotenv import load_dotenv
import os
import pandas as pd
import random
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time

# Setup
sourceData = "../../../../Data/HarvardData/ldata_R_unique.csv"  # set to './ldata_R_unique' if on SCC

load_dotenv()
google_api_key = os.environ.get("google_api_key")
assert google_api_key
engine = os.environ.get("engine")
assert engine
resource = build("customsearch", "v1", developerKey=google_api_key).cse()

states = {
    "al": "Alabama",
    "ak": "Alaska",
    "az": "Arizona",
    "ar": "Arkansas",
    "ca": "California",
    "co": "Colorado",
    "ct": "Connecticut",
    "de": "Delaware",
    "fl": "Florida",
    "ga": "Georgia",
    "hi": "Hawaii",
    "id": "Idaho",
    "il": "Illinois",
    "in": "Indiana",
    "ia": "Iowa",
    "ks": "Kansas",
    "ky": "Kentucky",
    "la": "Louisiana",
    "me": "Maine",
    "md": "Maryland",
    "ma": "Massachusetts",
    "mi": "Michigan",
    "mn": "Minnesota",
    "ms": "Mississippi",
    "mo": "Missouri",
    "mt": "Montana",
    "ne": "Nebraska",
    "nv": "Nevada",
    "nh": "New Hampshire",
    "nj": "New Jersey",
    "nm": "New Mexico",
    "ny": "New York",
    "nc": "North Carolina",
    "nd": "North Dakota",
    "oh": "Ohio",
    "ok": "Oklahoma",
    "or": "Oregon",
    "pa": "Pennsylvania",
    "ri": "Rhode Island",
    "sc": "South Carolina",
    "sd": "South Dakota",
    "tn": "Tennessee",
    "tx": "Texas",
    "ut": "Utah",
    "vt": "Vermont",
    "va": "Virginia",
    "wa": "Washington",
    "wv": "West Virginia",
    "wi": "Wisconsin",
    "wy": "Wyoming",
}

testRows = {"link test": 126073, "content test": 24526, "length test": 142634}


# Data Search
def search(n=1, r=4, read="random"):
    """
    Description
        - Wrapper function used to run the data search phase.
    Parameters
        - n: an integer that indicates the number of unique candidates for whom
        to gather biodata.
        - r: an integer that specifies the number of Google API search results
        to use during the gathering process. 1 <= r <= 4, and r is set to 4 by
        default.
        - read: a string that defines how the n unique candidates should be chosen.
        If read is set to "random", then n unique random candidates are used,
        and if read = ‘order’, then the first n unique candidates in order are
        used. read is set to random by default.
    Return
        - A dataframe containing each candidate’s Google Search results, first
        name, middle name, last name, full name, min year, state, and candid.
        This dataframe is also output to searches.csv.
    """

    startSearch = time.perf_counter()

    # verifies parameters
    assert n >= 0
    assert 1 <= r <= 4
    assert read in ["random", "order"]

    # reads candidate source data
    if read == "random":
        reps = randomRead(sourceData, n)  # chooses candidates randomly
    elif read == "order":
        reps = orderRead(sourceData, n)  # chooses candidates in order
    else:
        print('retrieval error - set "read" parameter to "random" or "order"')
        return -1
    doneRead = time.perf_counter()
    print(f"{read}Read: {doneRead - startSearch} seconds")

    # gets Google Search API URL results
    urls = []
    for index, rep in enumerate(reps):
        urls += [googleSearch(rep, r)]
        print(index, rep)
        time.sleep(0.6)
    doneGoogle = time.perf_counter()
    print(f"googleSearch: {doneGoogle - doneRead} seconds")

    # creates CSV containing Google URLs and other relevant candidate info
    searches = searchCSV(urls)
    doneSearchCSV = time.perf_counter()
    print(f"searchCSV: {doneSearchCSV - doneGoogle} seconds")

    doneSearch = time.perf_counter()
    print(f"data search: {doneSearch - startSearch} seconds")
    return searches


def searchRow(r=4, rows=[126073]):
    """
    Description
        - Wrapper function used to run the data search phase on specified candidates.
    Parameters
        - r: an integer that specifies the number of Google API search results
        to use during the gathering process. 1 <= r <= 4, and r is set to 4 by
        default.
        - rows: an integer array containing the candidates' row numbers for whom
        to gather biodata. The row number passed into the array for a candidate
        should be equal to the row number for that candidate in
        ldata_R_unique.csv - 2 in order to account for the indexing in pandas
        dataframes.
    Return
        - A dataframe containing each candidate's Google Search results, first
        name, middle name, last name, full name, min year, state, and candid.
        This dataframe is also output to searches.csv. This dataframe is also
        output to searches.csv.
    """

    startSearch = time.perf_counter()

    # verifies parameters
    assert 1 <= r <= 4

    # reads candidate source data for specified rows
    reps = rowRead(sourceData, rows)
    doneRead = time.perf_counter()
    print(f"rowRead: {doneRead - startSearch} seconds")

    # gets Google Search API URL results
    urls = []
    for rep in reps:
        urls.append(googleSearch(rep, r))
        time.sleep(0.25)
    doneGoogle = time.perf_counter()
    print(f"googleSearch: {doneGoogle - doneRead} seconds")

    # creates CSV containing ChatGPT prompts and other relevant candidate info
    searches = searchCSV(urls)
    doneSearchCSV = time.perf_counter()
    print(f"searchCSV: {doneSearchCSV - doneGoogle} seconds")

    doneSearch = time.perf_counter()
    print(f"data search: {doneSearch - startSearch} seconds")
    return searches


def randomRead(file, n):
    """
    Description
        - Reads the relevant candidate information from ldata_R_unique.csv for
        n randomly chosen candidates.
    Parameters
        - file: a string representing the file path to ldata_R_unique. The global
        variable sourceData is always passed in as file.
        - n: an integer that indicates the number of unique candidates for whom
        to read relevant information from ldata_R_unique.csv. The candidates are
        chosen randomly.
    Return
        - An array containing the relevant candidate information for each
        candidate as the elements in the array. Each element in the array is a
        dictionary containing the full name delimited by quotes, state, first name,
        last name, full name, and candid for the chosen candidate.
    """

    # verifies parameters
    assert n > 0

    df = pd.read_csv(file, index_col=None, encoding="latin-1")
    df["sab"] = df["sab"].str.strip().replace(states)
    alreadyRead = {}  # stores processed candidates
    reps = []  # stores candidate info

    i = 0
    while i < n:
        row = random.randint(0, len(df) - 1)  # randomly chooses candidate
        name = [
            str(df["first"][row]),
            str(df["middle"][row]),
            str(df["last"][row]),
            str(df["suffix"][row]),
        ]
        candidate = {
            "First": name[0].lower(),
            "Middle": name[1].lower(),
            "Last": name[2].lower(),
            "Full": " ".join(part for part in name if part != "nan").lower(),
            "Min Year": str(df["min_year"][row]),
            "State": df["sab"][row],
            "Candid": str(df["candid"][row]),
            "Row": str(row),
        }
        if candidate["Full"] not in alreadyRead:  # verifies candidate is unique
            reps.append(candidate)
            alreadyRead[candidate["Full"]] = 1
            i += 1
    return reps


def orderRead(file, n):
    """
    Description
        - Reads the relevant candidate information from ldata_R_unique.csv for
        the first n candidates in order.
    Parameters
        - file: a string representing the file path to ldata_R_unique. The global
        variable sourceData is always passed in as file.
        - n: an integer that indicates the number of unique candidates for whom
        to read relevant information from ldata_R_unique.csv. The candidates are
        chosen in order.
    Return
        - An array containing the relevant candidate information for each
        candidate as the elements in the array. Each element in the array is a
        dictionary containing the full name delimited by quotes, state, first name,
        last name, full name, and candid for the chosen candidate.
    """

    # verifies parameters
    assert n > 0

    df = pd.read_csv(file, index_col=None, encoding="latin-1")
    df["sab"] = df["sab"].str.strip().replace(states)
    reps = []  # stores candidate info

    row = 0  # set to desired starting row in source data
    while row < n:  # hardcode n to len(df) to read entire CSV
        name = [
            str(df["first"][row]),
            str(df["middle"][row]),
            str(df["last"][row]),
            str(df["suffix"][row]),
        ]
        candidate = {
            "First": name[0].lower(),
            "Middle": name[1].lower(),
            "Last": name[2].lower(),
            "Full": " ".join(part for part in name if part != "nan").lower(),
            "Min Year": str(df["min_year"][row]),
            "State": df["sab"][row],
            "Candid": str(df["candid"][row]),
            "Row": str(row),
        }
        reps.append(candidate)
        row += 1
    return reps


def rowRead(file, rows):
    """
    Description
        - Reads the relevant candidate information from ldata_R_unique.csv for
        the candidates who correspond to the specified rows.
    Parameters
        - file: a string representing the file path to ldata_R_unique. The
        global variable sourceData is always passed in as file.
        - rows: an integer array containing the candidates' row numbers for whom
        to gather biodata. The row number passed into the array for a candidate
        should be equal to the row number for that candidate in
        ldata_R_unique.csv - 2 to account for the indexing in pandas dataframes.
    Return
        - An array containing the relevant candidate information for each
        candidate as the elements in the array. Each element in the array is a
        dictionary containing the full name delimited by quotes, state, first name,
        last name, full name, and candid for the chosen candidate.
    """

    df = pd.read_csv(file, index_col=None, encoding="latin-1")
    df["sab"] = df["sab"].str.strip().replace(states)
    reps = []  # stores candidate info

    for row in rows:  # iterates through all specified candidates
        name = [
            str(df["first"][row]),
            str(df["middle"][row]),
            str(df["last"][row]),
            str(df["suffix"][row]),
        ]
        candidate = {
            "First": name[0].lower(),
            "Middle": name[1].lower(),
            "Last": name[2].lower(),
            "Full": " ".join(part for part in name if part != "nan").lower(),
            "Min Year": str(df["min_year"][row]),
            "State": df["sab"][row],
            "Candid": str(df["candid"][row]),
            "Row": str(row),
        }
        reps.append(candidate)
    return reps


@retry(
    wait=wait_random_exponential(min=45, max=75),
    stop=stop_after_attempt(6),
    before_sleep=lambda _: print("retrying googleSearch"),
)
def googleSearch(rep, r=4):
    """
    Description
        - Uses the Google Custom Search JSON API and a Google Custom Search Engine
        to gather the top URLs from the Google Search of each candidate.
    Parameters
        - rep: a dictionary containing the full name delimited by quotes, state, first
        name, last name, full name, and candid of a candidate. This is exactly
        an element from the output of randomRead, orderRead, or rowRead, depending
        on how the candidates were chosen.
        - r: an integer that specifies the number of Google API search results
        to use during the gathering process. 1 <= r <= 4, and r is set to 4 by
        default.
    Return
        - A dictionary containing the top r URLs from the Google Search, first name,
        middle name, last name, full name, min year, state, and candid of a
        candidate as keys. The value containing the top r URLs is a string array.
    """

    # verifies parameters
    assert 1 <= r <= 4

    # searches Google using query of format {full name} {state}
    query = f"{rep['Full'].title()} {rep['State']}"
    print(query)
    results = resource.list(q=query, cx=engine, lr="lang_en", cr="us", num=r).execute()
    rep["Sources"] = []

    # processes Google Search API results
    try:
        for item in results["items"]:
            if "link" in item:
                rep["Sources"].append(item["link"])
    except:
        rep["Sources"] = [""]
    return rep


def searchCSV(urls):
    """
    Description
        - Processes the data gathered in the data search phrase and converts it
        into a pandas dataframe and CSV file.
    Parameters
        - urls: an array containing the relevant candidate information for
        each candidate as the elements in the array. Each element is itself a
        dictionary containing the top r URLs from the Google Search, first name,
        middle name, last name, full name, min year, state, and candid of the
        candidate as keys. The value containing the top r URLs is a string array.
    Return
        - A dataframe containing each candidate’s Google Search results, first
        name, middle name, last name, full name, min year, state, and candid.
        This dataframe is also output to searches.csv.
    """

    # creates CSV containing Google URLs and other relevant candidate info
    rawData = {
        "Sources": [],
        "First": [],
        "Middle": [],
        "Last": [],
        "Full": [],
        "Min Year": [],
        "State": [],
        "Candid": [],
    }
    for cand in urls:
        if len(cand) == 9:
            try:
                rawData["Sources"].append(cand["Sources"])
                rawData["First"].append(cand["First"])
                rawData["Middle"].append(cand["Middle"])
                rawData["Last"].append(cand["Last"])
                rawData["Full"].append(cand["Full"])
                rawData["Min Year"].append(cand["Min Year"])
                rawData["State"].append(cand["State"])
                rawData["Candid"].append(cand["Candid"])
            except:
                continue
    df = pd.DataFrame(
        rawData,
        columns=[
            "Sources",
            "First",
            "Middle",
            "Last",
            "Full",
            "Min Year",
            "State",
            "Candid",
        ],
    )
    df.to_csv("searches.csv")
    return df
