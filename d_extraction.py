"""
CandidateBios Data Extraction

Created by Victor Verma
Last edited May 7, 2024

This file was used to feed the scraped information that was temporarily stored 
in c1_retrievals.csv, which was produced in c_retrieval.py, into the OpenAI API 
for summarization. The OpenAI responses that were correctly parsed were stored 
in d1_extractions.csv. The responses in this file can be interpreted as the final 
biodata output. The ChatGPT prompts that ran into errors with the API were stored 
in d2_promptErrors.csv. The OpenAI responses that were incorrectly parsed were 
stored in d3_parseErrors.csv. The prompt errors in d2_promptErrors.csv could also 
be retried directly from the file, and the results were stored in d4_reruns.csv.
"""

# Imports
import concurrent.futures
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time

# Setup
retrievalData = "./c1_retrievals.csv"  # set accordingly to relevant retrievals csv

promptErrorData = "./d2_promptErrors.csv"

# promptErrorData = "../Results/Missing/Summary/promptErrors-missing-summary.csv"

load_dotenv()
openai_api_key = os.environ.get("openai_api_key")
assert openai_api_key


# Data Extraction
def extract(csvColumns="regular"):
    """
    Description
        - Wrapper function used to run the data extraction phase.
    Parameters
        - csvColumns: a string that describes the names of the columns of the
        source CSV file. If csvColumns is set to regular, then the program parses
        the column names "ChatGPT Prompt", "Sources", "Full Name", "Min Year",
        "State", and "Candid". If csvColumns is set to "condensed", then the
        program parses the column names "chatgptprompt", "sources", "fullname",
        "minyear", "state", and "candid".
    Return
        - A dataframe containing each candidate’s name, state, min year, candid,
        college major, undergraduate institution, highest degree and institution,
        work history, sources, and ChatGPT confidence. This dataframe is also
        output to d1_extractions.csv.
    """

    # verifies parameters
    assert csvColumns in ["regular", "condensed"]

    startExtract = time.perf_counter()

    # processes retrieval data
    if csvColumns == "regular":
        try:
            df = pd.read_csv(retrievalData, index_col=None, encoding="latin-1")
            prompts = [
                {
                    "Prompt": prompt,
                    "Sources": sources,
                    "Full Name": full_name,
                    "Min Year": min_year,
                    "State": state,
                    "Candid": candid,
                }
                for prompt, sources, full_name, min_year, state, candid in zip(
                    df["ChatGPT Prompt"],
                    df["Sources"],
                    df["Full Name"],
                    df["Min Year"],
                    df["State"],
                    df["Candid"],
                )
            ]
        except:
            print("extract - retrievalData processing error")
    elif csvColumns == "condensed":
        try:
            df = pd.read_csv(retrievalData, index_col=None, encoding="latin-1")
            prompts = [
                {
                    "Prompt": prompt,
                    "Sources": sources,
                    "Full Name": full_name,
                    "Min Year": min_year,
                    "State": state,
                    "Candid": candid,
                }
                for prompt, sources, full_name, min_year, state, candid in zip(
                    df["chatgptprompt"],
                    df["sources"],
                    df["fullname"],
                    df["minyear"],
                    df["state"],
                    df["candid"],
                )
            ]
        except:
            print("extract - retrievalData processing error")

    # summarizes prompts using ChatGPT API and multithreading
    outputs = []
    promptErrors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(chatFeed, prompt): prompt for prompt in prompts}
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                print(f"{output} extract - chatFeed generated an exception: {exc}")
                promptErrors += [output]
                with open("errors.txt", "a") as f:
                    f.write(
                        f"\n\n{output} extract - chatFeed generated an exception: {exc}"
                    )
    doneFeed = time.perf_counter()
    print(f"chatFeed: {doneFeed - startExtract} seconds")

    # creates CSVs containing the final results and errors
    extractions = extractCSV(outputs, promptErrors, variant="normal")
    doneExtractCSV = time.perf_counter()
    print(f"extractCSV: {doneExtractCSV - doneFeed} seconds")

    doneExtract = time.perf_counter()
    print(f"data extraction: {doneExtract - startExtract} seconds")
    return extractions


def extractAgain(attempt="first"):
    """
    Description
        - Wrapper function used to rerun the data extraction phase for candidates
        who encountered prompt errors.
    Parameters
        - attempt: a string that indicates which type of rerun is being processed.
        If attempt is set to "first", a new CSV called d4_reruns.csv is created
        from scratch. If attempt is set to "later", the new results are appended
        to an already existing d4_reruns.csv. This allows the function to be called
        multiple times without erasing the progress from previous reruns. attempt
        is set to "first" by default.
    Return
        - A dataframe containing each rerun candidate’s name, state, min year,
        candid, college major, undergraduate institution, highest degree and
        institution, work history, sources, and ChatGPT confidence. This dataframe
        is also output to d4_reruns.csv.
    """

    # verifies parameters
    assert attempt in ["first", "later"]

    startRerun = time.perf_counter()

    # processes prompt error data
    df = pd.read_csv(promptErrorData, index_col=None, encoding="latin-1")
    prompts = [
        {
            "Prompt": prompt,
            "Sources": sources,
            "Full Name": full_name,
            "Min Year": min_year,
            "State": state,
            "Candid": candid,
        }
        for prompt, sources, full_name, min_year, state, candid in zip(
            df["ChatGPT Prompt"],
            df["Sources"],
            df["Full Name"],
            df["Min Year"],
            df["State"],
            df["Candid"],
        )
    ]

    # summarizes prompts using ChatGPT API and multithreading
    outputs = []
    promptErrors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(chatFeed, prompt): prompt for prompt in prompts[2000:]
        }
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                print(f"{output} rerun - chatFeed generated an exception: {exc}")
                promptErrors += [output]
                with open("errors.txt", "a") as f:
                    f.write(
                        f"\n\n{output} rerun - chatFeed generated an exception: {exc}"
                    )
    doneFeed = time.perf_counter()
    print(f"chatFeed: {doneFeed - startRerun} seconds")

    # creates CSVs containing the final results and errors
    reruns = extractCSV(outputs, promptErrors, variant="rerun", attempt=attempt)
    doneRerunCSV = time.perf_counter()
    print(f"rerunCSV: {doneRerunCSV - doneFeed} seconds")

    doneRerun = time.perf_counter()
    print(f"prompt error rerun: {doneRerun - startRerun} seconds")
    return reruns


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda _: print("retrying chatFeed"),
)
def chatFeed(p):
    """
    Description
        - Uses the ChatGPT API to summarize the biodata from the scraped text
        and provide a JSON response.
    Parameters
        - p: A dictionary containing the ChatGPT prompt, source URLs, full name,
        min year, state, and candid of a candidate as keys. The value containing
        the source URLs is a string array.
    Return
        - A dictionary containing the ChatGPT response, source URLs, full name,
        min year, state, and candid of a candidate as keys. The value containing
        the source URLs is a string array.
    """

    # gets ChatGPT response
    client = OpenAI(api_key=openai_api_key, max_retries=10)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0,
        max_tokens=200,
        messages=[
            {"role": "system", "content": "Act as a summarizer"},
            {"role": "system", "content": p["Prompt"]},
        ],
    )

    output = p
    output["Response"] = response.choices[0].message.content

    return output


def getBirthYear():
    """
    Description
        - Processes the retrieval data, extracts the birth year of the candidate, and converts it into a pandas dataframe and CSV.
    Parameters
        - No input parameters.
    Return
        - A dataframe containing each candidate’s candid and year of birth.
    """

    retrievalData = "../DataTests/Samples/order1000retrievals.csv"
    # converts the existing scraped data into appropriate prompts to feed into ChatGPT
    df = pd.read_csv(retrievalData, index_col=None, encoding="latin-1")
    scraped_text = df["ChatGPT Prompt"].apply(
        lambda row: row.split("text: ")[-1].split("If any desired")[0]
    )
    prompts = []
    for i in range(len(df)):
        scraped_text[i] = (
            f"Print a value indicating the year of birth of {df.iloc[i]['Full Name']}, a state representative candidate from {df.iloc[i]['State']}. If the year of birth is present, print only the year as a number. If the year of undergraduate graduation is present, subtract 22 from that year and print that. No full sentences. If the information is not present, print N/A, and nothing else: {scraped_text[i]}"
        )
        prompts.append(
            {
                "Prompt": scraped_text[i],
                "Sources": df.iloc[i]["Sources"],
                "Full Name": df.iloc[i]["Full Name"],
                "Min Year": df.iloc[i]["Min Year"],
                "State": df.iloc[i]["State"],
                "Candid": df.iloc[i]["Candid"],
            }
        )

    # gets candidate years of birth using ChatGPT API and multithreading
    outputs = []
    promptErrors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(chatFeed, prompt): prompt for prompt in prompts}
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                print(f"{output} extract - chatFeed generated an exception: {exc}")
                promptErrors += [output]
                with open("errors.txt", "a") as f:
                    f.write(
                        f"\n\n{output} extract - chatFeed generated an exception: {exc}"
                    )

    yearResults = {"Candid": [], "Birth Year": []}
    for output in outputs:
        yearResults["Candid"].append(output["Candid"])
        yearResults["Birth Year"].append(output["Response"])
    df = pd.DataFrame(
        yearResults,
        columns=[
            "Candid",
            "Birth Year",
        ],
    )
    df.to_csv("birthYears.csv", index=False)
    return df


def extractCSV(outputs, promptErrors, variant="normal", attempt="first"):
    """
    Description
        - Processes the data gathered in the data extraction stage and converts
        it into the corresponding pandas dataframes and CSVs. Handles normal
        responses, prompt errors, and parse errors, which are stored in
        d1_extractions.csv, d2_promptErrors.csv, and d3_parseErrors.csv, respectively.
    Parameters
        - outputs: an array containing the relevant candidate information for
        each candidate as the elements in the array. Each element is itself a
        dictionary containing the ChatGPT response, source URLs, full name, min year,
        state, and candid of a candidate as keys. The value containing the source
        URLs is a string array.
        - promptErrors: an array containing all candidates that encountered
        prompt errors during the chatFeed function. Each element is itself a
        dictionary containing the ChatGPT prompt, source URLs, full name, min year,
        state, and candid of a candidate. The value containing the source URLs
        is a string array.
        - variant: a string that specifies if the outputs are being processed
        normally or as part of a rerun. If variant is set to "normal", the
        dataframe containing the final results will be output to d1_extractions.csv.
        If variant is set to "rerun", the dataframe containing the final results
        will be output to d4_reruns.csv. variant is set to "normal" by default.
        - attempt: a string that indicates which type of rerun is being processed.
        If attempt is set to "first", a new CSV called d4_reruns.csv is created from
        scratch. If attempt is set to "later", the new results are appended to an
        already existing d4_reruns.csv. This allows the function to be called multiple
        times without erasing the progress from previous reruns. attempt is set
        to "first" by default.
    Return
        - A dataframe containing each candidate’s name, state, min year, candid,
        college major, undergraduate institution, highest degree and institution,
        work history, sources, and ChatGPT confidence. If variant is set to
        "normal", this dataframe is also output to d1_extractions.csv. If variant
        is set to "rerun", this dataframe is instead output to d4_reruns.csv.
    """

    # verifies parameters
    assert variant in ["normal", "rerun"]
    assert attempt in ["first", "later"]

    # creates CSV containing prompt errors
    rawPromptErrors = {
        "ChatGPT Prompt": [],
        "Sources": [],
        "Full Name": [],
        "Min Year": [],
        "State": [],
        "Candid": [],
    }
    for error in promptErrors:
        try:
            rawPromptErrors["ChatGPT Prompt"].append(error["Prompt"])
            rawPromptErrors["Sources"].append(error["Sources"])
            rawPromptErrors["Full Name"].append(error["Full Name"])
            rawPromptErrors["Min Year"].append(error["Min Year"])
            rawPromptErrors["State"].append(error["State"])
            rawPromptErrors["Candid"].append(error["Candid"])
        except:
            continue
    try:
        promptErrorFrame = pd.DataFrame(
            rawPromptErrors,
            columns=[
                "ChatGPT Prompt",
                "Sources",
                "Full Name",
                "Min Year",
                "State",
                "Candid",
            ],
        )
        promptErrorFrame.to_csv("d2_promptErrors.csv")
        print(f"\n{promptErrorFrame.head()}\n{len(promptErrorFrame)} rows\n")
    except:
        print("promptErrorFrame not constructed")

    # creates or appends to CSV containing final results
    parseErrors = []
    rawResults = {
        "Name": [],
        "State": [],
        "Min Year": [],
        "Candid": [],
        "College Major": [],
        "Undergraduate Institution": [],
        "Highest Degree and Institution": [],
        "Work History": [],
        "Sources": [],
        "ChatGPT Confidence": [],
    }

    # parses ChatGPT responses using multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(parse, output): output for output in outputs}
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                data = future.result()
                if data != None:  # verifies data exists
                    rawResults["Name"].append(data["Full Name"])
                    rawResults["College Major"].append(data["College Major"])
                    rawResults["Undergraduate Institution"].append(
                        data["Undergraduate Institution"]
                    )
                    rawResults["Highest Degree and Institution"].append(
                        data["Highest Degree and Institution"]
                    )
                    rawResults["Work History"].append(data["Work History"])
                    rawResults["ChatGPT Confidence"].append(data["Confidence Level"])
                    rawResults["Sources"].append(data["Sources"])
                    rawResults["Min Year"].append(data["Min Year"])
                    rawResults["State"].append(data["State"])
                    rawResults["Candid"].append(data["Candid"])
                elif data[0] == -1:
                    parseErrors.append(data[1])
            except Exception as exc:
                print(f"{output} extract - parse generated an exception: {exc}")
                with open("errors.txt", "a") as f:
                    f.write(
                        f"\n\n{output} extract - parse generated an exception: {exc}"
                    )
    try:
        df = pd.DataFrame(
            rawResults,
            columns=[
                "Name",
                "State",
                "Min Year",
                "Candid",
                "College Major",
                "Undergraduate Institution",
                "Highest Degree and Institution",
                "Work History",
                "Sources",
                "ChatGPT Confidence",
            ],
        )
        if variant == "normal":
            df.to_csv("d1_extractions.csv")  # stores results to d1_extractions.csv
        elif variant == "rerun":
            if attempt == "first":
                df.to_csv("d4_reruns.csv")  # stores new results in d4_reruns.csv
            else:
                df.to_csv(
                    "d4_reruns.csv", mode="a"
                )  # appends new results to d4_reruns.csv
        else:
            print("invalid extractCSV variant")
    except:
        df = -1

    # creates or appends to CSV containing parse errors
    rawParseErrors = {"Parse Error": parseErrors}
    try:
        parseErrorFrame = pd.DataFrame(rawParseErrors, columns=["Parse Error"])
        if variant == "normal":
            parseErrorFrame.to_csv("d3_parseErrors.csv")
        elif variant == "rerun":
            parseErrorFrame.to_csv("d3_parseErrors.csv", mode="a")
        else:
            print("invalid extractCSV variant")
        print(f"{parseErrorFrame.head()}\n{len(parseErrorFrame)} rows\n")
    except:
        print("parseErrorFrame not constructed")

    if df.empty:
        return rawResults
    else:
        return df


def parse(output):
    """
    Description
        - Reads the JSON formatted ChatGPT response of a candidate and extracts
        the full name, college major, undergraduate institution, highest degree
        and institution, and work history. Candidates whose responses get parsed
        incorrectly are appended to parseErrors.
    Parameters
        - output: a dictionary containing the ChatGPT response, source URLs, full
        name, min year, state, and candid of a candidate as keys. The value containing
        the source URLs is a string array.
    Return
        - If successful, a dictionary containing the full name, college major,
        undergraduate institution, highest degree and institution, work history,
        ChatGPT confidence, sources, min year, state, and candid of a candidate
        as keys. The value containing the source URLs is a string array.
        If unsuccessful, the return value is an array whose first element is -1.
    """

    data = {
        "College Major": "",
        "Undergraduate Institution": "",
        "Highest Degree and Institution": "",
        "Work History": "",
        "Confidence Level": "",
    }
    try:
        d = json.loads(output["Response"].replace("\n", ""))  # splits JSON data
        data["Sources"] = output["Sources"]
        data["Full Name"] = output["Full Name"]
        data["Min Year"] = output["Min Year"]
        data["State"] = output["State"]
        data["Candid"] = output["Candid"]
        data.update(d)  # updates ChatGPT response data
    except:
        return [-1, output]
    return data
