# Imports
import concurrent.futures
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
import time

# Setup
retrievalData = "./retrievals.csv"  # set accordingly to relevant retrievals csv

promptErrorData = "./promptErrors.csv"

load_dotenv()
openai_api_key = os.environ.get("openai.api_key")
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
        output to extractions.csv.
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
        If attempt is set to "first", a new CSV called reruns.csv is created
        from scratch. If attempt is set to "later", the new results are appended
        to an already existing reruns.csv. This allows the function to be called
        multiple times without erasing the progress from previous reruns. attempt
        is set to "first" by default.
    Return
        - A dataframe containing each rerun candidate’s name, state, min year,
        candid, college major, undergraduate institution, highest degree and
        institution, work history, sources, and ChatGPT confidence. This dataframe
        is also output to reruns.csv.
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
        futures = {executor.submit(chatFeed, prompt): prompt for prompt in prompts}
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
        model="gpt-3.5-turbo-0613",
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


def getAge():
    """
    Description
        - Uses the data gathered during the data retrieval stage to determine
        the age of each candidate.
    Parameters
        - No input parameters.
    Return
        - An array containing the age or year of birth, source URLs, full name,
        min year, state, and candid of a candidate. The element containing the
        source URLs is a string array.
    """

    # converts the existing scraped data into appropriate prompts to feed into ChatGPT
    df = pd.read_csv(
        "../Samples/order1000retrievals.csv", index_col=None, encoding="latin-1"
    )
    scraped_text = df["ChatGPT Prompt"].apply(
        lambda row: row.split("text: ")[-1].split("If any desired")[0]
    )
    prompts = []
    for i in range(len(df)):
        scraped_text[i] = (
            f"Print a value indicating year of birth either the age of {df.iloc[i]['Full Name']}, a state representative candidate from {df.iloc[i]['State']}. If the birth year or the date of birth is present, print only the birth year as a number. If the year of undergraduate graduation is present, subtract 22 from that year and print that. No full sentences. If the information is not present, print N/A: {scraped_text[i]}"
        )
        prompts.append(
            [
                scraped_text[i],
                df.iloc[i]["Sources"],
                df.iloc[i]["Full Name"],
                df.iloc[i]["Min Year"],
                df.iloc[i]["State"],
                df.iloc[i]["Candid"],
            ]
        )

    # gets candidate ages using ChatGPT API and multithreading
    outputs = []
    promptErrors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(chatFeed, prompt): prompt for prompt in prompts[:10]}
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
    return [output[0] for output in outputs]


def extractCSV(outputs, promptErrors, variant="normal", attempt="first"):
    """
    Description
        - Processes the data gathered in the data extraction stage and converts
        it into the corresponding pandas dataframes and CSVs. Handles normal
        responses, prompt errors, and parse errors, which are stored in
        extractions.csv, promptErrors.csv, and parseErrors.csv, respectively.
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
        dataframe containing the final results will be output to extractions.csv.
        If variant is set to "rerun", the dataframe containing the final results
        will be output to reruns.csv. variant is set to "normal" by default.
        - attempt: a string that indicates which type of rerun is being processed.
        If attempt is set to "first", a new CSV called reruns.csv is created from
        scratch. If attempt is set to "later", the new results are appended to an
        already existing reruns.csv. This allows the function to be called multiple
        times without erasing the progress from previous reruns. attempt is set
        to "first" by default.
    Return
        - A dataframe containing each candidate’s name, state, min year, candid,
        college major, undergraduate institution, highest degree and institution,
        work history, sources, and ChatGPT confidence. If variant is set to
        "normal", this dataframe is also output to extractions.csv. If variant
        is set to "rerun", this dataframe is instead output to reruns.csv.
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
        promptErrorFrame.to_csv("promptErrors.csv")
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
            df.to_csv("extractions.csv")  # stores results to extractions.csv
        elif variant == "rerun":
            if attempt == "first":
                df.to_csv("reruns.csv")  # stores new results in reruns.csv
            else:
                df.to_csv("reruns.csv", mode="a")  # appends new results to reruns.csv
        else:
            print("invalid extractCSV variant")
    except:
        df = -1

    # creates or appends to CSV containing parse errors
    rawParseErrors = {"Parse Error": parseErrors}
    try:
        parseErrorFrame = pd.DataFrame(rawParseErrors, columns=["Parse Error"])
        if variant == "normal":
            parseErrorFrame.to_csv("parseErrors.csv")
        elif variant == "rerun":
            parseErrorFrame.to_csv("parseErrors.csv", mode="a")
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
    # output = {"Prompt": "p", "Sources": ["source"], "Full Name": "name", "Min Year": 1, "State": "MA", "Candid": 1, "Response": '{\n  "College Major": "Biological Engineering",\n  "Undergraduate Institution": "University of Missouri-Columbia",\n  "Highest Degree and Institution": "MD, University of Missouri-Columbia",\n  "Work History": "2013-2010: Resident Physician, Internal Medicine, Barnes-Jewish Hospital//Washington University School of Medicine, St. Louis, MO\\n2017-2014: Clinical Fellow, Medical Oncology//Hematology, Barnes-Jewish Hospital//Washington University School of Medicine, St. Louis, MO\\nPresent-2021: Assistant Professor in Medicine, Department of Medicine (VA Division), Washington University School of Medicine, St. Louis, MO\\n2020-2017: Instructor in Medicine, Department of Medicine (VA Division), Washington University School of Medicine, St. Louis, MO",\n  "Confidence Level": 90\n}'}
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
