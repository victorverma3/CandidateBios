# Imports
from bs4 import BeautifulSoup
import concurrent.futures
import io
import pandas as pd
import PyPDF2
import re
from requests_html import HTMLSession
import time
import urllib.request

# Setup 
timeoutCandidates = []

searchData = './FinalResults/GoogleSearchResults/searches9700-19399.csv' # set accordingly to relevant searches

# Data Retrieval
def retrieve(searchData, timeout = 250):
    '''
    Description
        - Wrapper function used to run the data retrieval phase.
    Parameters
        - searchData: a global variable that stores the relative path to 
            searches.csv, which contains all of the information gathered in the 
            data search phase. This can optionally be configured to another CSV 
            of the proper format.
    Return
        - A dataframe containing each candidate’s ChatGPT prompt, sources, full 
        name, min year, state, and candid. This dataframe is also output to 
        retrievals.csv.
    '''
    
    # verifies parameters
    assert timeout > 0
    
    startRetrieve = time.perf_counter()
    
    # processes search data
    try: 
        df = pd.read_csv(searchData, index_col = None, encoding = 'latin-1')
        urls = [
            [sourceParser(source), first, middle, last, full, min_year, state, candid]
            for source, first, middle, last, full, min_year, state, candid in zip(df['Sources'], df['First'], df['Middle'], df['Last'], df['Full'], df['Min Year'], df['State'], df['Candid'])
        ]
    except:
        print('retrieve - searchData processing error')
        
    # splits candidate sources into batches of 100
    try: 
        cands = splitCandidates(urls, 100)
    except:
        print('retrieve - splitCandidates error')
    
    # scrapes candidate sources using batches of multithreading
    batchTimes = []
    bios = []
    
    # set firstBatch and lastBatch to desired batches to scrape
    firstBatch = 0
    lastBatch = 10
    
    # scrapes candidate source URLs multithreading
    for index, group in enumerate(cands[firstBatch: lastBatch]):
        batchStart = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(bioData, link): link for link in group}
                for future in futures:
                    info = futures[future]
                    try:
                        result = future.result(timeout)
                        bios.append(result)
                    except concurrent.futures.TimeoutError:
                        timeoutCandidates.append(info)
                        print(f'{info} retrieve - bioData timed out')
                        with open('errors.txt', 'a') as f:
                            f.write(f'\n\n{info} retrieve - bioData generated an exception: TimeoutError')
                        raise KeyboardInterrupt
                    except Exception as exc:
                        print(f'{info} retrieve - bioData generated an exception: {exc}')
                        with open('errors.txt', 'a') as f:
                            f.write(f'\n\n{info} retrieve - bioData generated an exception: {exc}')
            batchDone = time.perf_counter()
            batchTime = batchDone - batchStart
            batchTimes.append(batchTime)
            print(f'\nfinished scraping group {index + 1} / {lastBatch - firstBatch} in {batchTime} seconds\n')
        except:
            print(f'retrieve - error scraping group {index + 1}')
            batchTimes.append('error')
        
    doneBio = time.perf_counter()
    print(f'bioData: {doneBio - startRetrieve} seconds')
    print(f'batch times (seconds): {batchTimes}')
    
    # creates ChatGPT prompts for each candidate using multithreading
    prompts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(chatPrompt, info): info for info in bios}
        for future in concurrent.futures.as_completed(futures):
            prompt = futures[future]
            try:
                 prompts.append(future.result())
            except Exception as exc:
                 print(f'{prompt} retrieveRow - chatPrompt generated an exception: {exc}')
                 with open('errors.txt', 'a') as f:
                     f.write(f'\n\n{prompt} retrieveRow - chatPrompt generated an exception: {exc}')
    donePrompt = time.perf_counter()
    print(f'chatPrompt: {donePrompt - doneBio} seconds')
    
    # creates CSV containing ChatGPT prompts and other relevant candidate info
    retrievals = retrieveCSV(prompts)
    doneRetrieveCSV = time.perf_counter()
    print(f'retrieveCSV: {doneRetrieveCSV - donePrompt} seconds')
    
    doneRetrieve = time.perf_counter()
    print(f'data retrieval: {doneRetrieve - startRetrieve} seconds')
    return retrievals

def splitCandidates(urls, groupSize):
    '''
    Description
        - Splits the list of candidates to be scraped into sublists of size 
        batchSize, with the last sublist containing any leftovers.
    Parameters
        - urls: A 2-D list containing each candidate’s Google Search results, 
        first name, middle name, last name, full name, min year, state, and candid 
        as elements.
        - batchSize: an integer that specifies the size of each batch of candidates 
        to be scraped in a singular instance of the ThreadPoolExecutor().
    Return
        - A 3-D list containing the batches as elements. Each batch is itself a 2-D list of maximum size batchSize containing each candidate’s Google Search results, first name, middle name, last name, full name, min year, state, and candid as elements. 
    '''
    
    # verifies parameters
    assert groupSize > 0
    
    return [urls[i:i + groupSize] for i in range(0, len(urls), groupSize)]

def sourceParser(sources):
    '''
    Description
        - Converts a string representing an array of source URLs into a string 
        array with each source URL as an element within the array.
    Parameters
        - sources: a string representing an array of source URLs for a candidate.
    Return
        - A string array with each source URL as an element within the array.
    '''
        
    sources = sources[1:-1].split(",")
    sources = [source.strip().replace("'", "") for source in sources]
    return sources

def bioData(link):
    '''
    Description
        - Wrapper function used to scrape the source URLs for each candidate.
    Parameters
        - link: A 2D array containing the top r URLs from the Google Search, 
        first name, middle name, last name, full name, min year, state, and 
        candid of the candidate. The element containing the top r URLs is a 
        string array.
    Return
        - An array containing plain text scraped from the source URLs, the 
        source URLs, full name, year, state, and candid of a candidate. The 
        element containing the source URLs is a string array.
    '''
    
    # initializes utility variables for scraping
    info = []
    summaries = []
    
    # scrapes candidate source URLs
    for url in link[0]:
        if url != 'nan': # verifies URL exists
            try: 
                s = time.perf_counter()
                if '.pdf' in url:
                    try:
                        information = pdfReader(url).lower()  # handles PDFs
                    except:
                        print(f'pdfReader failed - {url}')
                    f = time.perf_counter()
                    print(f'pdfReader: {str(f - s)} seconds')
                else:
                    session = HTMLSession()
                    r = session.get(url)
                    soup = BeautifulSoup(r.html.raw_html, 'html.parser') # obtains html text of page
                    for tag in soup.find_all(['script', 'style']): # removes all javascript and css from page
                        tag.decompose()
                    information = ' '.join(soup.stripped_strings).strip().lower() # removes html tags, leading and trailing whitespaces, and makes text lowercase
                
                # scrapes text after last name if present
                if link[3] != 'nan':
                    summary = grabber(information, link[3])
                    
                # scrapes text after first name if present and last name not present
                if link[1] != 'nan' and not summary:
                    summary = grabber(information, link[1])
                    
                # scrapes text after middle name if present and last name and first name not present
                if link[2] != 'nan' and not summary:
                    summary = grabber(information, link[2])
                
                summaries.append(summary)
                doneScraping = time.perf_counter()
                print(f'web scraper: {str(doneScraping - s)} seconds')
            except:
                continue
    
    info += [' '.join(summaries), link[0], link[4], link[5], link[6], link[7]]
    return info

def pdfReader(url):
    '''
    Description
        - Scrapes up to the first 3 pages of a pdf.
    Parameters
        - url: a string that represents the web URL of a pdf.
    Return
        - A string representing the plain text of up to the first 3 pages of 
        the pdf.
    '''
    
    text = []

    with urllib.request.urlopen(url) as response: # opens pdf
        with io.BytesIO(response.read()) as file: # reads pdf
            reader = PyPDF2.PdfReader(file)
            for page in range(min(3, len(reader.pages))):
                text.append(reader.pages[page].extract_text()) # scrapes pdf

    return ' '.join(text)

def grabber(information, phrase):
    '''
    Description
        - Scrapes the first 400 plain text words following the occurrence of a 
        specified phrase on a webpage or pdf.
    Parameters
        - information: a string representing the plain text of a webpage or pdf.
        - phrase: a string that indicates where to start scraping the 400 words 
        within the plain text given by information. phrase can be either the last 
        name, first name, or middle name of the candidate.
    Return
        - A string representing the first 400 plain text words following the 
        occurrence of a specified phrase on a webpage or pdf.
    '''
        
    excerpt = information.partition(phrase) # splits text at the first appearance of specified phrase
    text = excerpt[1] + excerpt[2]
    text = re.sub(r'\s*\n\s*', ' ', text) # removes extra newlines and white spaces to improve word counting
    words = text.split(' ')
    if 'accessibility' in words[:400]:
        end = words.index('accessibility') # this phrase is usually followed by long text with no spaces which messes up word counting and causes prompt to go over token limit
        numWords = end + 1
    else:
        numWords = 400
    extractedWords = words[:numWords]
    summary = ' '.join(extractedWords) # creates a string of first 400 words after and including the specified phrase
    return summary

def chatPrompt(info):
    '''
    Description
        - Creates a ChatGPT prompt for the candidate using the scraped text from 
        its source URLs.
    Parameters
        - info: An array containing plain text scraped from the source URLs, the 
        source URLs, full name, year, state, and candid of a candidate. This is 
        exactly the output of the bioData() function. The element containing the 
        source URLs is a string array.
    Return
        - An array containing the ChatGPT prompt, source URLs, full name, min 
        year, state, and candid of a candidate. The element containing the source 
        URLs is a string array.
    '''
    
    p = f'Extract ONLY the College Major, Undergraduate Institution, Highest Degree \
        and Institution, and Work History of {info[2].title()}, a state representative \
        candidate from {info[4]}, from the following text: {info[0]}. If any desired \
        information is not present in the given text, write N/A instead. Determine \
        your confidence that the information you previously extracted correctly \
        describes {info[2].title()}, a {info[3]} state representative candidate from \
        {info[4]}, on a scale of 1 to 100. Display the college major, undergraduate \
        institution, highest degree and institution, work history, and your confidence \
        level as 5 elements of a JSON object.' 
    return [p, info[1], info[2], info[3], info[4], info[5]]

def retrieveCSV(prompts):
    '''
    Description
        - Processes the data gathered in the data retrieval phrase and converts 
        it into the corresponding pandas dataframes and CSVs. Handles both 
        successfully and unsuccessfully scraped candidates, storing the 
        information in retrievals.csv and scrapingTimeouts.csv, respectively.
    Parameters
        - prompt: a 2D array containing the relevant candidate information for 
        each candidate as the elements in the array. Each element is itself an 
        array containing the ChatGPT prompt, source URLs, full name, min year, 
        state, and candid of the candidate. The element containing the source 
        URLs is a string array.
    Return
        - A dataframe containing each candidate’s ChatGPT prompt, sources, full 
        name, min year, state, and candid. This dataframe is also output to 
        retrievals.csv.
    '''
    
    # creates CSV containing candidates whose web scraping timed out
    failData = {
        'Sources': [],
        'First': [],
        'Middle': [],
        'Last': [],
        'Full': [],
        'Min Year': [],
        'State': [],
        'Candid': []
    }
    for cand in timeoutCandidates:
        if len(cand) == 8: # verifies candidate info has correct format
            try:
                failData['Sources'].append(cand[0])
                failData['First'].append(cand[1])
                failData['Middle'].append(cand[2])
                failData['Last'].append(cand[3])
                failData['Full'].append(cand[4])
                failData['Min Year'].append(cand[5])
                failData['State'].append(cand[6])
                failData['Candid'].append(cand[7])
            except:
                continue
    df = pd.DataFrame(failData, columns = ['Sources', 'First', 'Middle', 'Last', 
                                          'Full', 'Min Year', 'State', 'Candid'])
    df.to_csv('scrapingTimeouts.csv')
    
    # creates CSV containing ChatGPT prompts and other relevant candidate info
    rawData = {
        'ChatGPT Prompt': [],
        'Sources': [],
        'Full Name': [],
        'Min Year': [],
        'State': [],
        'Candid': []
    }
    for cand in prompts:
        if cand != None:
            if len(cand) == 6: # verifies candidate info has correct format
                try:
                    rawData['ChatGPT Prompt'].append(cand[0])
                    try:
                        s = []
                        for i in range(4):
                            s.append(cand[1][i])
                        rawData['Sources'].append(s)
                    except:
                        rawData['Sources'].append('N/A')
                    rawData['Full Name'].append(cand[2])
                    rawData['Min Year'].append(cand[3])
                    rawData['State'].append(cand[4])
                    rawData['Candid'].append(cand[5])
                except: 
                    continue
    df = pd.DataFrame(rawData, columns = ['ChatGPT Prompt', 'Sources', 'Full Name', 
                                          'Min Year', 'State', 'Candid'])
    df.to_csv('retrievals.csv', escapechar = '/')
    return df