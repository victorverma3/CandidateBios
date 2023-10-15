# Using Large Language Models for Massive Political Science Data Scraping

# Abstract
Large Language Models like ChatGPT have recently gained popularity, but much of their potential still needs to be explored. For this project, I developed Python software utilizing ChatGPT’s ability to efficiently summarize large amounts of text in order to obtain the biographical data of nearly 150,000 U.S. state legislator candidates who ran for office between 1967 and 2017. Specifically, I gathered the college major, undergraduate institution, highest degree and institution, and work history of each candidate. Prior to this effort, there was no comprehensive database containing the desired biodata for this scale. This unique dataset offers us insights into the background experience and influences of state legislative candidates. First, I used the Google Custom Search JSON API to identify relevant websites for each candidate. Next, I scraped the chosen websites using the Beautiful Soup and PDF scraping libraries in Python. Finally, I utilized the ChatGPT API to efficiently summarize the scraped website text and extract the desired biodata. I found that approximately 40% of all U.S. state legislator candidates between 1967 and 2017 had biodata available online on Google, but this percentage generally increases as one searches for more recent candidates. The gathered biodata will be used as part of a larger project studying lobbying laws and the revolving door in politics. The methodology used in this project is extremely powerful and can serve as a general blueprint to create large-scale information databases related to any topic that can be searched on the web.

# Research Goals
- Create Python software to programmatically gather the biodata of U.S. state legislator candidates from 1967 – 2017.
- Display each candidate’s college major, undergraduate institution, highest degree and institution, work history, and source URLs in a CSV file.
- Improve my ability to write neat and efficient code.
- Gain experience in utilizing APIs and external libraries while programming.
- Better my critical thinking and problem-solving skills.

# Research Methods
- Google Search Engine – Google Custom Search API.
- Web Scraping – BeautifulSoup Python library.
- Text Summarization and Processing – ChatGPT API.
- Data Compilation – pandas and json Python libraries.
- Documentation – the inputs, parameters, and outputs of each function were summarized in a PDF file to ensure transparency and replicability.
- Content Validation –  sample outputs were manually cross-referenced to assess the overall accuracy.

# Data and Results
- Through testing so far, I can estimate that approximately 40% of U.S. state legislator candidates between 1967 and 2017 had biodata available on the web. This percentage generally increases as one searches for more recent candidates.
- Based on the manual verification of the data produced for about 700 candidates, I can say that the final output is approximately 70% accurate.
- If the candidates are filtered to only those who ran after 1998, this accuracy jumps to nearly 80%.

# Data Loss
- 92% of candidates appear in the final output.
- 2% of candidates lost their data due to “prompt errors”, where the ChatGPT prompt was too long for the API.
- 4% of candidates lost their data due to “parse errors”,  where the response is in the wrong format.
- 2% of the candidates lost their data because of issues scraping their source URLs.
- The only way to get the missing data (for now) is to manually search for the candidate information.

# Major Conclusions
- It is worrying that only 40% of U.S. state legislators have biodata available online. Who are these people who are getting elected and making decisions in our lives?
- The methodology used in this project is extremely powerful and can serve as a general blueprint to create large-scale information databases related to any topic that can be searched on the web.
- Although large language models like ChatGPT are extremely powerful tools, their tendency to hallucinate necessitates a cautious and critical interpretation of their responses.
- Human intervention can improve the quantity and quality of the data output, emphasizing the role of humans in monitoring AI in this era of technology.

# Future Research Directions
- There is room to improve the selection process for the source URLs of the candidates - currently, the first 4 results are chosen for the sake of convenience.
- A better web scraping algorithm could be created to reduce the number of candidates whose source URLs cannot be processed.
- With additional funding, it would be possible to use a ChatGPT API model with a larger token context, allowing for larger prompt inputs and greater candidate coverage.
- A robust methodology needs to be developed in order to verify the accuracy of the gathered data. Currently, manual verification is the only way to know if the gathered data is accurate.

# Research in Context
- My mentor, Professor Jetson Leder-Luis, is working with a team of researchers to assess the impact of revolving door lobbying laws on the composition of U.S. state legislators.
- The biodata I am gathering will be used as part of the evaluation of the background and credentials of all the candidates.
- Specifically, the research team will analyze the biodata using machine-learning techniques to find hidden patterns.

# References
- Google’s API Documentation
- OpenAI’s API Documentation
- API Documentation for the utilized Python libraries
- Google, YouTube, and ChatGPT

# Acknowledgements
- Professor Jetson Leder-Luis (Boston University)
- Professor Raymond Fisman (Harvard University)
- Professor Silvia Vannutelli (Northwestern University)
- Catherine O’Donnell
- Undergraduate Research Opportunities Program
- Boston University
