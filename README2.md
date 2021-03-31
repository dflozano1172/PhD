# Requirements
Install of `urllib`, ´ssl´, ´htmldate´, ´BeautifulSoup´,´tika´,´pdfminer´ you can do using the command

```python
pip install urllib
pip install ssl
pip install htmldate
pip install BeautifulSoup
pip install tika
pip install pdfminer
```
# Objective
The Script _Sentences_Scrapper_V2.ipnby_ extracts the sentences/paragraphs from a set of URL that contain keywords specific to each URL. 

# Input
In the same folder as the script, must be a CSV file named _OutputData_clean2.csv_ . This file should cointain at least two columns one called *Url* and other *Keyword*. The Url column must have the *URL* of websites that want to be extracted. The column *Keyword* contain the set of keywords to look at the URL in the same row. More than one key word can be specified and each "keyword" can have more than one word. Same keyword words are connected by an underline, and each keyword is separated by space. For example

| *URL* | *Keyword* | 
| https://www.gov.uk/government/collections/secure-by-design | CoP_consumer_IOT CoP_consumer_IoT COP_consumer_IOT |
| https://iotsecuritymapping.uk/code-of-practice-guideline-no-11/ | DCMS_IoT DCMS_IOT |

Each row is independent. In the first row the system will look 3 different keywords "CoP consumer IOT", "CoP consumer IoT", "COP consumer IOT" in the content of the URL https://www.gov.uk/government/collections/secure-by-design

# Output

3 files will be created 
+ _extraction_results.csv_   : information of response, publication date and result of extraction
+ _List_sentences.csv_       : Each row presents the idx of the URL, the position in the website content, and the sentence. The URL can be obtained from _extraction_results.csv_
+ _Extraction_result.pickle_ : Infromation for further processing
  + extrac_restult - pandas dataframe
  + web_texts - list of strings
  + lst_sentences - list of lists 

