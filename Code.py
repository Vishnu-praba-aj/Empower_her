#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[1]:


def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Customize this part to extract specific data from the webpage using BeautifulSoup

        # For example, to extract text from all paragraphs:
        paragraphs = soup.find_all("p")
        text_content = "\n".join(paragraph.get_text() for paragraph in paragraphs)

        return text_content

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None


# In[3]:


#collects relevent links 
from bs4 import BeautifulSoup
import requests

def google_search(api_key, cx, query, start=1, num=10):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "start": start,
        "num": num
    }

    results = []

    while True:
        response = requests.get(base_url, params=params)
        data = response.json()

        if "items" in data:
            results.extend(data["items"])

        # Check if there are more pages
        if "queries" in data and "nextPage" in data["queries"]:
            start = data["queries"]["nextPage"][0]["startIndex"]
            params["start"] = start
        else:
            break

    return results

#  API key and custom search engine ID (cx)
api_key = "AIzaSyAvzocdMuMuKX64sgGnUUQPzM802TdDkUI"
cx = "c54ee4552c3a347f6"
query = "girls scholarship in India"

search_results = google_search(api_key, cx, query)

# Print titles and links of the results
for result in search_results:
    title = result.get("title", "")
    link = result.get("link", "")
    print(f"Title: {title}\nLink: {link}\n")


# In[ ]:





# In[ ]:





# In[5]:


#content of the link
from bs4 import BeautifulSoup
def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Customize this part to extract specific data from the webpage using BeautifulSoup

        # For example, to extract text from all paragraphs:
        paragraphs = soup.find_all("p")
        text_content = "\n".join(paragraph.get_text() for paragraph in paragraphs)

        return text_content

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

# Iterate over the URLs and scrape each website
for result in search_results:
    title = result.get("title", "")
    link = result.get("link", "")
    print(f"Title: {title}\nLink: {link}")

    # Scrape data from the webpage
    data_from_webpage = scrape_webpage(link)

    if data_from_webpage is not None:
        print("Data from Webpage:")
        print(data_from_webpage)

    print("\n" + "-"*50 + "\n")


# In[6]:


#relevent lines from the scrapped data
import spacy
def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Customize this part to extract specific data from the webpage using BeautifulSoup

        # For example, to extract text from all paragraphs:
        paragraphs = soup.find_all("p")
        text_content = "\n".join(paragraph.get_text() for paragraph in paragraphs)

        return text_content

    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

nlp = spacy.load("en_core_web_sm")

#  web page content
web_page_content = scrape_webpage("https://www.buddy4study.com/article/scholarships-for-indian-girls-and-women")

# Apply NER
doc = nlp(web_page_content)

# Extract sentences containing relevant entities
relevant_sentences = []

for sent in doc.sents:
    if any(ent.label_ in {"DATE", "MONEY", "ORG"} for ent in sent.ents):
        relevant_sentences.append(sent.text)

print(relevant_sentences)


# In[13]:


#extracts date,amount,elgibility
import spacy
from spacy.pipeline import EntityRuler
# Load the spaCy English model
def content(paragraph):
    nlp = spacy.load("en_core_web_sm")

#entity ruler


    patterns = [
    {"label": "IS_CURRENCY", "pattern": [{"LOWER": "inr"}]},  # Recognize "INR" as a currency
    # Add more patterns for other currency symbols if needed
    ]

# Example paragraph

#paragraph = "This scholarship is open to students graduating in the Dallas-Fort Worth Metropolitan area or residents of DFW only.The application submission starts on January 2, 2024, and due on May 31, 2024."
#paragraph="The Brown Girls Do Ballet® Tiny Dancer Scholarship is full-year ballet tuition for young dancers and is only available to young girls aged 4-8.', 'We know that the journey to Principal Ballerina begins with their very first class, and we hope that we can help spark the interest of a young girl who will one day lead a company.', 'Parent/Guardian Letter of Interest\nA one-minute intro video.', 'Tell us why you want to take dance!\nMust provide ballet tuition information for the program of your choice.\nProof of annual household income (such as a W-2 form; you may mark out Social Security Numbers)\n', 'The application opens April 1, 2024, and closes June 30, 2024.', 'Submissions must be received by June 30, 2024 at 11:59pm CST.\n', 'The chosen recipient will be notified by August 2024.\n', 'The Kennedy George and Ava Holloway Dance for Change Scholarship\xa0is awarded to female dancers of color ages 6-16 who have registered for a year-long dance program.', 'The Brown Girls Do Ballet® Micro Grant Program is designed to increase access to and opportunity for education, travel, and professional tools for small dance programs and dancers of all ages year-round. \n', 'Requirements:\nDance Resume\nLetter of Recommendation\nProof of annual household income (such as a W-2 form; you may mark out Social Security Numbers)\nAward amounts are currently up to $1,500.00. \n', 'If you meet the eligibility criteria (outlined above), you may apply at any time throughout the year. \n', 'The application opens on January 2, 2024, and closes on March 30, 2024.', 'Submissions must be received by March 30, 2024 at 11:59pm CST.\n', 'The chosen recipient will be notified by April 30, 2024.\n', 'The annual Brown Girls Do Ballet® Summer Intensive Scholarship is a $1,000.00 scholarship given to dancers ages 9-18 who have been accepted to and have registered for a summer intensive program.', 'The application opens May 1, 2024, and closes May 31, 2024.', 'Submissions must be received by May 31, 2024 at 11:59pm CST. \n', 'The chosen recipient will be notified by June 30, 2024.\n', 'A one-minute solo video (can be classical or contemporary)\nSummer Intensive acceptance letter and recent invoice showing payment\nProof of annual household income (such as a W-2 form; you may mark out Social Security Numbers)\n\ufeff\nThe Cydni Lawson Morris Memorial Scholarship is currently a $500 scholarship.', 'One or more scholarships will be awarded each year to a Dallas-Fort Worth (Texas) graduating high school senior and/or continuing college or vocational student.', 'This scholarship is open to students graduating in the Dallas-Fort Worth Metropolitan area or residents of DFW only.', 'The application opens on January 2, 2024, and closes on May 31, 2024.', 'Submissions must be received by May 31, 2024 at 11:59pm CST.\n', 'The chosen recipient will be notified by June 30, 2024.\n', 'The annual Brown Girls Do®Inc College Scholarship is a $2,000.00 scholarship awarded to full-time college undergraduate students across any field of study*. \n', 'The application opens May 1, 2024  and closes June 30, 2024.', 'Submissions must be received by June 30, 2024 at 11:59pm CST.\n', 'The chosen recipient will be notified by August 30, 2024.\n', 'Requirements\nEnrollment verification from a 4-year university (must be enrolled full-time)\n', 'Good Academic Standing (if you do not currently have a college GPA, you must also submit a high school report card from your final semester)\nProof of annual household income (such as a W-2 form; you may mark out Social Security Numbers)\nEssay questions\nLetter of Recommendation\n*Note: You do not need to have been a dancer to qualify.\n\xa0\nSubscribe to our newsletter for updates.'"

# Joining the strings
    paragraph=' '.join(relevant_sentences)

# Process the paragraph using spaCy
    doc = nlp(paragraph)

# Initialize variables to store extracted information
    scholarship_info = {}

    eligibility_keywords = ["eligible", "qualification", "candidates","students"]

    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    
    eligible_contexts = [token.sent.text for token in doc if any(keyword in token.text.lower() for keyword in eligibility_keywords)]
    print(eligible_contexts)
    # Iterate through entities in the processed paragraph
    openFlag=True
    closeFlag=True
    for ent in doc.ents:
    # Check for relevant information (you may need to customize this based on your data)
        if ent.label_ == "DATE":
            if "open_date" not in scholarship_info and  openFlag:
                scholarship_info["open_date"] = ent.text
                openFlag=False
            elif "close_date" not in scholarship_info and closeFlag:
                scholarship_info["close_date"] = ent.text
                closeFlag=False
        elif ent.label_ == "MONEY" :
            scholarship_info["amount"] = ent.text
    print(scholarship_info)
#print(scholarship_info)
#print("Eligibility Contexts:", eligible_contexts[0])
#print("dates:",dates)


# In[14]:


content(relevant_sentences)


# In[ ]:





# In[32]:


import re
import requests
from bs4 import BeautifulSoup
def extract_scholarships_info(text):
    scholarships = []

    # Extracting scholarships information using regex patterns
    scholarship_patterns = [
        r'(?:scholarship\s+for\s+girls).*?(?=(?:Eligibility|Apply|Purpose|Award|Join\s+Telegram|Join\s+WhatsApp|$))',
        r'(?:scholarship\s+amount\s+of\s+[^.]*).*?(?=(?:Eligibility|Apply|Purpose|Award|Join\s+Telegram|Join\s+WhatsApp|$))',
        r'(?:eligibility\s*-\s*)(.*?)(?:Apply|$)',
        r'(?:application\s*-\s*)(.*?)(?:Click|$)',
        r'(?:purpose\s*-\s*)(.*?)(?:Award|$)',
        r'(?:award\s*-\s*)(.*?)(?:Eligibility|Apply|$)'
    ]

    for pattern in scholarship_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            scholarships.append(matches)

    return scholarships
text=scrape_webpage("https://www.buddy4study.com/article/scholarships-for-indian-girls-and-women")
result=extract_scholarships_info(text)
type(result)
#print(result[1])
#amount -5 , purpose 4 , application time -3 , eligibility - 2,
result[2]


# In[ ]:


result[2]


# In[23]:


#buddy data
from bs4 import BeautifulSoup
import requests
import pandas as pd

# URL of the website
url = "https://www.buddy4study.com/article/scholarships-for-indian-girls-and-women"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Initialize lists to store scholarship information
    scholarship_data = []
    
    # Find all <h3> tags containing scholarship names
    scholarship_names = soup.find_all('h3')
    
    # Iterate over each scholarship name and its details
    for name in scholarship_names:
        # Extract scholarship name
        scholarship_name = name.text.strip()
        
        # Find all <p> tags following the scholarship name
        details = name.find_next_siblings('p')
        
        # Initialize variables to store details
        purpose = ""
        award = ""
        eligibility = ""
        apply = ""
        application = ""
        # Iterate over details to extract relevant information
        for detail in details:
    # Initialize variables to store details
            key = ""
            value = ""
   
            for content in detail.contents:
                if content.name=="strong":
                    key = content.text.strip()
                else:
                    value=content.text
                    
            if key.lower().startswith("purpose"):
                purpose = value
            elif key.lower().startswith("award"):
                award = value
            elif key.lower().startswith("eligibility"):
                eligibility = value
            elif key.lower().startswith("apply"):
                apply = value
            elif key.lower().startswith("application"):
                application = value
                scholarship_data.append({
            'Scholarship Name': scholarship_name,
            'Purpose': purpose,
            'Award': award,
            'Eligibility': eligibility,
            'Apply': apply,
            'Application': application
        })
        
                flag=True
                break
        if flag:
            continue
print(scholarship_data)


# In[44]:


#converts to excel
# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(scholarship_data)
    #print(scholarship_data)
    # Save the DataFrame to an Excel file

#
df.to_excel("scholarships.xlsx", index=False)
    
print("Excel file 'scholarships.xlsx' has been created successfully.")
    


# In[45]:


import pandas as pd

# Read the Excel file into a pandas DataFrame
df = pd.read_excel("scholarships.xlsx")
df['Eligibility']=result[2]
df['Award']=result[5]
df['Purpose']=result[4]
df=df.head(22)
df['Application']=result[3]

# Display the DataFrame
print(df.head())
print(df.columns)
print(df.shape)
#amount -5 , purpose 4 , application time -3 , eligibility - 2,
#df.dropna()
#print(df.shape)


# In[ ]:





# In[34]:


#recommendation model using kaggle dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_excel("dataset_combined.xlsx")
df=df.head(50)
# Features and target
X = df['Education Qualification'] + ' ' + df['Gender'] + ' ' + df['Community'] + ' ' + df['Religion'] + ' ' + df['Exservice-men'] + ' ' + df['Disability'] + ' ' + df['Sports'] + ' ' + df['Annual-Percentage'] + ' ' + df['Income'] + ' ' + df['India']
y = df['Name']  # Assuming 'Name' is the column containing the scholarship names

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Function to get user input and provide recommendations
def get_recommendations():
    education = input("Enter your education qualification: ")
    gender = input("Enter your gender: ")
    community = input("Enter your community: ")
    religion = input("Enter your religion: ")
    exservicemen = input("Are you an ex-servicemen? ")
    disability = input("Do you have a disability? ")
    sports = input("Do you participate in sports? ")
    percentage = input("Enter your annual percentage: ")
    income = input("Enter your annual income: ")
    india = input("Are you from India? ")

    # Combine user input into a single string
    user_input = f'{education} {gender} {community} {religion} {exservicemen} {disability} {sports} {percentage} {income} {india}'

    # Vectorize user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Predict scholarship name
    predicted_scholarship = model.predict(user_input_tfidf)[0]

    print(f"Based on your profile, you are recommended the following scholarship: {predicted_scholarship}")

# Get recommendations for the user
get_recommendations()


# In[46]:



#recommendation model using scrapped dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Read the Excel file into a pandas DataFrame

#df.fillna("na")

# Display the DataFrame


# Features and target
X =  df['Purpose'] + ' ' + df['Award'] + ' ' + df['Eligibility'] +  ' ' + df['Application'] 
y = df['Scholarship Name']  # Assuming 'Name' is the column containing the scholarship names

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Function to get user input and provide recommendations
def get_recommendations():
    Eligibility = input("Enter your education qualification: ")
    

    # Combine user input into a single string
    user_input = f'{Eligibility}'

    # Vectorize user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Predict scholarship name
    predicted_scholarship = model.predict(user_input_tfidf)[0]

    print(f"Based on your profile, you are recommended the following scholarship: {predicted_scholarship}")

# Get recommendations for the user
get_recommendations()


# In[38]:


def get_recommendations():
    Eligibility = input("Enter your education qualification: ")
    

    # Combine user input into a single string
    user_input = f'{Eligibility}'

    # Vectorize user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Predict scholarship name
    predicted_scholarship = model.predict(user_input_tfidf)[0]
    return predicted_scholarship
    
recommend=get_recommendations()
# Get recommendations for the user
print(f"Based on your profile, you are recommended the following scholarship: {recommend}")


# In[39]:


df.loc[df['Scholarship Name']==recommend]


# In[54]:


# Function to get user input and provide recommendations
def get_recommendations():
    Eligibility = input("Enter your education qualification: ")
    
    # Combine user input into a single string
    user_input = f'{Eligibility}'

    # Vectorize user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Predict scholarship names
    predicted_scholarships = model.predict(user_input_tfidf)
    
    if len(predicted_scholarships) == 0:
        print("Sorry, no scholarships found for your profile.")
    else:
        print("Based on your profile, you are recommended the following scholarships:")
        for scholarship in predicted_scholarships:
            print(scholarship)

# Get recommendations for the user
get_recommendations()


# In[52]:


df['Eligibility']
# state, nation , education , percent , age , caste 


# In[1]:





# In[ ]:





# In[2]:





# In[3]:





# In[ ]:




