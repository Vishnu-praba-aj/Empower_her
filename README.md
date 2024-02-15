
![App Screenshot](https://i.pinimg.com/474x/e0/c6/dc/e0c6dc435417f37fe2ef1ff08399caa8.jpg)


# EmpowerHer - Opportunities for women
  There are many opportunities available for women, such as scholarships, free courses, career guidance, and more. However, many women are unaware of these resources and miss out on the chance to grow and succeed. 
  
  Our project aims to change that by providing a single platform where women can input their information and discover the opportunities, they're eligible for. This platform is designed for various types of women, including students, working professionals, aspiring entrepreneurs, and those returning to the workforce after a break.
	
  To start, we are focusing on college students, helping them find scholarships, special women-centric courses, distance learning options, and internships. Our goal is to make these opportunities more accessible to women and empower them to achieve their goals.


## Inclusion of Intel's OneAPI
**Performance Optimization:**
`oneAPI` provides `Intel cloud` that can significantly boost the performance of the platform. This is particularly important when matching user profiles with relevant opportunities and generating real-time recommendations, ensuring a faster and more responsive user experience.

**Real-Time Recommendation Engine:**
The optimization provided by oneAPI enhances the real-time recommendation system. Students can receive instant and relevant suggestions for educational programs, career opportunities,internships and jobs etc. improving their user experience.

**Future-Proofing:**
Incorporating `oneAPI` future-proofs your project. It ensures that the platform remains adaptable to future hardware innovations, ensuring its long-term viability in the ever-evolving technology landscape.

## Data scraping
Data scraping involves collecting information about scholarships from various sources such as scholarship websites, forums, and educational institutions.
This information typically includes details about eligibility criteria, award amounts, application deadlines, and other relevant information.
The collected data is then stored in a structured format, such as a spreadsheet or database, for further processing and analysis.
## Dataset
The model is trained on a dataset called `scholarships.xlsx` containing scholarship information such as eligibility criteria, award amount, purpose, and application details.The dataset is appropriately formatted and contains the necessary information for training the model.
## API Reference
#### Get all items

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` |  AIzaSyAvzocdMuMuKX64sgGnUUQPzM802TdDkUI |




## Machine Learning

In this project, a RandomForestClassifier model is trained using features extracted from the scholarship data.
The RandomForestClassifier is a supervised learning algorithm that can classify input data into multiple classes based on the features provided.
## Usage

Users interact with the Scholarship Recommendation System through a web interface.
They input their education qualifications into the system, which processes the input using the trained machine learning model.
The system then generates personalized recommendations for scholarships that match the user's qualifications and preferences.
Users can review the recommended scholarships and access additional information or application links for further consideration.
The system provides a convenient and user-friendly way for students and individuals to discover relevant scholarship opportunities tailored to their needs.



## Business Understanding

The goal of the project is to develop a Scholarship Recommendation System that provides personalized recommendations to users based on their education qualifications.
The system aims to help students and individuals find suitable scholarships that match their educational background, interests, and eligibility criteria.
By providing tailored recommendations, the system assists users in accessing financial aid for their education, ultimately supporting their academic pursuits and career aspirations.
## Feature engineering
Feature engineering involves transforming raw data into informative features that can be used by machine learning algorithms.
In the context of the Scholarship Recommendation System, features includes eligibility criteria, award amounts, purpose statements, application requirements, etc.

Feature engineering techniques such as *text preprocessing*, *tokenization*, and *TF-IDF* (Term Frequency-Inverse Document Frequency) vectorization were applied to convert textual features into numerical representations suitable for machine learning.
## Lessons Learned

While building this project, I learned how to scrape scholarship information from websites using Python and BeautifulSoup. I faced challenges in accurately extracting the scholarship details due to variations in HTML structure across different websites. To overcome this, I utilized BeautifulSoup's methods to navigate and extract relevant information effectively.

