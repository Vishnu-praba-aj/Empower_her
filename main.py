import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model and vectorizer
df = pd.read_excel("scholarships.xlsx")
df['Eligibility'] = df['Eligibility'].fillna('')  # Replace NaN values with empty strings
df['Award'] = df['Award'].fillna('')
df['Purpose'] = df['Purpose'].fillna('')
df['Application'] = df['Application'].fillna('')

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(df['Purpose'] + ' ' +
   df['Award'] + ' ' +
   df['Eligibility'] + ' ' +
   df['Application'])

model = RandomForestClassifier()
model.fit(X_train_tfidf, df['Scholarship Name'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    eligibility = request.form["eligibility"]
    user_input = f'{eligibility}'
    user_input_tfidf = vectorizer.transform([user_input])
    predicted_scholarship = model.predict(user_input_tfidf)[0]
    return render_template("result.html", recommendation=predicted_scholarship)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
