# IMDb Reviews Sentiment Analysis

## Data Processing and Cleaning

### Processing Steps
1. **Loading Data**:
    - Load the reviews from the IMDb website using `requests` and `BeautifulSoup`.
    - Store the reviews in a list and then into a pandas DataFrame.
  
    ```python
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    def get_imdb_reviews(movie_id):
        url = f'https://www.imdb.com/title/{movie_id}/reviews'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        review_containers = soup.find_all('div', class_='lister-item mode-detail imdb-user-review collapsable')
        review_data = []

        for container in review_containers:
            review_text = container.find('div', class_='text show-more__control').get_text()
            rating_element = container.find('span', class_='rating-other-user-rating')
            rating = rating_element.find('span').get_text() if rating_element else 'N/A'
            sentiment = TextBlob(review_text).sentiment.polarity
            review_data.append({'Review': review_text, 'Rating': rating, 'Sentiment': sentiment})

        return pd.DataFrame(review_data)
    ```

2. **Cleaning Data**:
    - Remove unwanted characters, HTML tags, and punctuation from the reviews.
    - Convert text to lowercase to ensure uniformity.
    - Tokenize the reviews to break down text into individual words.
    - Remove stop words (common words that do not contribute to sentiment, like 'and', 'the').
    - Lemmatize or stem the words to reduce them to their base or root form.

    ```python
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import re

    def clean_review(review):
        review = BeautifulSoup(review, "html.parser").get_text()  # Remove HTML tags
        review = re.sub("[^a-zA-Z]", " ", review)  # Remove non-letter characters
        review = review.lower()  # Convert to lowercase
        words = review.split()  # Tokenize
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]  # Remove stop words
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]  # Lemmatize
        return " ".join(lemmatized_words)
    ```

## Training a ML Classifier for Sentiment Prediction

### Steps
1. **Vectorization**:
    - Convert cleaned reviews into numerical features using techniques like TF-IDF or Count Vectorization.

    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(reviews_df['Cleaned_Review'])
    ```

2. **Label Encoding**:
    - Convert sentiment scores into binary labels for classification (positive/negative).

    ```python
    reviews_df['Sentiment_Label'] = reviews_df['Sentiment'].apply(lambda x: 1 if x > 0 else 0)
    y = reviews_df['Sentiment_Label']
    ```

3. **Train-Test Split**:
    - Split the dataset into training and testing sets.

    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. **Model Training**:
    - Train a machine learning classifier (e.g., Logistic Regression, SVM, Random Forest).

    ```python
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    ```

5. **Evaluation**:
    - Evaluate the model using accuracy, precision, recall, and F1 score.

    ```python
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ```

## Incorporating New Reviews into an Already Trained Model

### Steps
1. **Clean and Vectorize New Reviews**:
    - Apply the same cleaning and vectorization steps to new incoming reviews.

    ```python
    new_reviews = ["This movie was fantastic! A must-watch.", "Terrible movie. Waste of time."]
    cleaned_new_reviews = [clean_review(review) for review in new_reviews]
    X_new = vectorizer.transform(cleaned_new_reviews)
    ```

2. **Predict Sentiment**:
    - Use the trained model to predict the sentiment of the new reviews.

    ```python
    new_predictions = model.predict(X_new)
    ```

3. **Update Dataset and Retrain if Necessary**:
    - Append the new reviews and their predicted sentiments to the existing dataset.
    - Periodically retrain the model with the updated dataset to improve its performance.

    ```python
    new_data = pd.DataFrame({'Review': new_reviews, 'Cleaned_Review': cleaned_new_reviews, 'Sentiment_Label': new_predictions})
    reviews_df = reviews_df.append(new_data, ignore_index=True)

    # Retrain model periodically
    X = vectorizer.fit_transform(reviews_df['Cleaned_Review'])
    y = reviews_df['Sentiment_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    ```

Feel free to modify this content according to your needs or to include more details as required.
