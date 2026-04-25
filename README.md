# Email Spam Classifier using Machine Learning

## Overview

This project implements an Email Spam Classifier using Machine Learning techniques. The model is trained to classify emails as either **Spam** or **Ham (Safe Email)** based on their content.

The system uses Natural Language Processing (NLP) and a Logistic Regression model to perform text classification.

---

## Features

* Classifies emails into Spam or Ham
* Uses TF-IDF vectorization for text processing
* Trained using Logistic Regression
* Achieves high accuracy on training and testing data
* Optional interactive interface using Gradio
* Simple and easy to understand implementation

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Gradio (for UI)

---

## Dataset

The dataset used is a collection of labeled email messages:

* **Spam (0)** → Unwanted or promotional emails
* **Ham (1)** → Legitimate emails

File used:

```
mail_data.csv
```

Dataset contains:

* Category (spam/ham)
* Message (email text)

---

## Project Structure

```
Email-Spam-Classifier/
│
├── spam_model.py
├── mail_data.csv
├── email_spam_classifier.ipynb
└── README.md
```

---

## How It Works

### 1. Data Preprocessing

* Load dataset using Pandas
* Replace missing values with empty strings
* Convert labels:

  * spam → 0
  * ham → 1

### 2. Train-Test Split

* 80% training data
* 20% testing data

### 3. Feature Extraction

* TF-IDF Vectorizer converts text into numerical form
* Removes stopwords
* Converts text to lowercase

### 4. Model Training

* Logistic Regression is used
* Model is trained on extracted features

### 5. Evaluation

* Accuracy is calculated for:

  * Training data (~96.7%)
  * Testing data (~96.6%)

### 6. Prediction

* New email input is transformed using TF-IDF
* Model predicts:

  * 1 → Ham
  * 0 → Spam

---

## Installation

### Step 1: Clone the Repository

```
git clone https://github.com/Saijaswanth18/Email_spam.git
cd email-spam-classifier
```

### Step 2: Install Dependencies

```
pip install numpy pandas scikit-learn gradio
```

---

## How to Run

### Option 1: Run Python Script

```
python spam_model.py
```

### Option 2: Run Jupyter Notebook

```
jupyter notebook email_spam_classifier.ipynb
```

---

## Example

### Input:

```
Congratulations! You have won $1,000,000. Click the link to claim now.
```

### Output:

```
SPAM MAIL
```

---

## Gradio Interface (Optional)

This project includes a simple UI using Gradio where users can:

* Enter email text
* Get classification result
* View dashboard summary

To run:

```
python spam_model.py
```

If using Google Colab:

```
demo.launch(share=True)
```

---

## Code Explanation

### Key Components

#### TF-IDF Vectorizer

Converts text data into numerical vectors based on word importance.

#### Logistic Regression

A supervised learning algorithm used for binary classification.

#### Accuracy Score

Measures how well the model performs.

---

## Results

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | ~96.7% |
| Testing Accuracy  | ~96.6% |

---

## Advantages

* Simple and efficient model
* High accuracy
* Easy to extend with new datasets
* Lightweight and fast

---

## Limitations

* Depends on dataset quality
* Cannot detect very complex spam patterns
* No deep learning used

---

## Future Improvements

* Use Deep Learning models (LSTM, BERT)
* Add real-time email filtering
* Deploy as a web application
* Improve UI and dashboard

---

## Conclusion

This project demonstrates how Machine Learning and NLP can be used to solve real-world problems like spam detection. It is a beginner-friendly project suitable for academic and resume purposes.

---

## Author

Vikkurthi Sai Jaswanth
---

## License

This project is for educational purposes.
