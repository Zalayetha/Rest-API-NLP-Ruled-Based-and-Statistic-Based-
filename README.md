# Documentation

## What
This API classifies text to identify the type of disaster, location, impact, and time of disaster. This API consists of two algorithms, namely: a rule-based algorithm and a statistical algorithm using the Naive Bayes classifier.

## Tech Stack
- Flask

## Endpoint
| Endpoint   | Http Method | Parameter | Type   | Description                   |
|------------|-------------|-----------|--------|-------------------------------|
| /rule      | GET         | text      | String | the text you want to classify |
| /statistic | GET         | text      | String | the text you want to classify |

## Setup
```
git clone https://github.com/Zalayetha/Rest-API-NLP-Ruled-Based-and-Statistic-Based-.git

pip install -r requirements. txt

flask run
```
