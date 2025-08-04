# Formula One Machine Learning Workshop

Formula 1 Machine Learning Workshop repository! This workshop is designed to introduce participants to the exciting application of machine learning in the world of Formula One racing. Through hands-on projects, we'll explore how to use historical race data, driver performance, and car telemetry to make informed predictions and optimizations.

## Workshop Overview

This workshop is part of a multi-session event where participants will collaborate to build and refine machine learning models using Formula One data. Our aim is to apply these models to solve real-world problems like optimizing race strategies, predicting race outcomes, and improving team performance.

### Sessions
- **Session 1:** Introduction to Formula One Data and Machine Learning
- **Session 2:** Data Collection and Preprocessing
- **Session 3:** Building Predictive Models
- **Session 4:** Model Evaluation and Refinement
- **Session 5:** Final Presentations and Discussion

## Learning Objectives
- Understand the role of data in Formula One racing.
- Learn to process and cleanse data specific to Formula One.
- Develop predictive models to analyze driver performance and race outcomes.
- Evaluate and refine machine learning models.
- Present model insights and implications.

## Prerequisites
Participants are expected to have a basic understanding of Python and machine learning concepts. Familiarity with tools like Jupyter Notebooks, pandas, and scikit-learn will be beneficial.

## Tools and Resources
- **Python:** Main programming language used.
- **Jupyter Notebook:** For interactive coding sessions.
- **pandas:** For data manipulation and analysis.
- **scikit-learn:** For building machine learning models.
- **Matplotlib/Seaborn:** For data visualization.

## Installation
To get started with the project, clone this repository and install the required Python packages:
```bash
git clone https://github.com/yourgithubusername/formula-one-ml-workshop.git
cd formula-one-ml-workshop
pip install -r requirements.txt
```

## Running the F1 Fantasy Fetcher

To manually fetch F1 Fantasy data:
```bash
cd notebooks/advanced
python f1_fantasy_fetcher.py --output-dir ../../data/f1_fantasy
```

This will:
- Fetch current driver standings and statistics
- Get race-by-race performance data
- Save results to CSV files in `/data/f1_fantasy/`
- Create metadata file for tracking updates


