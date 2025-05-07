# Capstone Project: Predicting Whether a Fatal Victim of a Traffic Collision in San Francisco, CA is a Pedestrian
*Clarissa*

---

## Problem Statement  
For Driving Distracted Awareness Month in April and Traffic Safety Awareness Month in August, the San Francisco Public could use an engaging way to become aware and more mindful of traffic safety, especially for pedestrians, as they are the leading type of fatal victims at traffic collisions in San Francisco. 
  
This project aims to provide a solution to this problem by:
1. Building a predictive ML model that predicts whether the fatal victim of a traffic collision on San Francisco city streets is a pedestrian
2. Presenting that data in an interactive PSA website that will help people, especially pedestrians, to become more aware and savvy about practicing traffic safety.

## The Question
How might we predict whether a fatal victim of a traffic collision in San Francisco is a pedestrian, and help the public practice better traffic safety in an engaging way?

## The Audience

The general public of San Francisco, especially those whose primary source of transportation is on foot.

---

## Repository Structure

**code:**  
This folder contains several notebooks dedicated to creating a model to address the problem statement.

- Notebook [01](./code/01-eda-fatalities.ipynb) is where I collected and cleaned data.
- Notebook [02](./code/02-data-viz-fatalities.ipynb) is where I visualized the findings of the [cleaned data](./data/cleaned_fatalities.csv).
- Notebook [03](./code/03-model-fit-train-eval.ipynb) is where I developed, fit, trained, and validated my ML model.
- Notebook [04](./code/04-findings-tech-report.ipynb) contains my Executive Summary as well as summaries of my findings from my data analysis and my modeling process.

**data:**  
This folder contains the datasets used to develop my model.

-  [cleaned data](./data/cleaned_fatalities.csv)
- [feature importance dataset](./data/feature-importance.csv)
- [coriginal data](./data/Traffic_Crashes_Resulting_in_Fatality_20250411.csv)

**images:**  
This folder contains data visualizations created during our data exploration, which can explain our findings to our audience.

- [Hourly Collisions](./images/fatal_collisions_by_hour.png)
- [Age and Collision Categories](./images/fatalities-age-collion-categories.png)
- [Fatalities by Age and Victim Type](./images/fatalities-by-age-victim-type-box-chart.png)
- [Fatalities by Collision Category](./images/fatalities-collision-category.png) 
- [Fatalities Geospatial Map](./images/fatalities_geographic.png)
- [Fatalities Heatmap](./images/fatalities-heatmap.png)
- [Fatalities by Sex](./images/fatalities-sex.png)
- [Fatalities by Time of Day](./images/fatalities-time-of-day.png)
- [Fatalities by Victim Type](./images/fatalities-victim-type.png)
- [Fatalities Yearly](./images/fatalities-yearly.png)

**other:**  

- [Slides](/slides.pdf)

- README.md - The file you are reading right now. It contains the summaries, techniques used, data visualizations, conclusions, recommended next steps, and appendix of our project.

---

# Sources  
[More about Vision Zero](https://www.visionzerosf.org/maps-data/)
  
[gitignore Template Maker](https://www.toptal.com/developers/gitignore/api/jupyternotebooks,macos,virtualenv,windows)  
  
[San Francisco Traffic Crashes Resulting in Fatality](https://data.sfgov.org/Public-Safety/Traffic-Crashes-Resulting-in-Fatality/dau3-4s8f/about_data)  
