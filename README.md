# Dengue-Outbreak-Prediction

## Overview:

The goal of this project is to predict the total_cases label for each (city, year, weekofyear) in the test set. There are two cities, San Juan and Iquitos, with test data for each city spanning 5 and 3 years respectively

## Description of Data:

Climate Data:
 The climate data includes temperature, perception and vegetationdata of Sun Juan gathered from various sources from 1990 to 2008. The climate dataset has been combined   from four open datasets of US National Oceanic and At-mospheric Administration (Driven Data, 2017).

Dengue Data:
 This includes total cases of dengue by week of the year. The datasetincludes the dengue cases of San Juan from 1990 to 2008. The dataset is used fordeveloping         prediction model.
 
Test Data:
 The dataset contains the dengue cases data with climate parameters from2009 to 2013. The dataset is used as test data for checking the efficiency of the de-veloped model.

## Materials and methods:

This project involved two datasets. The data that we are going to use for this is gathered from NOAA: (https://dengueforecasting.noaa.gov/). This dataset is publicly available for research. As part of the work, the task of analyzing climate effect on dengue and its trend is required. For the final part we build a predictive model to predict the future outbreak.

## File Structure:

EDA.ipynb: Exploratory Data Analysis of the features.

NDVI_analysis.ipynb: NDVI short for normalized difference vegetation index is used to indicate vegitation ranging from sparse vegetation (0.1 - 0.5) to dense green vegetation (0.6 and above).

Untitled1.ipynb: Dengue begins abruptly after an incubation period of 5–7 days. Created lag features to see if there is any correlation between total_cases and lag-features.

Untitled2.ipynb: Same as above but with more ndvi analysis.

Untitled3.ipynb: Statistical modelling and feature selection using RFE and VIF to check multicollinearity.

## Libraries used:

1. pandas
2. statsmodels
3. sklearn
4. matplotlib
5. seaborn
6. plotly
