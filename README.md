# Flight delays at Newark Liberty International Airport

### 1. List of key files:
1. Main Jupyter notebook : **flight_delay_notebook.ipynb**
2. Formulas Jupyter notebook :  **library.py**
3. Presentation in <a href="https://docs.google.com/presentation/d/1t2DY1rbv1-DBIsj7A76iKW4GZFG_0Hob4WD-7mGCARg/">Google Slides</a> format
4. Presentation in PDF: **Presentation_slides.pdf**
5. Saved trained models in Pickle : **/pickle folder**
6. Main airline delay data: **newark_flights.csv**
7. Weather API data import Jupyter notebook: **dark_sky.ipynb**
8. Aircraft data import Jupyter notebook: **plane_registration.ipynb**

### 2. United Airlines at Newark Airport

Newark Liberty International Airport is ranked as the worst US airport in terms of delays in 2019 according to New York Post whereby only 64% of flights departed on time. Our stakeholder, United Airlines, is not only by far the airport's biggest operator, but also has the airport's worst on-time performance compared to its competitors. The issue with flighy delays certainly does not help with the airline's public image, which was made worse by the release of a video of a passenger being forcefully dragged off the plane that went viral and in the case when a puppy was forced to be put in an overhead bin that lead to its death.

### 3. Project Objectives

In this report, we seek to determine factors affecting United Airlines' delays (both departure and arrival) and if these factors could be reliably used as predictors to forecast if a given flight is delayed. Flight delays have negative impacts for airlines, airports and customers. For the airline, the ability to warn its passengers in advance and to be prepared with contingency plans would help them to retain the loyalty of its customer base and maintain a good public image. On the costs side of the equation, the airline will also be able to be better prepared to manage compensations, penalties and additional operational costs such as crew and aircraft retention.

### 4. Methodology

We have chosen flight delays data for full year 2015 from Dept of Transportation via Kaggle. In addition, we also brought together data from Dark Sky weather API and aircraft registration data from FAA to deep deeper into the effects of weather conditions and the airline's fleet.

After data cleansing, EDA and feature selection processes, we explored several classification models including Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boost and SVM using train and validation data sets to classify whether a flight would be delayed or not. For the first three models, we also employed grid search with k-fold cross-validation technique to find the optimum value for the hyperparameters. After comparing the results from the different models, we decided upon Random Forest as the best model. The final stage of the the process is to identify the appropriate threshold for our model based on the airline's business needs and perform a final evaluation of the model using test data set.

### 5. Findings

Our main delay predictors are the time of flight schedule during the day, day of the week, distance of flight route and ground temperature. Flight delays often occurs 3pm and 9pm and they are more likely on Tuesdays. Surprisingly, high temperature have a higher impact on delays compared to colder and potentially snowy days. If temperature goes above 35 degrees Celsius, delays will happen more often. This can be explained by the fact that during winter, airlines will be more prepared for snow and rain. However, in the case of extremely hot weather the crew are not meant to work in the heat for too long due labour laws and union agreements. This is further exacerbated due to the fact that there are more flights during warmer summer months than in the winter.

### 6. Classification model results

Due to imbalanced data between on-time and delayed data, we employed a SMOTE over-sampling method during training and validation while training our models and hyperparameters tuning. Using various cost metrics including customer compensation, staff cost, penalties and other operational costs, we calculated our model threshold for our business needs to be 0.46 for the selected Random Forest model.

| Data set | Sampling | Power (TPr) | Alpha (FPr) | Beta (FNr) |
| --- | --- | :---: | :---: | :---: |
| Training data | SMOTE | 87% | 15% | 13% |
| Test data | no sampling | 54% | 22% | 46% |

At this stage, our model is only able to predict just over half of delayed flights. The high false negative is a concern as one of our key objectives is to reduce this error since it would be costlier for United Airlines to be unprepared for a flight delay when there is one, than to be slightly over-prepared for a delay when there is none.

The lower performing test results suggest that there has been some overfitting during our training stages. However, we are confident that our test results can be improved further given more time considering computational resource limitations that we faced in order to run more hyperparameter tuning tasks.

<img src="images\model_results.png">

### 7. Recommendations

From our studies, we propose the following recommendations to United Airlines operating from its base in Newark Airport:
1. Due to the higher likelihood of delays between 3pm to 9pm, there should be more stand-by aircrafts based at the airport in order absorb some of these delays by providing alternative aircraft without having to rely mainly on incoming aircrafts that will most likely be delayed.
2. As flights on Tuesdays are more likely to be delayed, additional casual or subcontracted staff should be employed in order to support existing full time staff at the airport, particularly in handling customer service issues. Delayed flights cause inconvenience to passengers and thus additional customer service representatives at the counters and all around the airport could help to improve the airline's image by providing the latest information and offer alternative assistance where applicable.
3. While weather-related delays could anticipated using advanced weather warning system, delays stemming from hot summer days can be addressed by performing more maintenance checks during the evening during cooler temperature. Additional summer traffic to leisure destinations can be scheduled during less busy hours in the airport in order to free up the slots for other regular traffic in order to reduce the bottlenecks at critical hours during the day.


