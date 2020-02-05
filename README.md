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

Newark Liberty International Airport ranked as the worst US airport in terms of delays (NY Post, 2019). According to Business Insinder (2019), only 63.9% of flights departing from the airport were on-time. United Airlines is the airport's biggest operator (with 64% market share) but it also has the worst on-time performance, followed by Southwest and American Eagle. Furthermore, the airline's public image had been scrutinised since 2017. The brand's reputation quickly dropped after a video of a passenger being forcefully dragged off the plane went viral and the fact that a puppy was forced to be put in an overhead bin leading to its death. 

Due to these incidents, the percentage of customers considering flying with has United Airlines has dropped significantly down to 25% which accounts for 11% below the domestic airline industry average (Media Post 2018).

### 3. Target Audience

Our target audience is United Airlines. Other airlines operating at Newark Liberty International Airport and the airport management are also encouraged to make use of this report.

### 4. Project Objectives

In this report, we attempt to determine factors affecting United Airlines' delays (both departure and arrival) at the airport. Flight delays have negative impacts for airlines, airports and customers. Being able to predict flight delays will help passengers help their time and cost. If the airline can warn its passengers in advance, it can be a great support for their marketing strategy as airlines rely on cusomers' loyalty to support their frequent-flyer programmes. On the other hand, airlines will also be able to avoid penalties, fines, additional operational costs such as crew and aircrafts retention at the airport. Furthermore, environmentally flight delays cause damage by increasing fuel consumption and gas emission.

For this purpose, we have chosen the 2015 Flight Delays and Cancellations provided by the Department of Transportation on Kaggle which contains three csv fies of airports, airlines and flights. In combination with this dataset, we have exploited the Dark Sky API and plane registration dates from FAA to see whether weather and plane age or condition would affect the airline's delays or not.

### 5. Methodology

We attempted to use Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boost and SVM to classify whether a flight would be delayed or not. After comparing the results from the different models, we chose Random Forest as the best model. Based on our findings, we would be able to identify the biggest factors to delayed flights. Note that our model and test results are currently limited to resource limitations, both in terms of time and computational power. We are confident that the results could be improved further if these challenges are addressed.

### 6. Findings

Our main delay predictors are the time of flight schedule during the day, day of the week, distance of flight route and ground temperature. Flight delays often occurs 3pm and 9pm and they are more likely on Tuesdays. Surprisingly, high temperature have a higher impact on delays compared to colder and potentially snowy days. If temperature goes above 35 degrees Celsius, delays will happen more often. This can be explained by the fact that during winter, airlines will be more prepared for snow and rain. However, in the case of extremely hot weather the crew are not meant to work in the heat for too long due labour laws and union agreements. This is further exacerbated due to the fact that there are more flights during warmer summer months than in the winter.

### 7. Classification model results

<img src="images\model_results.png>

### 7. Recommendations

From our studies, we propose the following recommendations to United Airlines operating from its base in Newark Airport:
1. Due to the higher likelihood of delays between 3pm to 9pm, there should be more stand-by aircrafts based at the airport in order absorb some of these delays by providing alternative aircraft without having to rely mainly on incoming aircrafts that will most likely be delayed.
2. As flights on Tuesdays are more likely to be delayed, additional casual or subcontracted staff should be employed in order to support existing full time staff at the airport, particularly in handling customer service issues. Delayed flights cause inconvenience to passengers and thus additional customer service representatives at the counters and all around the airport could help to improve the airline's image by providing the latest information and offer alternative assistance where applicable.
3. While weather-related delays could anticipated using advanced weather warning system, delays stemming from hot summer days can be addressed by performing more maintenance checks during the evening during cooler temperature. Additional summer traffic to leisure destinations can be scheduled during less busy hours in the airport in order to free up the slots for other regular traffic in order to reduce the bottlenecks at critical hours during the day.


