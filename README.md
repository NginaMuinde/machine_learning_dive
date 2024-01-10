## TOPIC: Customer Churn Prediction and Retention Strategies for Vodafone Corporation

#### Project Understanding and Description:
In order for Telco companies to grow their revenue generating base, it is important for them to attract new customers and at the same time avoid contract terminations (known as churn). In the real world different reasons cause customers to terminate their contracts, for example; better price offers and more interesting packages from competitors, bad service experiences with current provider or change of customers’ personal preferences and situations.

#### Project Objective:
The main objective in this project based on the data provided, is to leverage machine learning models to predict customer churn within Vodafone Corporation, a leading telecommunication company. Customer churn or the loss of customers over time, is a critical concern and understanding the factors influencing churn can inform proactive retention strategies for our client.

Our task is to train machine learning models to predict churn on an individual customer basis and take counter measures such as discounts, special offers or other gratifications to keep their customers. A customer churn analysis is a typical classification problem within the domain of supervised learning.

In this project, a basic machine learning pipeline based on a sample data set from Kaggle is build and performance of different model types is compared. The pipeline used for this example consists of 8 steps:

Step 1: Problem Definition

Why do customers churn? We need to establish the reasons why customers churn and train a machine model to be able to predict whether a customer will churn.
<img width="1440" alt="Screenshot 2024-01-10 at 13 06 51" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/1e3624ff-d6fc-48ce-bba7-2409bf7b315f">

Step 2: Data Collection
We have three sources of data.
The first and second data are for machine training the third data is for machine testing.

First I started by importing the relevant libraries and modules

<img width="1388" alt="Screenshot 2024-01-10 at 13 56 30" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/f72ace47-4e35-4f1d-a967-1e85ff18951d">

Then did the data loading of all three datasets

<img width="1440" alt="Screenshot 2024-01-10 at 13 19 59" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/2779c365-f879-4e8f-9bf4-6f4c8905f0fd">

<img width="1440" alt="Screenshot 2024-01-10 at 13 23 57" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/838a12a9-31dc-45f5-ad29-41014d2c5937">

<img width="1440" alt="Screenshot 2024-01-10 at 13 29 15" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/12a3c102-976c-4798-ab97-f02e4eb29d2b">

<img width="1440" alt="Screenshot 2024-01-10 at 13 31 49" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/e625b61c-8209-4184-af06-5460d65d6697">

<img width="1440" alt="Screenshot 2024-01-10 at 13 32 01" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/68baaa15-ee15-441a-acec-e15db6bedef2">

<img width="1440" alt="Screenshot 2024-01-10 at 13 32 07" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/ba7d8364-b8a7-4000-9181-d32eef7dcabb">


<img width="1440" alt="Screenshot 2024-01-10 at 13 32 14" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/f6978eae-8ecb-4342-8537-d37faacf900e">

<img width="1440" alt="Screenshot 2024-01-10 at 14 22 25" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/aa010b21-44a4-4a6c-a5b9-a884d0ab5deb">

<img width="1440" alt="Screenshot 2024-01-10 at 14 36 42" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/c1d26799-e8c7-43ec-aea6-09dd8af817fa">


<img width="1440" alt="Screenshot 2024-01-10 at 14 37 10" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/8eec406d-0da7-4681-bcd9-37d89acc69f6">


<img width="1440" alt="Screenshot 2024-01-10 at 15 20 14" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/2757478b-b1e3-4bed-8ad9-e3ccbf98ef1e">

<img width="1440" alt="Screenshot 2024-01-10 at 15 20 24" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/06195d3e-4479-4f42-9bac-b43b62a8d641">

<img width="1440" alt="Screenshot 2024-01-10 at 15 20 32" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/d40e61fc-5d45-4827-8b4b-62348d9b504c">

<img width="1440" alt="Screenshot 2024-01-10 at 15 20 44" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/95ee3baf-a96f-40d3-8ba2-731c8bcb9cee">

<img width="1440" alt="Screenshot 2024-01-10 at 15 21 13" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/10739c52-e6b8-40d8-99f7-f82c3ede18c4">

<img width="1440" alt="Screenshot 2024-01-10 at 15 21 25" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/43ebd22a-d3c3-49db-a832-eba594d6de1c">



Step 3: Exploratory Data Analysis (EDA)

Started by merging the first and second datasets

<img width="1440" alt="Screenshot 2024-01-10 at 13 38 27" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/c21da504-1289-493c-845d-b314edaafff0">


<img width="1440" alt="Screenshot 2024-01-10 at 13 38 40" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/7f88cd14-3243-4c9b-91df-8e471df9cadf">


<img width="1440" alt="Screenshot 2024-01-10 at 13 38 52" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/2cd4fd33-fc6b-4937-9219-827aa6550d27">


<img width="1440" alt="Screenshot 2024-01-10 at 13 39 04" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/f25ab94a-8ab9-4b96-8357-045b0d542889">


<img width="1440" alt="Screenshot 2024-01-10 at 15 21 37" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/35bcab0c-2ae1-4e1c-8075-fe2c1a1c5680">


<img width="1440" alt="Screenshot 2024-01-10 at 15 21 43" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/a12b644c-61e4-4e4c-9ae6-635133181def">


<img width="1440" alt="Screenshot 2024-01-10 at 15 21 49" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/63d7837d-7630-4309-ad19-612f9b47e194">


<img width="1440" alt="Screenshot 2024-01-10 at 15 21 56" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/16dff81e-b386-4d17-b64e-5e09d18bc0cd">


<img width="1440" alt="Screenshot 2024-01-10 at 15 22 21" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/8d76c9a3-ce10-4647-abab-b56db0dd92ff">


<img width="1440" alt="Screenshot 2024-01-10 at 15 22 39" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/7882559a-1785-4507-bc80-9f97229d1b41">


<img width="1440" alt="Screenshot 2024-01-10 at 15 22 48" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/c84b7633-0791-4f52-bf01-edc733d0d8f5">


<img width="1440" alt="Screenshot 2024-01-10 at 15 22 56" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/5b358ec2-aea6-4bb2-b06e-8a411546cf37">



<img width="1440" alt="Screenshot 2024-01-10 at 15 23 35" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/7ee5a3d1-71a4-4ee9-8472-724412348c6b">


<img width="1440" alt="Screenshot 2024-01-10 at 15 23 47" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/0be6f61d-e3d9-4f51-8244-f424e6237f4e">


<img width="1440" alt="Screenshot 2024-01-10 at 15 24 11" src="https://github.com/NginaMuinde/machine_learning_dive/assets/149095447/babd7ed9-327d-48cb-9d98-9984ca13169d">


![Uploading Screenshot 2024-01-10 at 15.24.34.png…]()



![Uploading Screenshot 2024-01-10 at 15.24.56.png…]()


![Uploading Screenshot 2024-01-10 at 15.25.10.png…]()

Step 4: Feature Engineering

Step 5: Train/Test Split

Step 6: Model Evaluation Metrics Definition

Step 7: Model Selection, Training, Prediction and Assessment

Step 8: Hyperparameter Tuning/Model Improvement

### Key Components:

#### Data Collection and Exploration:`

Gather and explore data provided by the marketing and sales teams, encompassing customer demographics, service usage, and historical churn records.

#### Hypothesis Formation:
Develop hypotheses around potential factors influencing customer churn, considering aspects such as contract duration, service-related issues, pricing, billing preferences, and customer demographics.

#### Null Hypothesis:
There is no significant difference in churn likelihood between customers with shorter and longer contract durations.

#### Alternative Hypothesis:
Customers with shorter contracts are more likely to churn than those with longer contracts.

#### Analytical Questions

##### Question 1: 
How does the length of a customer's contract term correlate with the likelihood of churn?

##### Question 2: 
What is the distribution of contract durations among customers who have churned compared to those who have not?

##### Question 3: 
What is the Proportion of customers with short contract durations using additional services

##### Question 4: 
How does the method of payment impact customer churn?

##### Question 5: 
Does the monthly and Total charges affect the probability of a customer churning?

#### `Data Preprocessing and Feature Engineering:`
Cleanse the data, handle missing values, and engineer relevant features to enhance model performance.


##### Model Building:
Select and train machine learning models to predict customer churn based on historical data.


##### Evaluation and Key Indicators:
Evaluate the models' performance and identify key indicators contributing to customer churn.


##### Retention Strategies:
Collaborate with business development, marketing, and sales teams to derive actionable insights from the models and formulate effective retention strategies.


##### Model Deployment and Integration:
Deploy the trained models into Vodafone's systems for real-time or batch predictions, integrating them seamlessly into the business processes.


##### Documentation and Reporting:
Document the entire data science process, including preprocessing steps, model selection, and deployment procedures. Provide clear and concise reports on model performance, key indicators, and recommended retention strategies.
#### Export key components

##### Continuous Improvement:
Establish a feedback loop for continuous improvement, iterating on models and strategies based on real-world feedback and evolving business dynamics.

By the end of this project, we aim to equip Vodafone Corporation with predictive capabilities to anticipate customer churn and implement targeted retention measures, ultimately fostering customer satisfaction and business sustainability.
