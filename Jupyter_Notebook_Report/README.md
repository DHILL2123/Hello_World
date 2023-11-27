# What Is Affecting Churn

# Project Description 

In business your customers are you life force so the more of them you retain the better.Price and quality of service or product are usually driving
factors in keeping or losing customers. We would love to keep all of our customers satisfied and happy but at the end of the day that usually isn't
very practical. We could look into factors that are potentially causing our customers to leave our company and try and mitigate some of those reasons.
I have decided to find potential reasons our customers are leaving and provide recommendations on what we could do to keep more customers that might be on the fence about our services. 

# Project Goal

* Discover what is driving churn amongst our customers
* Use those drivers to create a machine learning model classify whats making a customer leave
* Churn is described as stay or leave based on certain factors
* We could use this information to target at risk of churn customers

# Thinking Process
* My hypothesis is that churn is directly affected by certain drivers in our data
* Highlight those drivers and verify my thought

# My Plan

* Aquire the data from telco
* Prepare the data
  * Create engineered columns from the data
    *tenure
    *monthly_charges
    *total_charges
    *contract_type_id


# Exlpore the data to find the drivers influence on churn
  * Questions to answer
  * How does tenure affect churn?
  * Are there clear areas of oppurtunity for improvement in this driver?
  * How does monthly charges affect churn?
  * Is there a particular price range for keeping more customers?
  * How does total charges affect churn?
  * Is there a total amount customers aren't willing to pay?
  * Could more contracts help us improve churn?

# Created a model to predict if a customer would churn
  * Built a predictive model using selected drivers
  * Evaluated the model on the train and validate dataset
  * Selected the best model based on high accuracy
  * Evaluated the highest accuracy model on the test dataset
 
 # Data Dictionary
 
  *Feature : Definition 
 
  *Tenure  : The number of months a customer is with the company
  Monthly
 
  *Charges : The total amount the customer pays per month
  Total
  Charges : The total amount the customer pays over their tenure
 
  *Contract
  Type ID : Separates service as month-to-month, one-year contract, or two-year contract.
 
 
 # Steps to reproduce
  * Clone the repo
  * Acquire the data from mysqlworkbench(you'll need your own credentials) or .csv
  * Add the data to the file containing the cloned repo
  * Run the notebook

# Takeaways and Conclusions
  * Tenure has a direct correlation to churn
  * Majority of churn happens in the first month through the first six months
  * Monthly charges, total charges, and contract type are also drivers of churn
  * In conclusion knowing whats causing customers to leave gives use an oppurtunity to limit churn

# Recommendations 
  * In order to limit churn we need to dive deeper into each of our drivers looking for whats keeping some customers and compare.
  * Work on keeping our churn customers features in line with our customer who stay with the company. 
 




