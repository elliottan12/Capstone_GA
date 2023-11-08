# Capstone (Grain Business Dashboard)

### Overview

The F&B industry in Singapore is valued at $13 billion a year. The food catering industry specifically is picking up post-covid19 and is expected to grow in the next few years. However, there are many issues that catering companies have to solve. We will focus on 3 issues.

1. Inventory management 
2. Price negotiation 
3. Food quality: Delivery driver allocation 

---

### Problem Statement

Given past data, can we forecast demand to efficiently carry out inventory management?
What is the recommended price for customers from different sectors?
Is there a way to automatically assign drivers for deliveries to pick up hot food at the right timings and to effectively distribute the deliveries so that only a final check is needed?

---

### Datasets

For the purpose of this analysis, I managed to get the data for mini buffets in the year 2022. I have cleaned the dataset and will be using the imputed_data.csv file. 

Please refer to data dictionaries below for the full infomation found in the datasets.	

---

### Data Dictionary 

|Feature                        	| Type   |Dataset      |Description |
|:------------------------------	|:-------|:------------|:-----------|
|event_id		                 	|string  |imputed_data |This holds the unique id of each event|
|event_state                    	|string  |imputed_data |This holds the status of the event (Published)|
|sales_type                     	|string  |imputed_data |This shows what type of sale is it (Inbound / Outbound / Repeat) |
|sector			            		|string  |imputed_data |This shows which sector does the company belong to |
|company_name 						|string  |imputed_data |This shows the company's name|
|customer_id		 				|int     |imputed_data |This holds the id of each customer|
|cost_per_pax			 			|float   |imputed_data |Cost per pax |
|pax 								|float   |imputed_data |Number of people the customer has ordered for |
|address 						 	|string  |imputed_data |Full address of the customer's event location |
|postal_code 						|int     |imputed_data |Postal code of the customer's event location|
|special_instructions 				|string  |imputed_data |Holds the special instructions and simple descriptioon of what to pack |
|pickup_date_time					|string  |imputed_data |Timestamp of when the delivery is to be picked up|
|serving_date_time					|string  |imputed_data |Timestamp of the serving time |
|revenue 							|float   |imputed_data |Holds the revenue from each event |
|postal_code_new 					|int     |imputed_data |Postal code of customer's event location with leading zeros for those with only 5 digits |

---

### Conclusion

After analysis, I found that the revenue is growing and that most of the deliveries go to the central business district. This is probably due to the massive amount of offices with working adults in that area. I managed to train 2 models. 1 model was used to predict the number of pax that Grain can expect in a week. The other model was used to predict the price of a meal box to a company since they offer differientiated pricing. 

With these models, I managed to build a dashboard where the information is easily available. 

---

### Recommendations

Phase 1 - Improve the model
- Gather more specific data about the food items and their costs.
- Train the model again to improve the accuracy

Phase 2 - Deploy the solution
- The sales department can get recommended prices.
- The fulfilment department can use it to ease their daily operations

Phase 3 - Update
- Constantly change the model and add new features.
- Add a chatbot and a recommender system.