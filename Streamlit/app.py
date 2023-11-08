# Import the libraries
import streamlit as st 										# This helps us to structure the app
from streamlit_option_menu import option_menu 				# This lets us use a navigation bar 
from pathlib import Path 									# This lets us use the file path 
import pandas as pd 										# To use dataframes
import pickle 												# To get our trained model
from prophet import Prophet
import streamlit.components.v1 as components
import matplotlib.pyplot as plt # Used to plot bar charts
import plotly.express as px
import numpy as np
from xgboost import XGBRegressor
import requests
from datetime import datetime, time, timedelta
import datetime


# Import dataset
data_path = Path(__file__).parent / 'data/imputed_data.csv'
df = pd.read_csv(data_path,lineterminator='\n')

# Import models
# model_filepath = Path(__file__).parent / 'models/prophet_model.pkl'
# m = pickle.load(open(model_filepath, 'rb')) 

model_filepath = Path(__file__).parent / 'models/best_xgboost_model.pkl'
model = pickle.load(open(model_filepath, 'rb')) 

# Page configurations
st.set_page_config(
	page_title='Grain Business Dashboard',
	page_icon='🌾',
	layout='wide',
	initial_sidebar_state='expanded'
	)

# Page title 
st.title('Grain Business Dashboard')
style = "<div style='background-color:green; padding:2px'></div>"
st.markdown(style, unsafe_allow_html = True)

# This creates a navigation bar for the user to choose from
selected = option_menu(
	menu_title=None,
	options=["Insights", "Sales", "Fulfilment"],
	icons=['glasses', 'bar-chart', 'pin-map'],
	default_index = 0,
	orientation='horizontal',
	# Define the styles
	styles = {
	'nav-link-selected': {
		'background-color': 'green',  
		'color': 'white'  
    	}
	}
	)

if selected == 'Insights':
	# Tab to switch between live and historical data
	where, when, who = st.tabs(['Where', 'When', 'Who'])
	with where:
		# Plot the heatmap for the deliveries in 2022
		st.subheader('Heatmap of the deliveries in 2022')
		p = open("data/2022_delivery_heatmap.html").read()
		# Use CSS classes for styling
		st.write(f'<style>div.plot-container {{ width: 800px; height: 600px; }}</style>', unsafe_allow_html=True)
		components.html(p, height=700)

		# Plot the heatmap for the deliveries in December 2022
		st.subheader('Heatmap of the deliveries in December 2022')
		p = open("data/december_2022_delivery_heatmap.html").read()
		# Use CSS classes for styling
		st.write(f'<style>div.plot-container {{ width: 800px; height: 600px; }}</style>', unsafe_allow_html=True)
		components.html(p, height=700)

		# Plot the heatmap for the deliveries in week 1 of December 2022
		st.subheader('Heatmap of the deliveries in week 1 of December 2022')
		p = open("data/4-10_december_2022_delivery_heatmap.html").read()
		# Use CSS classes for styling
		st.write(f'<style>div.plot-container {{ width: 800px; height: 600px; }}</style>', unsafe_allow_html=True)
		components.html(p, height=700)

		# Write some insights
		st.write("There seems to be 3 hot spots for delivries:")
		st.markdown("1. Central business district (Raffles place mrt) - This could be because there are many companies in that area.")
		st.markdown("2. Education district (Kent ridge mrt) - This could be because there are 2 universities and many companies in that area.")
		st.markdown("3. Residential district (Sengkang mrt) - This could be because there is a hosipital and young working adults in that area.")

	with when:
		# Create columns to separate meat and vegan
		left_column, right_column  = st.columns(2)

		# Convert 'pickup_date_time' to datetime
		df['pickup_date_time'] = pd.to_datetime(df['pickup_date_time'])

		# Filter deliveries for the year 2022
		deliveries_2022 = df[df['pickup_date_time'].dt.year == 2022]

		# Group the DataFrame by month and year
		deliveries_2022['year'] = deliveries_2022['pickup_date_time'].dt.year
		deliveries_2022['month'] = deliveries_2022['pickup_date_time'].dt.month

		# Plot the yearly and weekly trend on the left 
		with left_column:
			st.subheader("Yearly trend")
			# Group by year and month, and aggregate the data
			monthly_aggregated = deliveries_2022.groupby(['year', 'month']).agg({
			    'revenue': 'sum',
			    'pax': 'sum'
			}).reset_index()

			# Rename columns for clarity
			monthly_aggregated = monthly_aggregated.rename(columns={
			    'revenue': 'total_revenue',
			    'pax': 'total_pax'
			})

			# Calculate the revenue per pax
			monthly_aggregated['revenue_per_pax'] = monthly_aggregated['total_revenue'] / monthly_aggregated['total_pax']

			# Create the data for the line chart
			x = monthly_aggregated['year'].astype(str) + '-' + monthly_aggregated['month'].astype(str).str.zfill(2)
			y = monthly_aggregated['total_revenue']

			# Create the plot using Matplotlib
			fig, ax = plt.subplots(figsize=(6, 4))
			ax.plot(x, y, color='green', alpha=1)
			ax.set_xlabel('Months', color='white')  # Set x-axis label color to white
			ax.set_ylabel('Total revenue', color='white')  # Set y-axis label color to white
			ax.set_xticklabels(ax.get_xticks(), color='white')
			ax.set_yticklabels(ax.get_yticks(), color='white')  # Set y-axis tick label color to white
			ax.set_facecolor('none')  # Make the figure background transparent
			fig.patch.set_alpha(0)  # Make the figure background transparent

			# Display the plot in Streamlit
			st.pyplot(fig)

			# Write some insights
			st.write("Across the year of 2022, the revenue has been going up and this is probably due to the easing of government restrictions. To meet the increase in demand, Grain could hire more people in the later part of the year.")


			st.subheader("Weekly trend")
			# Convert 'pickup_date_time' to datetime
			df['pickup_date_time'] = pd.to_datetime(df['pickup_date_time'])

			# Filter deliveries for the year 2022
			deliveries_2022 = df[df['pickup_date_time'].dt.year == 2022]

			# Define a function to calculate the day of the week (Monday to Sunday)
			def calculate_day_of_week(row):
			    return row['pickup_date_time'].weekday()  # Monday = 0, Sunday = 6

			# Apply the function to calculate the day of the week
			deliveries_2022['day_of_week'] = deliveries_2022.apply(calculate_day_of_week, axis=1)

			# Group by day of the week (0 = Monday, 6 = Sunday) and aggregate the data
			weekly_aggregated = deliveries_2022.groupby(['day_of_week']).agg({
			    'revenue': 'sum',
			    'pax': 'sum'
			}).reset_index()

			# Create a dictionary to hold the data
			days = {
			    0: 'Monday',
			    1: 'Tuesday',
			    2: 'Wednesday',
			    3: 'Thursday',
			    4: 'Friday',
			    5: 'Saturday',
			    6: 'Sunday'
			}

			# Add day names to the DataFrame
			weekly_aggregated['day_name'] = weekly_aggregated['day_of_week'].map(days)

			# Data
			day_names = weekly_aggregated['day_name']
			revenue = weekly_aggregated['revenue']

			# Create the plot using Matplotlib
			fig, ax = plt.subplots(figsize=(6, 4))
			ax.plot(day_names, revenue, color='green', alpha=1)
			ax.set_xlabel('Days', color='white')  # Set x-axis label color to white
			ax.set_ylabel('Total revenue', color='white')  # Set y-axis label color to white
			ax.set_xticklabels(ax.get_xticks(), color='white')
			ax.set_yticklabels(ax.get_yticks(), color='white')  # Set y-axis tick label color to white
			ax.set_facecolor('none')  # Make the figure background transparent
			fig.patch.set_alpha(0)  # Make the figure background transparent

			# Display the plot in Streamlit
			st.pyplot(fig)

			# Write some insights
			st.write("Wednesday, Thursday and Friday are the busiest days of the week on average. This could be because most people are in the office during those days and is in the later part of the week.")


		with right_column:
			st.subheader("Monthly trend")
			# Convert 'pickup_date_time' to datetime
			df['pickup_date_time'] = pd.to_datetime(df['pickup_date_time'])

			# Filter deliveries for the year 2022
			deliveries_2022 = df[df['pickup_date_time'].dt.year == 2022]
			deliveries_2022['year'] = deliveries_2022['pickup_date_time'].dt.year
			deliveries_2022['month'] = deliveries_2022['pickup_date_time'].dt.month

			# Define a function to calculate the week number within a month (starting from Monday)
			def calculate_week_number(row):
			    days_in_week = 7
			    day_of_month = row['pickup_date_time'].day
			    day_of_week = row['pickup_date_time'].weekday() # Monday = 0, Sunday = 6
			    if day_of_week >= 5: # If the first day of the week is Sunday, week 1 should start from the previous Monday
			        day_of_month -= day_of_week - 4
			    return (day_of_month - 1) // days_in_week + 1

			# Apply the function to calculate the week number
			deliveries_2022['week_number'] = deliveries_2022.apply(calculate_week_number, axis=1)

			# Group by year, month, and week_number, and aggregate the data
			weekly_aggregated = deliveries_2022.groupby(['year', 'month', 'week_number']).agg({
			    'revenue': 'sum',
			    'pax': 'sum'
			}).reset_index()

			# Print the dataframe
			print(weekly_aggregated)

			# Create a dictionary to hold the data
			weeks = {}

			# Iterate over weekly_aggregated to get the specific weeks. 
			for index, row in weekly_aggregated.iterrows():
			    # Access individual columns for each row
			    year = row['year']
			    month = row['month']
			    week_number = row['week_number']
			    total_revenue = row['revenue']
			    total_pax = row['pax']
			    
			    if week_number in weeks:
			        weeks[week_number]['total_revenue'] += total_revenue
			        weeks[week_number]['total_pax'] += total_pax
			    else:
			        weeks[week_number] = {
			            'week_number': week_number,
			            'total_revenue': total_revenue,
			            'total_pax': total_pax
			        }

			# Create a DataFrame from the weeks data
			weeks_df = pd.DataFrame(list(weeks.values()))

			# Drop week_number 0 and 5
			weeks_df = weeks_df[(weeks_df['week_number'] != 0.0) & (weeks_df['week_number'] != 5.0)]
			
			# Create the plot using Matplotlib
			fig, ax = plt.subplots(figsize=(6, 4))
			ax.plot(weeks_df['week_number'], weeks_df['total_revenue'], color='green', alpha=1)
			ax.set_xlabel('Weeks', color='white')  # Set x-axis label color to white
			ax.set_ylabel('Total revenue', color='white')  # Set y-axis label color to white
			ax.set_xticklabels(ax.get_xticks(), color='white')
			ax.set_yticklabels(ax.get_yticks(), color='white')  # Set y-axis tick label color to white
			ax.set_facecolor('none')  # Make the figure background transparent
			fig.patch.set_alpha(0)  # Make the figure background transparent

			# Display the plot in Streamlit
			st.pyplot(fig)

			# Write some insights
			st.write("Across the year of 2022, if we were to split each month into 4 weeks, we would see that the last week is about 20% busier than the other weeks. This could be because companies have hit their KPIs and would celebrate with food. By allocating more manpower to the last week of the month, Grain would be prepared to handle the larger volume.")


			st.subheader("Daily trend")
			# Convert 'pickup_date_time' to datetime
			df['serving_date_time'] = pd.to_datetime(df['serving_date_time'])

			# Filter deliveries for the year 2022
			deliveries_2022 = df[df['serving_date_time'].dt.year == 2022]

			# Extract the pickup time component (e.g., "09:00:00")
			deliveries_2022['serving_date_time'] = deliveries_2022['serving_date_time'].dt.strftime('%H:%M:%S')

			# Group by pickup time and aggregate the data
			time_aggregated = deliveries_2022.groupby(['serving_date_time']).agg({
			    'revenue': 'sum',
			    'pax': 'sum'
			}).reset_index()

			# Create the plot using Matplotlib
			fig, ax = plt.subplots(figsize=(6, 4))
			ax.bar(time_aggregated['serving_date_time'], time_aggregated['pax'], color='green', alpha=1)
			ax.set_xlabel('Time', color='white')  # Set x-axis label color to white
			ax.set_ylabel('Total revenue', color='white')  # Set y-axis label color to white
			ax.set_xticklabels(ax.get_xticks(), color='white', rotation=90)
			ax.set_yticklabels(ax.get_yticks(), color='white')  # Set y-axis tick label color to white
			ax.set_facecolor('none')  # Make the figure background transparent
			fig.patch.set_alpha(0)  # Make the figure background transparent

			# Display the plot in Streamlit
			st.pyplot(fig)

			# Write some insights
			st.write("The most popular time to order mini buffets are during lunch time. More specifically from 11.30pm to 12pm. It would be important to allocate more manpower in the early part of the day to prepare the orders.")


	with who:
		st.subheader("Customer breakdown")
		# Convert 'pickup_date_time' to datetime
		df['pickup_date_time'] = pd.to_datetime(df['pickup_date_time'])

		# Filter deliveries for the year 2022
		deliveries_2022 = df[df['pickup_date_time'].dt.year == 2022]

		# Group by 'company_name' and aggregate the data
		company_aggregated = deliveries_2022.groupby(['company_name', 'sector']).agg({
		    'revenue': 'sum',
		    'pax': 'sum'
		}).reset_index()

		# Remove the company_name and sector_name called missing and filter out negative revenue
		company_aggregated = company_aggregated[
		    (company_aggregated['company_name'] != 'missing') &
		    (company_aggregated['sector'] != 'missing') &
		    (company_aggregated['revenue'] > 0)  # Filter out negative revenue
		]


		# Create the Treemap using Plotly Express
		fig = px.treemap(company_aggregated, path=[px.Constant("Customers"), 'sector', 'company_name'], values='revenue',
		                 color='revenue',
		                 color_continuous_scale='RdBu',
		                 color_continuous_midpoint=np.average(company_aggregated['revenue'], weights=company_aggregated['revenue']))

		# Customize the layout
		fig.update_layout(
		    autosize=True,
		    width=1300,  # Adjust the width as needed
		    height=800,  # Adjust the height as needed
		    margin=dict(t=0, l=25, r=25, b=0),
		)

		# Display the Treemap in Streamlit
		st.plotly_chart(fig)

		# Use a horizontal rule to separate the Treemap and the words in st.write
		st.write('.')
		st.write('.')
		st.write('.')
		st.write('.')
		st.write('.')

		# Write some insights
		st.write("The top 3 sectors are Education, Technology(Software) and Hospitals. The most amount of deliveries in terms of pax goes to the hospitals sector. This could be because the people in hospitals are more health-conscious and would order healthier choices from caterers that offer healthier choices. The top 3 customers are NUS, Caterspot and Bytedance.")


if selected == 'Sales':
	st.header('Which company would you like to get the price for?')
	option = st.selectbox(
    ' ',
    ('Choose an option', 'National University of Singapore', 'Amazon Web Services Singapore Pte Ltd', 'Agency for Integrated Care')) # Choose an option is the default value
	if option != 'Choose an option': 
		st.write('You selected:', option)
		if option == 'National University of Singapore':
			company = 'National University of Singapore'
			# Filter rows where 'special_instructions' contains the word "bento" and 'cost_per_pax' is between 9 and 15
			filtered_df = df[(df['special_instructions'].str.contains('bento', case=False, na=False)) & (df['cost_per_pax'] >= 9) & (df['cost_per_pax'] <= 15)]

			# Calculate sector-wise cost_per_pax average
			sector_avg = filtered_df.groupby('sector')['cost_per_pax'].transform('mean')

			# Add the 'sector_average' column to your DataFrame
			filtered_df['sector_average'] = sector_avg

			# Filter the DataFrame for the specific company
			company_data = filtered_df[filtered_df['company_name'] == company]

			# Select the relevant columns
			selected_columns = ['customer_id', 'pax', 'postal_code', 'revenue', 'sector_average']			
			company_data = company_data[selected_columns]

			# Make predictions on both train and test data
			price = model.predict(company_data)

			# Calculate the average value of the column
			average_value = price.mean()

			st.write('Recommended price for a normal meal box:', average_value)

		if option == 'Amazon Web Services Singapore Pte Ltd':
			company = 'Amazon Web Services Singapore Pte Ltd'
			# Filter rows where 'special_instructions' contains the word "bento" and 'cost_per_pax' is between 9 and 15
			filtered_df = df[(df['special_instructions'].str.contains('bento', case=False, na=False)) & (df['cost_per_pax'] >= 9) & (df['cost_per_pax'] <= 15)]

			# Calculate sector-wise cost_per_pax average
			sector_avg = filtered_df.groupby('sector')['cost_per_pax'].transform('mean')

			# Add the 'sector_average' column to your DataFrame
			filtered_df['sector_average'] = sector_avg

			# Filter the DataFrame for the specific company
			company_data = filtered_df[filtered_df['company_name'] == company]

			# Select the relevant columns
			selected_columns = ['customer_id', 'pax', 'postal_code', 'revenue', 'sector_average']			
			company_data = company_data[selected_columns]

			# Make predictions on both train and test data
			price = model.predict(company_data)

			# Calculate the average value of the column
			average_value = price.mean()

			st.write('Recommended price for a normal meal box:', average_value)

		if option == 'Agency for Integrated Care':
			company = 'Agency for Integrated Care'
			# Filter rows where 'special_instructions' contains the word "bento" and 'cost_per_pax' is between 9 and 15
			filtered_df = df[(df['special_instructions'].str.contains('bento', case=False, na=False)) & (df['cost_per_pax'] >= 9) & (df['cost_per_pax'] <= 15)]

			# Calculate sector-wise cost_per_pax average
			sector_avg = filtered_df.groupby('sector')['cost_per_pax'].transform('mean')

			# Add the 'sector_average' column to your DataFrame
			filtered_df['sector_average'] = sector_avg

			# Filter the DataFrame for the specific company
			company_data = filtered_df[filtered_df['company_name'] == company]

			# Select the relevant columns
			selected_columns = ['customer_id', 'pax', 'postal_code', 'revenue', 'sector_average']			
			company_data = company_data[selected_columns]

			# Make predictions on both train and test data
			price = model.predict(company_data)

			# Calculate the average value of the column
			average_value = price.mean()

			st.write('Recommended price for a normal meal box:', average_value)

if selected == 'Fulfilment':
	# Create columns to separate meat and vegan
	left_column, right_column  = st.columns(2)

	with left_column:
		st.subheader('Demand Forecasting')
		# Convert 'pickup_date_time' to datetime
		df['pickup_date_time'] = pd.to_datetime(df['pickup_date_time'])

		# Set 'pickup_date_time' as the index
		df.set_index('pickup_date_time', inplace=True)

		# Group by weeks and aggregate the 'pax' column
		weekly_pax = df['pax'].resample('W').sum().reset_index()

		# Rename the columns for clarity
		weekly_pax.columns = ['Week Start Date', 'Total Pax']

		# Resets any prevailing indexes of the DataFrame, and use the default one instead
		weekly_pax = weekly_pax.reset_index()

		# Requirement for FBProphet Model - Date column to be named as ds and Y column to be named as y
		weekly_pax=weekly_pax.rename(columns={'Week Start Date':'ds', 'Total Pax':'y'})

		# Get the predictions for the year of 2023
		#future = m.make_future_dataframe(periods=47, freq = 'w')

		# The propher version I am using is an older one as that is the one that works but when I upload to stremalit, it's not compaitable with Python. 
		data_path = Path(__file__).parent / 'data/forecast.csv'
		forecast = pd.read_csv(data_path,lineterminator='\n')

		# Create a selectbox to choose a date
		selected_date = st.selectbox("Select a Date", forecast['ds'])

		# Find the corresponding yhat value
		yhat_value = forecast.loc[forecast['ds'] == selected_date, 'yhat'].values[0]

		# Display the selected date and yhat value
		st.write(f"Selected Date: {selected_date}")
		st.write(f"Predicted number of pax: <font color='green'> {round(yhat_value, 0)}</font>", unsafe_allow_html=True)

		# Create a Streamlit line chart
		st.write("Forecasted Data")
		st.line_chart(forecast.set_index('ds')['yhat'])

	with right_column:
		st.subheader('Driver allocation')

		# Create a file uploader
		uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

		if uploaded_file is not None:
		    # Read the uploaded CSV file into a DataFrame
		    sample_df = pd.read_csv(uploaded_file)

		# Filter out rows where 'event_state' is not equal to 'DELETED'
		sample_df = sample_df[sample_df['event_state'] != 'DELETED']

		# Reorganise the dataframe
		new_df = sample_df[['type', 'event_id', 'pickup_date_time', 'serving_date_time', 'pax', 'address', 'postal_code', 'special_instructions']]

		# Assuming 'df' is your original DataFrame
		new_df1 = new_df.copy()

		# Convert the 'postal_code' column to integer
		new_df1['postal_code'] = new_df1['postal_code'].astype(str)

		# Define a function to add a leading zero to 5-digit postal codes
		def add_leading_zero(p_code):
		    if len(p_code) == 5:
		        return '0' + p_code
		    else:
		        return p_code

		# Apply the function to the 'postal_code' column in the DataFrame
		new_df1['postal_code_new'] = new_df1['postal_code'].apply(add_leading_zero)

		# Convert the 'postal_code' column back to integer
		new_df1['postal_code'] = new_df1['postal_code'].astype(int)

		# Now you can modify 'new_df' without encountering the "SettingWithCopyWarning"
		new_df1.loc[:, 'pickup_date_time'] = pd.to_datetime(new_df1['pickup_date_time'])
		new_df1.loc[:, 'serving_date_time'] = pd.to_datetime(new_df1['serving_date_time'])

		# Sort the DataFrame by 'pickup_date_time', then 'postal_code_new' in ascending order
		new_df_sorted_by_time = new_df1.sort_values(by=['pickup_date_time', 'postal_code_new'])

		# Reset the index
		new_df_sorted_by_time = new_df_sorted_by_time.reset_index(drop=True)

		# Google Maps Distance Matrix API key
		api_key = "AIzaSyAbqeweMLjbIJQSL2yI_eWptsX9nWgl0y4"

		# Define HQ postal code
		HQ = "369972"

		# Postal codes
		postal_codes = new_df_sorted_by_time['postal_code_new'].tolist()

		# Define the API endpoint
		endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

		# Initialize list to store data
		hq_duration_mins = []

		# Iterate through pairs of postal codes
		for i in range(len(postal_codes)):
		    # Define origins and destinations using the postal codes
		    origins = HQ
		    destinations = str(postal_codes[i])  # Note that you should use 'i' as the destination index
		    
		    # Build the API request URL
		    params = {
		        "origins": f"Singapore {origins}",
		        "destinations": f"Singapore {destinations}",
		        "mode": "driving",
		        "key": api_key
		    }

		    # Make the API request
		    response = requests.get(endpoint, params=params)

		    # Check if the response is valid
		    if response.status_code == 200:
		        data = response.json()
		        try:
		            # Extract duration
		            duration_text = data['rows'][0]['elements'][0]['duration']['text']
		            
		            # Extract the numerical part (e.g., "30" from "30 mins") and convert it to an integer
		            duration_mins = int(duration_text.split()[0])
		            
		            # Append the duration in minutes to the list
		            hq_duration_mins.append(duration_mins)
		        except IndexError:
		            hq_duration_mins.append('N/A')
		    else:
		        hq_duration_mins.append('N/A')

		# Add hq_duration to the DataFrame
		new_df_sorted_by_time['hq_duration_mins'] = hq_duration_mins

		# Postal codes
		postal_codes = new_df_sorted_by_time['postal_code_new'].tolist()

		# Define the API endpoint
		endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

		# Initialize lists to store data
		origins_list = []
		destinations_list = []
		distances = []
		durations = []

		# Iterate through pairs of postal codes
		for i in range(len(postal_codes)):
		    for j in range(i + 1, len(postal_codes)):
		        # Define origins and destinations using the postal codes
		        origins = str(postal_codes[i])
		        destinations = str(postal_codes[j])

		        # Parse the pickup times
		        pickup_time_i = new_df_sorted_by_time.iloc[i]['pickup_date_time'].time()
		        pickup_time_j = new_df_sorted_by_time.iloc[j]['pickup_date_time'].time()

		        # Calculate the time difference
		        time_difference = abs((pickup_time_i.hour * 60 + pickup_time_i.minute) - (pickup_time_j.hour * 60 + pickup_time_j.minute))

		        # Check if the time difference is within 30 minutes
		        if time_difference <= 30:
		            # Build the API request URL
		            params = {
		                "origins": f"Singapore {origins}",
		                "destinations": f"Singapore {destinations}",
		                "mode": "driving",
		                "key": api_key
		                    }

		            # Make the API request
		            response = requests.get(endpoint, params=params)
		    
		            # Check if the response is valid
		            if response.status_code == 200:
		                data = response.json()
		                try:
		                    # Extract distance and duration
		                    distance = data['rows'][0]['elements'][0]['distance']['text']
		                    duration = data['rows'][0]['elements'][0]['duration']['text']
		                    
		                    # Extract the numerical part (e.g., "9.0" from "9.0 km") and convert it to a float
		                    distance_km = float(distance.split()[0])
		                    
		                    # Extract the numerical part (e.g., "30" from "30 mins") and convert it to an integer
		                    duration_mins = int(duration.split()[0])
		                    
		                    # Append to the lists
		                    origins_list.append(origins)
		                    destinations_list.append(destinations)
		                    distances.append(distance_km)
		                    durations.append(duration_mins)
		                    
		                except IndexError:
		                    distances.append('N/A')
		                    durations.append('N/A')
		            else:
		                print(f"Error: {response.status_code}")
		                print(f"Response content: {response.content}")

		# Create a DataFrame
		data = {
		    'Origin': origins_list,
		    'Destination': destinations_list,
		    'Distance_km': distances,
		    'Duration_mins': durations
		}

		# Convert it to a dataframe
		df_distances = pd.DataFrame(data)

		# Display the dataframe
		df_distances.head()

		# Initialize the first_delivery_time with the pickup time of the first delivery
		first_delivery_time = new_df_sorted_by_time.iloc[0]['pickup_date_time']

		# There are 9 drivers including llm. Create a dictionary to store driver information

		driver_info = {
		    'KHAI': {'time': first_delivery_time, 'deliveries': 0},
		    'ANG': {'time': first_delivery_time, 'deliveries': 0},
		    'ZHI PENG': {'time': first_delivery_time, 'deliveries': 0},
		    'DAENG': {'time': first_delivery_time, 'deliveries': 0},
		    'RIO': {'time': first_delivery_time, 'deliveries': 0},
		    'SC': {'time': first_delivery_time, 'deliveries': 0},
		    'ROGER': {'time': first_delivery_time, 'deliveries': 0},
		    'RIZWAN': {'time': first_delivery_time, 'deliveries': 0},
		    'LLM': {'time': first_delivery_time, 'deliveries': 0},
		}

		# Initialize driver_allocation as None for the current row
		new_df_sorted_by_time['driver_allocation'] = None

		# Iterate over the entire delivery dataframe
		for index_i, row_i in new_df_sorted_by_time.iterrows():
		    
		    for index_j, row_j in new_df_sorted_by_time.iterrows():
		        # Check if it's the same row
		        if index_i == index_j:
		            continue  # Skip the same row
		        # Calculate the time difference
		        pickup_time_i = row_i['pickup_date_time'].time()
		        pickup_time_j = row_j['pickup_date_time'].time()
		        time_difference = abs((pickup_time_i.hour * 60 + pickup_time_i.minute) - (pickup_time_j.hour * 60 + pickup_time_j.minute))

		        for driver_name, driver_data in driver_info.items():
		            # Check if the time difference is within 30 minutes
		            if time_difference <= 30:
		                # Set the delivery time and convert it to integer
		                delivery = int(new_df_sorted_by_time['hq_duration_mins'].iloc[index_i])
		                delivery_time = datetime.timedelta(minutes=delivery)

		                # Set up the loading bay waiting time
		                loading_bay_wait_time = timedelta(minutes=15)

		                # Add up the time
		                driver_data['time'] = driver_data['time'] + delivery_time + loading_bay_wait_time
		                
		                # Check if the next location is the same. If it is, add 5 mins for delivery
		                if row_i['postal_code_new'] == row_j['postal_code_new']:
		                    # Set up the second delivery time
		                    delivery_time = timedelta(minutes=5)
		                    
		                    # Add up the time
		                    driver_data['time'] = driver_data['time'] + delivery_time
		                    
		                    # Check if the time is before the next delivery serving time
		                    if driver_data['time'] < row_j['serving_date_time']:
		                        new_df_sorted_by_time.at[index_i, 'driver_allocation'] = driver_name  # Assign the driver
		                        new_df_sorted_by_time.at[index_j, 'driver_allocation'] = driver_name  # Assign the driver
		                        driver_data['deliveries'] += 2  # Increment the number of deliveries
		                        print('Driver assignment 1')
		                        index_i += 1
		                        break  # Break the loop if a driver is allocated
		                    break  # Break the loop if a driver is allocated
		        
		                # If it's not the same location, add the time taken to go to the next location
		                else:
		                    # Iterate over the distance dataframe and check when origin and destination match
		                    for index_k, dist_row_k in df_distances.iterrows():
		                        # Check if the origin and destination match certain criteria
		                        if dist_row_k['Origin'] == row_i['postal_code_new'] and dist_row_k['Destination'] == row_j['postal_code_new']:
		                            # Set up the next delivery time 
		                            delivery = int(dist_row_k['Duration_mins'])
		                            delivery_time = datetime.timedelta(minutes=delivery)

		                            # Add up the time
		                            driver_data['time'] = driver_data['time'] + delivery_time + loading_bay_wait_time

		                            # Check if the driver can make both the deliveries
		                            if driver_data['time'] < row_j['serving_date_time']:
		                                new_df_sorted_by_time.at[index_i, 'driver_allocation'] = driver_name  # Assign the driver
		                                new_df_sorted_by_time.at[index_j, 'driver_allocation'] = driver_name  # Assign the driver
		                                driver_data['deliveries'] += 2  # Increment the number of deliveries
		                                print('Driver assignment 2')
		                                index_i += 1
		                                break  # Break the loop if a driver is allocated
		                            break  # Break the loop if a driver is allocated
		        
		            else:
		                # For the deliveries that are not within the 30 mins window
		                if new_df_sorted_by_time.at[index_i, 'driver_allocation'] == None: 
		                # if driver_data['time'] < row_i['pickup_date_time']: 
		                    driver_data['time'] = row_i['pickup_date_time']
		                    new_df_sorted_by_time.at[index_i, 'driver_allocation'] = driver_name
		                    driver_data['deliveries'] += 1  # Increment the number of deliveries
		                
		                #else:
		                    # Go to the next driver 

		# Handle the case where no driver is allocated
		new_df_sorted_by_time['driver_allocation'].fillna("No Driver Available", inplace=True)

		# Display the dataframe
		new_df_sorted_by_time


