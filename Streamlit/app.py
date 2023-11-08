# Import the libraries
import streamlit as st 										# This helps us to structure the app
from streamlit_option_menu import option_menu 				# This lets us use a navigation bar 
from pathlib import Path 									# This lets us use the file path 
import pandas as pd 										# To use dataframes
import pickle 												# To get our trained model
from prophet import Prophet
import streamlit.components.v1 as components
import matplotlib.pyplot as plt # Used to plot bar charts


# Import dataset
data_path = Path(__file__).parent / 'data/imputed_data.csv'
df = pd.read_csv(data_path,lineterminator='\n')

# Import models
# model_filepath = Path(__file__).parent / 'models/neural_prophet_model.pkl'
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
style = "<div style='background-color:grey; padding:2px'></div>"
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
		'background-color': 'black',  
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
		# Convert 'pickup_date_time' to datetime
		df['pickup_date_time'] = pd.to_datetime(df['pickup_date_time'])

		# Filter deliveries for the year 2022
		deliveries_2022 = df[df['pickup_date_time'].dt.year == 2022]

		# Group the DataFrame by month and year
		deliveries_2022['year'] = deliveries_2022['pickup_date_time'].dt.year
		deliveries_2022['month'] = deliveries_2022['pickup_date_time'].dt.month

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
		y = monthly_aggregated['total_pax']

		# Create the plot using Matplotlib
		fig, ax = plt.subplots(figsize=(6, 4))
		ax.plot(x, y, color='blue', alpha=1)
		ax.set_xlabel('Months', color='white')  # Set x-axis label color to white
		ax.set_ylabel('Total pax', color='white')  # Set y-axis label color to white
		ax.set_xticklabels(ax.get_xticks(), color='white')
		ax.set_yticklabels(ax.get_yticks(), color='white')  # Set y-axis tick label color to white
		ax.set_facecolor('none')  # Make the figure background transparent
		fig.patch.set_alpha(0)  # Make the figure background transparent

		# Display the plot in Streamlit
		st.pyplot(fig)

if selected == 'Sales':
	st.header('Which company would you like to get the price for?')
	option = st.selectbox(
    ' ',
    ('Choose an option', 'National University of Singapore', 'Amazon Web Services Singapore Pte Ltd', 'Agency for Integrated Care')) # Choose an option is the default value
	if option != 'Choose an option': 
		st.write('You selected:', option)
		if option == 'National University of Singapore':
			st.write('Happy')

if selected == 'Fulfilment':
	st.write('Hi')