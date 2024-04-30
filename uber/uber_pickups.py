import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# make sure the data doesnt get loaded if not needed
@st.cache_data

# loading data from other locations
def load_data(nrows):
    data=pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text('Loading data...done!')

#### adding a button to toggle data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


# Plotting num of pickups by hour
st.subheader('Number of pickups by hour')

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)

# Overlaying a map
#st.subheader('Map of all pickups')
#st.map(data)

# Creating a filtered dataset with hour=17 to see the distribution

#hour_to_filter=17
#using a slider
hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h

filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)
