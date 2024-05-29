#!/usr/bin/env python
# coding: utf-8

# In[21]:





# In[1]:


import plotly.express as px
import numpy as np
import pandas as pd
import numpy as np
from faker import Faker
from dash import dcc
import dash
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import geopandas as gpd
from shapely.geometry import Point


# In[2]:


#Create 

# Set random seed for reproducibility
np.random.seed(0)

# Initialize Faker to generate fake data
fake = Faker()

# Define the number of rows in the dataset
num_rows = 10000

# Create a DataFrame to store the fake dataset
practice_df = pd.DataFrame()

# Create opening hour variable
opening_hour_range = range(7,9)
opening_hour = np.random.choice(opening_hour_range, size=num_rows)

# Create closing hour variable
closing_hour_range = range(17,22)
closing_hour = np.random.choice(closing_hour_range, size=num_rows)

# Create DaysofWeek variable
daysofweek_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daysofweek = [np.random.choice(daysofweek_categories, size=np.random.randint(1, len(daysofweek_categories) + 1), replace=False) for _ in range(num_rows)]

# Generate random opacity values for each row in the DataFrame
opacity = np.random.uniform(0.1, 0.9, size=num_rows)

# Add opacity values to the DataFrame
practice_df['opacity'] = opacity

# Add DaysofWeek to DataFrame
practice_df['daysofweek'] = daysofweek

# Add the opening hour variable to the DataFrame
practice_df['opening_hour'] = opening_hour

# Add the closing hour variable to the DataFrame
practice_df['closing_hour'] = closing_hour

# Generate aed location variable
min_lat, max_lat = 49.9, 51.1  # Latitude boundaries of Belgium
min_lon, max_lon = 3, 6    # Longitude boundaries of Belgium
practice_df['Latitude'] = np.random.uniform(min_lat, max_lat, size=num_rows)
practice_df['Longitude'] = np.random.uniform(min_lon, max_lon, size=num_rows)
practice_df['aed_coordinates'] = list(zip(practice_df['Latitude'], practice_df['Longitude']))


aed_placement_categories = ['outside', 'inside', 'difficult']
aed_placement = np.random.choice(aed_placement_categories, size=num_rows)
practice_df['aed_placement'] = aed_placement

# Create GeoDataFrame
practice_df['geometry'] = practice_df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
gdf = gpd.GeoDataFrame(practice_df, geometry='geometry')

# Define buffer distances
buffer_distances = {
    'outside': 200 / 81000,  # ~200 meters in degrees
    'inside': 150 / 81000,   # ~150 meters in degrees
    'difficult': 125 / 81000 # ~125 meters in degrees
}

# Apply buffer
gdf['buffer'] = gdf.apply(lambda row: row.geometry.buffer(buffer_distances[row['aed_placement']]), axis=1)





# In[14]:


# Initialize a dictionary to hold all the new columns
new_columns_dict = {}

# Pre-calculate colors based on all possible combinations of days and hours
for day in daysofweek_categories:
    for hour in range(24):
        column_name = f'color_{day}_{hour}'
        condition = (
            (practice_df['daysofweek'].str.contains(day)) & 
            (practice_df['opening_hour'] <= hour) & 
            (practice_df['closing_hour'] >= hour)
        )
        new_columns_dict[column_name] = np.where(condition, 'green', 'red')

# Create a DataFrame from the dictionary
color_df = pd.DataFrame(new_columns_dict)

# Concatenate the new columns to the original DataFrame in a single operation
practice_df = pd.concat([practice_df, color_df], axis=1)



practice_df.head()


# In[19]:


# Initialize the Dash app
app = dash.Dash(__name__, prevent_initial_callbacks=True)

# Create Days of week radio items
radioitems = dcc.RadioItems(
    id='day-radioitems',
    options=[{'label': day, 'value': day} for day in daysofweek_categories],
    value='Monday',  # Default selected value
    inline=True
)

# Set Mapbox access token
px.set_mapbox_access_token('pk.eyJ1Ijoia2V6aWFoZHV0dCIsImEiOiJjbHZnaXo3Y3cwcW16Mmpudnh6anRzZXp5In0.JawqtWmD9uBcGjDAt8C5zg')
center_lat = 50.5  # Latitude of the center of the country
center_lon = 4.3517  # Longitude of the center of the country
zoom_level = 6       # Zoom level (1-20)

# Create the initial map with colors and buffers
initial_day = 'Monday'
initial_hour = 12
color_column = f'color_{initial_day}_{initial_hour}'
practice_df['color'] = practice_df[color_column]
gdf['color'] = practice_df['color']  # Ensure gdf has the same colors


# Create the initial scatter plot layer with the coordinate locations and colors
map_with_aed_points = px.scatter_mapbox(practice_df, lat='Latitude', lon='Longitude', color='color', color_discrete_map={"green": "green", "red": "red"}, zoom=zoom_level, mapbox_style="light")

# Add buffer polygons to the initial map
for _, row in gdf.iterrows():
    buffer_polygon = row['buffer']
    x, y = buffer_polygon.exterior.xy
    color = row['color']
    opacity = row['opacity']
    map_with_aed_points.add_trace(go.Scattermapbox(
        lon=list(x),
        lat=list(y),
        mode="lines",
        fill='toself',
        fillcolor='rgba(0, 128, 0, {})'.format(opacity) if color == 'green' else 'rgba(255, 0, 0, {})'.format(opacity),
        line=dict(color=color, width=0),
        name='Buffer',
        showlegend=False
    ))

# Define the layout of the app
app.layout = html.Div([
    radioitems,
    dcc.Slider(
        id='time-slider',
        min=0,
        max=23,
        step=1,
        value=initial_hour,
        marks={i: str(i) for i in range(24)}
    ),
    dcc.Graph(
        id='map-with-aed-points',
        figure=map_with_aed_points
    )
])

# Define callback to update the map based on the radio items
@app.callback(
    Output('map-with-aed-points', 'figure'),
    [Input('day-radioitems', 'value'),
     Input('time-slider', 'value')]
)
def update_map(selected_day, selected_hour):
    color_column = f'color_{selected_day}_{selected_hour}'
    practice_df['color'] = practice_df[color_column]
    gdf['color'] = practice_df['color']  # Ensure gdf has the same colors
    
    updated_map = px.scatter_mapbox(practice_df, lat='Latitude', lon='Longitude', color='color', color_discrete_map={"green": "green", "red": "red"}, 
                                    zoom=zoom_level, mapbox_style="light")

    # Update marker properties
    updated_map.update_traces(marker=dict(symbol='circle', opacity=0))

    # Add buffer polygons to the map
    for _, row in gdf.iterrows():
        buffer_polygon = row['buffer']
        x, y = buffer_polygon.exterior.xy
        color = row['color']
        opacity = row['opacity']
        updated_map.add_trace(go.Scattermapbox(
            lon=list(x),
            lat=list(y),
            mode="lines",
            fill='toself',
            fillcolor='rgba(0, 128, 0, {})'.format(opacity) if color == 'green' else 'rgba(255, 0, 0, {})'.format(opacity),
            line=dict(color=color, width=0),
            name='Buffer',
            showlegend=False
        ))

    return updated_map

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

