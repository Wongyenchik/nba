import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm


st.set_page_config(
    page_icon= "☄️",
    page_title= "NBA Analysis",
    layout="wide"
)
data = pd.read_csv('players_stats_by_season_full_details.csv')

st.markdown("<h4 style='color: grey; '>Data used in this project is from 2000 to 2020. Anything above 2020 is prediction</h4>", unsafe_allow_html=True)
st.title("Will number of international players influence the 3PA?")
st.markdown("---")

# --------- Will international players influence the 3PA ---------
# NBA players nationality - Geological
def clean_data(data):
    data["FG%"] = data.FGM / data.FGA
    data["3P%"] = data["3PM"] / data["3PA"]
    data["Age"] = data.Season_Year - data.birth_year
    data['high_school'].fillna('NoHS', inplace =True)
    data["PPG"] = data.PTS / data.GP
    data["RPG"] = data.REB / data.GP
    data["SPG"] = data.STL / data.GP
    data["BPG"] = data.BLK / data.GP
    data["APG"] = data.AST / data.GP
    nba_data = data[data['League'] == 'NBA']
    nba_data['draft_round'].fillna(0, inplace =True)
    nba_data['draft_pick'].fillna(0, inplace =True)
    nba_data['draft_team'].fillna('Undrafted', inplace =True)
    nba_data['FG%'].fillna(0, inplace =True)
    nba_data['3P%'].fillna(0, inplace =True)
    return nba_data

def geological(data):
    cleaned_data = clean_data(data)
    unique_players = cleaned_data['Player'].unique()
    unique_player_data = cleaned_data[cleaned_data['Player'].isin(unique_players)]
    nationality_counts = unique_player_data['nationality'].value_counts()
    nationality_data = pd.DataFrame({'nationality': nationality_counts.index, 'Number of Players': nationality_counts.values})
    fig = px.choropleth(nationality_data, locations='nationality', locationmode='country names', color='Number of Players', title='NBA players nationality',
                    color_continuous_scale=px.colors.sequential.Plasma, range_color=(0, 7000))
    return fig
coll1, coll2 = st.columns([2,1])
coll1.plotly_chart(geological(data), use_container_width=True)
with coll2:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h4 style='line-height: 40px;'>We able to know which country NBA players came from. As we can see, majority of NBA players come from United State.</h4>", unsafe_allow_html=True)

# Define the range of years
min_year = 2000
max_year = 2020

# Create a slider for selecting the year
selected_year = st.slider('Select Year', min_value=min_year, max_value=max_year, value=min_year)
#  --------------------------------------
# Number of international player in NBA

def international(data, selected_year):
    cleaned_data = clean_data(data)
    unique_players = cleaned_data['Player'].unique()
    unique_player_data = cleaned_data[cleaned_data['Player'].isin(unique_players)]
    international_players_data = unique_player_data[unique_player_data['nationality'] != "United States"]
    international_player_counts = international_players_data.groupby('Season_Year').size().reset_index(name='Number of International Players')
    combined_data_filtered = international_player_counts[international_player_counts['Season_Year'] <= selected_year]

    # Plot using Plotly Express
    fig = px.bar(combined_data_filtered, x='Season_Year', y='Number of International Players', text='Number of International Players',
                labels={'Number of International Players': 'Number of International Players (Non-US)', 'Season_Year': 'Season Year'},
                title='Number of International Players in NBA by Year (Non-US)',
                template='seaborn')

    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')  # Removed the dollar sign here

    return fig

# ------------------------------------

def PA(data, selected_year):
    cleaned_data = clean_data(data)
    total_3pa_per_year = cleaned_data.groupby('Season_Year')['3PA'].sum().reset_index()

    # Filter data after combining original and predicted data
    combined_data_filtered = total_3pa_per_year[total_3pa_per_year['Season_Year'] <= selected_year]

    # Plot using Plotly Express
    fig = px.bar(combined_data_filtered, x='Season_Year', y='3PA', text='3PA',
                 labels={'3PA': 'Total 3PA'},
                 title='Total 3PA per year',
                 template='seaborn')

    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')

    return fig


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(international(data, selected_year), use_container_width=True)

with col2:
    st.plotly_chart(PA(data, selected_year), use_container_width=True)
    
st.markdown("###", unsafe_allow_html=True)
st.markdown("###", unsafe_allow_html=True)

#-------------------------------------------
# Mean value 3PA per year NBA US vs NBA International players
def mean_PA(data):
    cleaned_data = clean_data(data)
    # Filter the rows where "nationality" is "US"
    us_players = cleaned_data[cleaned_data['nationality'] == 'United States']

    # Group the filtered data by "Year," "Player," and "game" and calculate the sum of 3PA
    total_3pa_us_players = us_players.groupby(['Season_Year', 'Player', 'Stage'])['3PA'].sum().reset_index()

    # Calculate the mean 3PA per year for US players
    mean_3pa_us_players = total_3pa_us_players.groupby(['Season_Year'])['3PA'].mean().reset_index()

    # Filter the rows where "nationality" is "US"
    inter_players = cleaned_data[cleaned_data['nationality'] != 'United States']

    # Group the filtered data by "Year," "Player," and "game" and calculate the sum of 3PA
    total_3pa_inter_players = inter_players.groupby(['Season_Year', 'Player', 'Stage'])['3PA'].sum().reset_index()

    # Calculate the mean 3PA per year for US players
    mean_3pa_inter_players = total_3pa_inter_players.groupby(['Season_Year'])['3PA'].mean().reset_index()
    
    # Set a custom color palette using Plotly colors
    custom_palette = px.colors.qualitative.Set1

    # Plotting the mean 3PA for US and international players using Plotly Express
    fig = px.line(mean_3pa_us_players, x='Season_Year', y='3PA', color_discrete_sequence=[custom_palette[0]], 
                labels={'3PA': 'Mean 3PA'}, title='Mean 3PA for US Players and International Players')

    fig.add_scatter(x=mean_3pa_inter_players['Season_Year'], y=mean_3pa_inter_players['3PA'], mode='lines+markers',
                    name='International Players', line=dict(dash='dash'), marker=dict(symbol='square', size=10),
                    line_shape='linear', marker_color=custom_palette[1])
    
    fig.add_scatter(x=mean_3pa_us_players['Season_Year'], y=mean_3pa_us_players['3PA'], mode='lines+markers',
                    name='US Players', line=dict(dash='dash'), marker=dict(symbol='square', size=10),
                    line_shape='linear', marker_color=custom_palette[0])

    # Customize the layout
    fig.update_layout(xaxis_title='Season Year', yaxis_title='Mean 3PA', legend=dict(x=0, y=1),
                    xaxis=dict(tickmode='array', tickvals=mean_3pa_us_players['Season_Year'], ticktext=mean_3pa_us_players['Season_Year']))
    return fig

colll1, colll2 = st.columns([1,2])
with colll1:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h4 style='line-height: 40px;'>It proves that the increase of 3PA is not because of the number of international players as US players also shoot more threes nowadays.</h4>", unsafe_allow_html=True)

with colll2:
    st.plotly_chart(mean_PA(data), use_container_width=True)
#-------------------------------------------
# Is height matters in NBA?
# Correlation between height and Career PPG
st.title("Is height matters in NBA?")
st.markdown("---")

def height(data):
    cleaned_data = clean_data(data)
    # Get unique players from the 'Player' column
    unique_players = cleaned_data['Player'].unique()

    # Create an empty list to store average PPG for each player
    cppg_list = []

    # Loop through unique players and calculate average PPG
    for player_name in unique_players:
        player_data = cleaned_data[cleaned_data['Player'] == player_name]
        total_points = player_data['PTS'].sum()
        total_games_played = player_data['GP'].sum()
        average_ppg = total_points / total_games_played
        cppg_list.append(average_ppg)

    # Add the calculated average PPG values to a new column 'CPPG'
    cleaned_data['CPPG'] = cleaned_data['Player'].map(dict(zip(unique_players, cppg_list)))

    # Assuming you have the CPPG and height data in arrays cppg_values and height_values
    cppg_values = cleaned_data['CPPG']
    height_values = cleaned_data['height_cm']

    # Perform linear regression
    coefficients = np.polyfit(cppg_values, height_values, 1)
    polynomial = np.poly1d(coefficients)

    # Create a DataFrame for the regression line
    regression_line = pd.DataFrame({'CPPG': cppg_values, 'Height': polynomial(cppg_values)})

    # Create a scatter plot
    scatter_fig = px.scatter(x=cppg_values, y=height_values, labels={'x': 'Points Per Game (CPPG)', 'y': 'Height (cm)'},
                            title='Relationship between Career points per game and height(cm)', opacity=0.7,
                            template='seaborn')

    # Create a line plot for the regression line with specified color
    line_fig = px.line(x=regression_line['CPPG'], y=regression_line['Height'],
                    labels={'x': 'Points Per Game (CPPG)', 'y': 'Height (cm)'}, line_shape='linear',
                    title='Regression Line', line_dash_sequence=["solid"], template='seaborn')
    line_fig.data[0].marker.color = 'red'  # Set line color to red

    # Add the regression line to the scatter plot
    scatter_fig.add_trace(go.Scatter(x=line_fig.data[0].x, y=line_fig.data[0].y, mode='lines',
                                    line=dict(color='red', dash='solid'), name='Regression Line'))
    return scatter_fig

col1, col2 = st.columns([2,1])
with col1:
    # st.caption("In this graph, we saw that the number of international player is increasing every year.")
    st.plotly_chart(height(data), use_container_width=True)

with col2:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h4 style='line-height: 40px;'>Height may not significantly affect NBA performance, but it seems crucial for entry. Individuals around 180cm have a higher chance of making it to the league.</h4>", unsafe_allow_html=True)
    # st.markdown("<h6 style='color: grey; '>In this graph, we saw that the number of 3PA is increasing.</h6>", unsafe_allow_html=True)

st.markdown("###")


# ------------------------------------------
# Is it easier to get foul nowadays in NBA
st.title("Is it easier to get foul nowadays in NBA?")
st.markdown("---")

def FTA_year(data):
    cleaned_data = clean_data(data)
    # Group data by Season_Year and calculate the sum of FTA for each year
    yearly_totals = cleaned_data.groupby('Season_Year')['FTA'].sum().reset_index()

    fig = px.scatter(yearly_totals, x='Season_Year', y='FTA', color_discrete_sequence=['blue'],
                 labels={'FTA': 'Total FTA'}, title='Total Free Throw Attempts (FTA) by Year',
                 template='seaborn')

    # Customize marker size and style
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')),
                    selector=dict(mode='markers+lines'))

    # Customize x-axis ticks to display integer values
    fig.update_xaxes(tickvals=yearly_totals['Season_Year'], tickmode='array')

    return fig


# -----------------------------
# FGA each year
def FGA_year(data):
    cleaned_data = clean_data(data)
    # Group data by Season_Year and calculate the sum of FTA for each year
    yearly_totals = cleaned_data.groupby('Season_Year')['FGA'].sum().reset_index()

    fig = px.scatter(yearly_totals, x='Season_Year', y='FGA', color_discrete_sequence=['blue'],
                 labels={'FGA': 'Total FGA'}, title='Total Field Goal Attempts (FGA) by Year',
                 template='seaborn')

    # Customize marker size and style
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')),
                    selector=dict(mode='markers+lines'))

    # Customize x-axis ticks to display integer values
    fig.update_xaxes(tickvals=yearly_totals['Season_Year'], tickmode='array')

    return fig

st.markdown("<h4 style='line-height: 40px;'>We can see that the number of FTA each year does not fluctuate a lot but the number of FGA each year increased a lot in the recent 4 years. Hence, does it proves that it's harder to get free throw nowadays?", unsafe_allow_html=True)
col3, col4 = st.columns(2)
col3.plotly_chart(FTA_year(data), use_container_width=True)
col4.plotly_chart(FGA_year(data), use_container_width=True)

#---------------------------
#3PA vs FTA
def three_FTA(data):
    cleaned_data = clean_data(data)
    # Group data by year and calculate total 3PA and FTA
    grouped_data = cleaned_data.groupby('Season_Year').agg({'3PA': 'sum', 'FTA': 'sum'}).reset_index()

    # Create a Plotly bar chart
    fig = px.bar(grouped_data, x='Season_Year', y=['3PA', 'FTA'], labels={'value': 'Total Attempts', 'variable': 'Type'}, 
                title='Total 3PA and FTA by Year', 
                category_orders={'variable': ['3PA', 'FTA']})

    # Customize the appearance and layout (optional)
    fig.update_xaxes(tickangle=45)
    fig.update_layout(barmode='group')
    return fig

col1, col2 = st.columns([1,2])
with col1:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h6 style='line-height: 30px;'>In the past, players aimed for the paint, resulting in more free throws due to physical play. Now, with more three-point attempts, physical contact is reduced. Despite increased FGAs, consistent free throw attempts indicate it's not necessarily harder to earn them, as many FGAs are low-contact three-point shots.</h6>", unsafe_allow_html=True)
    
col2.plotly_chart(three_FTA(data), use_container_width=True)

#-------------------------------
# Predict who has the highest PPG
st.title("Prediction on top 3 points per game player in NBA")
st.markdown("---")

#Filter button
selected_year = st.selectbox('Select Year', range(2000, 2022))

def top3(data):
    cleaned_data = clean_data(data)
    #Filter the data to keep only regular season records
    regular_season_data = cleaned_data[cleaned_data['Stage'] == 'Regular_Season']

    #Create an empty DataFrame to store the top players from each year
    top_players_df = pd.DataFrame(columns=['Year', 'Player', 'PPG'])

    #Group the data by 'Season_Year'
    grouped = regular_season_data.groupby('Season_Year')

    #Iterate through each group (each year)
    for year, group in grouped:
        # Sort the group by 'PPG' in descending order
        sorted_group = group.sort_values(by='PPG', ascending=False)

    #Select the top 3 players for this year
        top_3_players = sorted_group.head(3)

    #Add the 'Year' column to match the current year
        top_3_players['Year'] = year

    #Append the top 3 players for this year to the 'top_players_df'
        top_players_df = top_players_df.append(top_3_players, ignore_index=True)
    return top_players_df


# Filter the table based on the selected year
def futuretop3():
    # Load data from the top3.csv file into a DataFrame
    arima = pd.read_csv('top3.csv')

    # Select specific columns ('Year', 'Player', 'PPG') from the DataFrame
    predictions_df = arima[['Year', 'Player', 'PPG']]
    return predictions_df

future_top3_data = futuretop3()
top3_data = top3(data)
combined_data = pd.concat([pd.DataFrame(top3_data), pd.DataFrame(future_top3_data)])

def combined(combined_data, selected_year):
    selected_columns = ['Year', 'Player', 'PPG']
    filtered_top_players_df = combined_data[selected_columns]
    filtered_top_players_df['Year'] = filtered_top_players_df['Year'].astype(int)
    filtered_table = filtered_top_players_df[filtered_top_players_df['Year'] == selected_year]
    return filtered_table

# Display the combined table using Streamlit
# Assuming combined() function generates the HTML table for the given combined_data and selected_year
html_table = combined(combined_data, selected_year).style.hide(axis="index").to_html()

# Apply custom CSS styles to make the table take full width
custom_css = """
<style>
table {
    width: 100%;
}
</style>
"""

# Combine the custom CSS and the HTML table
styled_html_table = custom_css + html_table

# Display the styled HTML table in Streamlit using st.markdown()
st.markdown(styled_html_table, unsafe_allow_html=True)
st.markdown("###")
st.markdown("###")

#----------------------------
# PPG by each players each year
def ppg(data):
    cleaned_data = clean_data(data)
    cleaned_data1 = cleaned_data[(cleaned_data['Player'] == 'James Harden') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data2 = cleaned_data[(cleaned_data['Player'] == 'Bradley Beal') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data3 = cleaned_data[(cleaned_data['Player'] == 'Giannis Antetokounmpo') & (cleaned_data['Stage'] == 'Regular_Season')]
    regular_season_data = cleaned_data[cleaned_data['Stage'] == 'Regular_Season']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=regular_season_data['Season_Year'], y=regular_season_data['PPG'],
        name='All Players',
        mode='markers',
        marker_color='rgb(173, 216, 230)'
    ))

    fig.add_trace(go.Scatter(
        x=top3_data['Season_Year'], y=top3_data['PPG'],
        mode='markers',
        marker_color='rgb(255, 2, 0)',
        name='Top 3 Players'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data1['Season_Year'], y=cleaned_data1['PPG'],
        name='James Harden',
        mode='markers',
        marker_color='rgb(255, 150, 0)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data2['Season_Year'], y=cleaned_data2['PPG'],
        name='Bradley Beal',
        mode='markers',
        marker_color='rgb(128, 0, 128)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data3['Season_Year'], y=cleaned_data3['PPG'],
        name='Giannis Antetokounmpo',
        mode='markers',
        marker_color='rgb(0, 128, 0)'
    ))

    fig.update_layout(title='PPG by each players each year',
                    yaxis_zeroline=False, xaxis_zeroline=False)

    return fig

#----------------------------
# 3PTS Made by each players each year
def plot(data):
    cleaned_data = clean_data(data)
    cleaned_data1 = cleaned_data[(cleaned_data['Player'] == 'James Harden') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data2 = cleaned_data[(cleaned_data['Player'] == 'Bradley Beal') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data3 = cleaned_data[(cleaned_data['Player'] == 'Giannis Antetokounmpo') & (cleaned_data['Stage'] == 'Regular_Season')]
    regular_season_data = cleaned_data[cleaned_data['Stage'] == 'Regular_Season']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=regular_season_data['Season_Year'], y=regular_season_data['3PA'],
        name='All Players',
        mode='markers',
        marker_color='rgb(173, 216, 230)'
    ))

    fig.add_trace(go.Scatter(
        x=top3_data['Season_Year'], y=top3_data['3PA'],
        mode='markers',
        marker_color='rgb(255, 2, 0)',
        name='Top 3 Players'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data1['Season_Year'], y=cleaned_data1['3PA'],
        name='James Harden',
        mode='markers',
        marker_color='rgb(255, 150, 0)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data2['Season_Year'], y=cleaned_data2['3PA'],
        name='Bradley Beal',
        mode='markers',
        marker_color='rgb(128, 0, 128)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data3['Season_Year'], y=cleaned_data3['3PA'],
        name='Giannis Antetokounmpo',
        mode='markers',
        marker_color='rgb(0, 128, 0)'
    ))

    fig.update_layout(title='3PA by each players each year',
                    yaxis_zeroline=False, xaxis_zeroline=False)

    return fig

#----------------------------
# FGA by each players each year
def fga(data):
    cleaned_data = clean_data(data)
    cleaned_data1 = cleaned_data[(cleaned_data['Player'] == 'James Harden') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data2 = cleaned_data[(cleaned_data['Player'] == 'Bradley Beal') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data3 = cleaned_data[(cleaned_data['Player'] == 'Giannis Antetokounmpo') & (cleaned_data['Stage'] == 'Regular_Season')]
    regular_season_data = cleaned_data[cleaned_data['Stage'] == 'Regular_Season']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=regular_season_data['Season_Year'], y=regular_season_data['FGA'],
        name='All Players',
        mode='markers',
        marker_color='rgb(173, 216, 230)'
    ))

    fig.add_trace(go.Scatter(
        x=top3_data['Season_Year'], y=top3_data['FGA'],
        mode='markers',
        marker_color='rgb(255, 2, 0)',
        name='Top 3 Players'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data1['Season_Year'], y=cleaned_data1['FGA'],
        name='James Harden',
        mode='markers',
        marker_color='rgb(255, 150, 0)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data2['Season_Year'], y=cleaned_data2['FGA'],
        name='Bradley Beal',
        mode='markers',
        marker_color='rgb(128, 0, 128)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data3['Season_Year'], y=cleaned_data3['FGA'],
        name='Giannis Antetokounmpo',
        mode='markers',
        marker_color='rgb(0, 128, 0)'
    ))

    fig.update_layout(title='FGA by each players each year',
                    yaxis_zeroline=False, xaxis_zeroline=False)

    return fig

#----------------------------
# ORB by each players each year
def orb(data):
    cleaned_data = clean_data(data)
    cleaned_data1 = cleaned_data[(cleaned_data['Player'] == 'James Harden') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data2 = cleaned_data[(cleaned_data['Player'] == 'Bradley Beal') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data3 = cleaned_data[(cleaned_data['Player'] == 'Giannis Antetokounmpo') & (cleaned_data['Stage'] == 'Regular_Season')]
    regular_season_data = cleaned_data[cleaned_data['Stage'] == 'Regular_Season']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=regular_season_data['Season_Year'], y=regular_season_data['ORB'],
        name='All Players',
        mode='markers',
        marker_color='rgb(173, 216, 230)'
    ))

    fig.add_trace(go.Scatter(
        x=top3_data['Season_Year'], y=top3_data['ORB'],
        mode='markers',
        marker_color='rgb(255, 2, 0)',
        name='Top 3 Players'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data1['Season_Year'], y=cleaned_data1['ORB'],
        name='James Harden',
        mode='markers',
        marker_color='rgb(255, 150, 0)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data2['Season_Year'], y=cleaned_data2['ORB'],
        name='Bradley Beal',
        mode='markers',
        marker_color='rgb(128, 0, 128)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data3['Season_Year'], y=cleaned_data3['ORB'],
        name='Giannis Antetokounmpo',
        mode='markers',
        marker_color='rgb(0, 128, 0)'
    ))

    fig.update_layout(title='ORB by each players each year',
                    yaxis_zeroline=False, xaxis_zeroline=False)

    return fig

#----------------------------
# FTA by each players each year
def fta(data):
    cleaned_data = clean_data(data)
    cleaned_data1 = cleaned_data[(cleaned_data['Player'] == 'James Harden') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data2 = cleaned_data[(cleaned_data['Player'] == 'Bradley Beal') & (cleaned_data['Stage'] == 'Regular_Season')]
    cleaned_data3 = cleaned_data[(cleaned_data['Player'] == 'Giannis Antetokounmpo') & (cleaned_data['Stage'] == 'Regular_Season')]
    regular_season_data = cleaned_data[cleaned_data['Stage'] == 'Regular_Season']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=regular_season_data['Season_Year'], y=regular_season_data['FTA'],
        name='All Players',
        mode='markers',
        marker_color='rgb(173, 216, 230)'
    ))

    fig.add_trace(go.Scatter(
        x=top3_data['Season_Year'], y=top3_data['FTA'],
        mode='markers',
        marker_color='rgb(255, 2, 0)',
        name='Top 3 Players'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data1['Season_Year'], y=cleaned_data1['FTA'],
        name='James Harden',
        mode='markers',
        marker_color='rgb(255, 150, 0)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data2['Season_Year'], y=cleaned_data2['FTA'],
        name='Bradley Beal',
        mode='markers',
        marker_color='rgb(128, 0, 128)'
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_data3['Season_Year'], y=cleaned_data3['FTA'],
        name='Giannis Antetokounmpo',
        mode='markers',
        marker_color='rgb(0, 128, 0)'
    ))

    fig.update_layout(title='FTA by each players each year',
                    yaxis_zeroline=False, xaxis_zeroline=False)

    return fig

st.markdown("<h4 style='line-height: 40px;'>Why James Harden, Bradley Beal, and Giannis Antetokounmpo are predicted to be the top 3 points per game player in NBA in 2021?", unsafe_allow_html=True)

col5, col6 = st.columns([2,1])
col5.plotly_chart(ppg(data),use_container_width=True)
with col6:
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")
    st.markdown("<h4 style='line-height: 40px;'>The PPG trend for James Harden, Bradley Beal, and Giannis Antetokounmpo is consistently upward, surpassing the majority of players over the last 4 years.", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fga(data),use_container_width=True)
    st.markdown("<h6 style='line-height: 30px;'>The graph clearly illustrates that James Harden, Bradley Beal, and Giannis Antetokounmpo have consistently high Field Goal Attempts (FGA), especially in the last 4 years. This suggests that a higher number of attempts leads to more scoring opportunities for these players.", unsafe_allow_html=True)
with col2:
    st.plotly_chart(plot(data), use_container_width=True)
    st.markdown("<h6 style='line-height: 30px;'>In the last 4 years, James Harden and Bradley Beal consistently rank in the league's top tier for 3-point attempts (3PA), indicating that a significant portion of their points come from 3 point line. However, Giannis Antetokounmpo's lower 3PA compared to James Harden and Bradley Beal suggests that a substantial portion of his points are derived from 2-point shots.", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(orb(data),use_container_width=True)
    st.markdown("<h6 style='line-height: 30px;'>Certainly, James Harden and Bradley Beal consistently rank at the lower end of the league in terms of offensive rebounds. On the contrary, Giannis Antetokounmpo's ORB is much higher than another two players, providing him with more opportunities to score in the paint. However, this observation implies that offensive rebounding might have a limited impact on increasing their points per game, as none of these players are among the league's top offensive rebounders.", unsafe_allow_html=True)
with col4:
    st.plotly_chart(fta(data),use_container_width=True)
    st.markdown("<h6 style='line-height: 30px;'>Observing their Free Throw Attempts (FTA) data, it's evident that James Harden consistently has significantly higher FTA compared to Bradley Beal and Giannis Antetokounmpo. This discrepancy could be a contributing factor to Harden having the highest Points Per Game (PPG) among the three. Interestingly, Bradley Beal has the lowest FTA among the trio. This suggests that his points are primarily from field goal made rather than free throws.", unsafe_allow_html=True)

st.markdown("##")
st.markdown("##")

#-------------------------------
st.title("Does draft pick matters?")
st.markdown("---")

# Point score during playoff and regular season for each round pick
def draftround(data):
    cleaned_data = clean_data(data)
    # Filter data for draft round 1, 2, 3 and separate by playoff and regular season
    draft_rounds = [1, 2]
    stages = ['Playoffs', 'Regular_Season']

    # Initialize a dictionary to store total scores for each draft round and stage
    total_scores = {1: {'Playoffs': 0, 'Regular_Season': 0},
                2: {'Playoffs': 0, 'Regular_Season': 0}}
                # 3: {'Playoffs': 0, 'Regular_Season': 0}}

    # Calculate total scores for each draft round and stage
    for round_num in draft_rounds:
        for stage in stages:
            filtered_data = cleaned_data[(cleaned_data['draft_round'] == round_num) & (cleaned_data['Stage'] == stage)]
            total_score = filtered_data['PTS'].sum()
            total_scores[round_num][stage] = total_score
    # Prepare data for Plotly bar chart
    draft_round_labels = [f'Round {round_num}' for round_num in draft_rounds]
    playoff_scores = [total_scores[round_num]['Playoffs'] for round_num in draft_rounds]
    regular_scores = [total_scores[round_num]['Regular_Season'] for round_num in draft_rounds]

    fig = go.Figure()

    # Add bar trace for Playoffs
    fig.add_trace(go.Bar(
        x=draft_round_labels,
        y=playoff_scores,
        name='Playoffs',
        marker_color='rgba(255, 0, 0, 0.7)'
    ))

    # Add bar trace for Regular Season
    fig.add_trace(go.Bar(
        x=draft_round_labels,
        y=regular_scores,
        name='Regular Season',
        marker_color='rgba(0, 0, 255, 0.7)'
    ))

    # Update layout
    fig.update_layout(
        title='Total Scores by Draft Round and Stage',
        xaxis=dict(tickvals=[p for p in range(len(draft_round_labels))], ticktext=draft_round_labels),
        xaxis_title='Draft Round',
        yaxis_title='Total Score',
        barmode='group'
    )
    return fig

col5, col6 = st.columns([1,2])
with col5:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h4 style='line-height: 40px;'>We can prove that round pick does matter as 1st round players will score more than 2nd round players.</h4>", unsafe_allow_html=True)
col6.plotly_chart(draftround(data), use_container_width=True)
#--------------------------------------
#Total Points for each pick in the 1st Round 
def first_round(data):
    cleaned_data = clean_data(data)
    # Filter data for 1st round picks
    first_round_picks = cleaned_data[cleaned_data['draft_round'] == 1]

    # Calculate total point for each pick in the first round
    ppg_by_player = first_round_picks.groupby('draft_pick')['PPG'].sum().reset_index()
    # Create a Plotly bar chart
    fig = px.bar(ppg_by_player, x='draft_pick', y='PPG', labels={'PPG': 'Total Points'}, color='PPG')

    # Customize the appearance and layout
    fig.update_layout(
        title='Total Points for each pick in the 1st Round',
        xaxis_title='Draft Pick',
        yaxis_title='Total Points',
        xaxis=dict(tickvals=ppg_by_player['draft_pick'], ticktext=ppg_by_player['draft_pick']),
        xaxis_tickangle=-45,
        bargap=0.1  # Adjust the gap between bars
    )
    return fig

# st.plotly_chart(first_round(data))

#--------------------------------------
#Total Points for each pick in the 2nd Round
def second_round(data):
    cleaned_data = clean_data(data)
    # Filter data for 1st round picks
    first_round_picks = cleaned_data[cleaned_data['draft_round'] == 2]

    # Calculate total point for each pick in the first round
    ppg_by_player = first_round_picks.groupby('draft_pick')['PPG'].sum().reset_index()
    # Create a Plotly bar chart
    fig = px.bar(ppg_by_player, x='draft_pick', y='PPG', labels={'PPG': 'Total Points'}, color='PPG',color_continuous_scale = 'viridis')

    # Customize the appearance and layout
    fig.update_layout(
        title='Total Points for each pick in the 2nd Round',
        xaxis_title='Draft Pick',
        yaxis_title='Total Points',
        xaxis=dict(tickvals=ppg_by_player['draft_pick'], ticktext=ppg_by_player['draft_pick']),
        xaxis_tickangle=-45,
        bargap=0.1, # Adjust the gap between bars
    )
    return fig

st.markdown("###")
st.markdown("<h4 style='line-height: 40px;'>However, higher draft picks don't guarantee higher points per game within the round.</h4>", unsafe_allow_html=True)
st.markdown("###")

col5, col6 = st.columns(2)
col5.plotly_chart(first_round(data), use_container_width=True)
col6.plotly_chart(second_round(data), use_container_width=True)
st.markdown("###")

#--------------------------------------
# Can NBA player improves their points per game by making more FGA?
st.title("Can NBA player improves their points per game by making more FGA?")
st.markdown("---")

def ppg_fga(data):
    cleaned_data = clean_data(data)
    # Get unique players from the 'Player' column
    unique_players = cleaned_data['Player'].unique()

    # Create an empty list to store average FGA for each player
    cfga_list = []

    # Loop through unique players and calculate average FGA
    for player_name in unique_players:
        player_data = cleaned_data[cleaned_data['Player'] == player_name]
        total_fga = player_data['FGA'].sum()
        total_games_played = player_data['GP'].sum()
        average_fga = total_fga / total_games_played
        cfga_list.append(average_fga)

    # Add the calculated average TOV values to a new column 'CTOV'
    cleaned_data['CFGA'] = cleaned_data['Player'].map(dict(zip(unique_players, cfga_list)))

    # Create an empty list to store average PPG for each player
    cppg_list = []

    # Loop through unique players and calculate average PPG
    for player_name in unique_players:
        player_data = cleaned_data[cleaned_data['Player'] == player_name]
        total_points = player_data['PTS'].sum()
        total_games_played = player_data['GP'].sum()
        average_ppg = total_points / total_games_played
        cppg_list.append(average_ppg)

    # Add the calculated average PPG values to a new column 'CPPG'
    cleaned_data['CPPG'] = cleaned_data['Player'].map(dict(zip(unique_players, cppg_list)))

    # Assuming you have the CPPG and height data in arrays cppg_values and height_values
    cppg_values = cleaned_data['CPPG']
    fga_values = cleaned_data['CFGA']

    # Perform linear regression
    coefficients = np.polyfit(cppg_values, fga_values, 1)
    polynomial = np.poly1d(coefficients)

    # Create a DataFrame for the regression line
    regression_line = pd.DataFrame({'CPPG': cppg_values, 'FGA': polynomial(cppg_values)})

    # Create a scatter plot
    scatter_fig = px.scatter(x=cppg_values, y=fga_values, labels={'x': 'Points Per Game (CPPG)', 'y': 'FGA'},
                            title='Relation between Points per game and FGA', opacity=0.7,
                            template='seaborn')

    # Create a line plot for the regression line with specified color
    line_fig = px.line(x=regression_line['CPPG'], y=regression_line['FGA'],
                    labels={'x': 'Points Per Game (CPPG)', 'y': 'FGA'}, line_shape='linear',
                    title='Regression Line', line_dash_sequence=["solid"], template='seaborn')
    line_fig.data[0].marker.color = 'red'  # Set line color to red

    # Add the regression line to the scatter plot
    scatter_fig.add_trace(go.Scatter(x=line_fig.data[0].x, y=line_fig.data[0].y, mode='lines',
                                    line=dict(color='red', dash='solid'), name='Regression Line'))
    return scatter_fig

# st.plotly_chart(tov(data))
#--------------------------------
def fga_tov(data): 
    cleaned_data = clean_data(data)
    # Assuming you have a DataFrame nba_data
    # Assuming you have a DataFrame nba_data
    fig = px.scatter(cleaned_data, x='FGA', y='TOV', labels={'TOV': 'Total TOV'}, title="NBA Player's FGA vs TOV")
    fig.update_traces(marker=dict(size=10, opacity=0.7))

    # Customize the appearance and layout
    fig.update_layout(
        xaxis_title='Total FGA',
        yaxis_title='Total TOV',
    )
    return fig

#--------------------------- 
# FGA vs. FTA for Individual Players
def FGA_FTA(data):
    cleaned_data = clean_data(data)
    # Group data by Season_Year and calculate the sum of FTA for each year    
    fig = px.scatter(cleaned_data, x='FGA', y='FTA', color='Player',
                 labels={'FGA': 'Field Goals Attempted (FGA)', 'FTA': 'Free Throw Attempts (FTA)'},
                 title='FGA vs. FTA for Individual Players', template='seaborn')
    return fig

col8, col9 = st.columns([1,2])
with col8:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h4 style='line-height: 40px;'>It shows that more FGA will lead to more points per game.</h4>", unsafe_allow_html=True)

col9.plotly_chart(ppg_fga(data), use_container_width=True)
st.markdown("###")
st.markdown("###")

col10, col11 = st.columns([2,1])
with col10:
    st.plotly_chart(FGA_FTA(data), use_container_width=True)

with col11:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h4 style='line-height: 40px;'>One of the reason can be the increased field goal attempts as it results in more free throw attempts (FTA), contributing to higher points, as indicated in the graph.</h4>", unsafe_allow_html=True)

st.markdown("###")
st.markdown("###")

col12, col13 = st.columns([1,2])
with col12:
    st.markdown("###")
    st.markdown("###")
    st.markdown("###")
    st.markdown("<h4 style='line-height: 40px;'>However, more field goal attempts (FGA) can lead to more turnovers (TOV), causing the player's team to lose points. While this might boost the player's points per game (PPG), it can negatively impact the team's overall performance.</h4>", unsafe_allow_html=True)

col13.plotly_chart(fga_tov(data), use_container_width=True) 


# # ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
