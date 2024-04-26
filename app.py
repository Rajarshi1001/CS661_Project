import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import numpy as np
import statsmodels.api as sm


PAGE_SIZE = 10

# df_yt = pd.read_csv('data/Global YouTube Statistics.csv', encoding="latin-1")
df_yt = pd.read_csv('data/cleaned_for_global_vis.csv', encoding="latin-1")
df_yt_global = pd.read_csv('data/cleaned_for_global_vis.csv', encoding="latin-1")

df_yt['category'] = df_yt['category'].fillna('Other')
df_yt['channel_type'] = df_yt['channel_type'].fillna('Other')
df_yt[df_yt['created_year'].isna()]

# filling the missing values in the timestamps
df_yt.at[236, 'created_year'] = 2006
df_yt.at[236, 'created_month'] = 'Dec'
df_yt.at[236, 'created_date'] = 20

df_yt.at[468, 'created_year'] = 2008
df_yt.at[468, 'created_month'] = 'Sep'
df_yt.at[468, 'created_date'] = 17

df_yt.at[508, 'created_year'] = 2009
df_yt.at[508, 'created_month'] = 'Aug'
df_yt.at[508, 'created_date'] = 22

df_yt.at[735, 'created_year'] = 2013
df_yt.at[735, 'created_month'] = 'May'
df_yt.at[735, 'created_date'] = 20

df_yt.at[762, 'created_year'] = 2017
df_yt.at[762, 'created_month'] = 'Mar'
df_yt.at[762, 'created_date'] = 8

countries_to_keep = ["Canada", "South Korea", "India","Japan", "United States", "United Kingdom", "Mexico", "Russia", "France", "Germany"]
df_yt = df_yt[df_yt["Country"].isin(countries_to_keep)]
df_yt = df_yt[~df_yt['channel_type'].isin(['Autos', 'Nonprofit'])]
df_yt['mean_monthly_earnings'] = df_yt[['lowest_monthly_earnings', 'highest_monthly_earnings']].mean(axis=1) 
df_yt['mean_yearly_earnings'] = df_yt[['lowest_yearly_earnings', 'highest_yearly_earnings']].mean(axis=1)

df_country = (df_yt.dropna(subset=['Country']))

countries = df_country['Country'].unique()

countries_global = np.append(countries, "Global")


#initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# create color palette
color_discrete_sequence = ['#0a9396','#94d2bd','#e9d8a6','#ee9b00', '#ca6702', '#bb3e03', '#ae2012']


app.layout = html.Div([
    html.H1(children='Global Youtube Stats Dashboard', style={'textAlign':'center', 'padding-bottom': '20px'}),

    ##########
    dbc.Row([
        dbc.Row([
            dbc.Col(
                html.Div(
                    dbc.RadioItems(
                        id="attribute-radios",
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-dark",
                        labelCheckedClassName="active",
                        options=[
                            {'label': 'Suscribers', 'value': 'subscribers'},
                            {'label': 'Views', 'value': 'video views'},
                            {'label': 'Uploads', 'value': 'uploads'},
                            {'label': 'Yearly Earnings (in $)', 'value': 'mean_yearly_earnings'},
                            {'label': 'Monthly Earnings (in $)', 'value': 'mean_monthly_earnings'},
                        ],
                        value='subscribers',
                    ),
                    className ="radio-group",
                    style = {'margin-top': '20px'}
                ),
            ),
            dbc.Col(
                html.Div(
                    dbc.Select(
                        id="country-dropdown_1",
                        options=[{"label": country, "value": country} for country in countries_global],
                        value="Global"  # Default value
                    ),
                    style={'padding':'7.5px'}
                ),
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure={}, id="attribute-bar-graph"),
                width=6
            ),
            dbc.Col(
                dcc.Graph(figure={}, id="attribute-pie-chart"),
                width=6
            )
        ])
    ]), 

    ###########
    dbc.Row(html.Div(style={'height': '50px'})),
    dbc.Row(dbc.Col(html.H2("Category based plots"), width={"size": 6, "offset": 3})),
    dbc.Row([
        dbc.Col([
            dbc.Row(
                html.Div(
                    dbc.Select(
                        id="country-dropdown",
                        options=[{"label": country, "value": country} for country in countries],
                        value="India"  # Default value
                    ),
                    style={'padding':'7.5px'}
                ),
            ),
            dbc.Row(
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[
                        {'label': 'Number of Subscribers', 'value': 'subscribers'},
                        {'label': 'Number of Video Views', 'value': 'video views'},
                        {'label': 'Monthly Revenue (in $)', 'value': 'mean_monthly_earnings'}
                    ],
                    value='subscribers',
                    clearable=False,
                    style={'padding':'5px', 'alignItems': 'center'}
                )
            )
        ]),
        dbc.Col(
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Total Channels"),
                            html.H4(id = "total_channel_country", children="")
                        ]), 
                        color="dark", 
                        outline=True,
                        className = 'text-dark',
                        style={'textAlign': 'center', 'margin-bottom': '20px'}
                    ),
                ),
                dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("Total Revenue (in $)"),
                                html.H4(id = "total_revenue_country", children="")
                            ]), 
                            color="dark", 
                            outline=True,
                            className = 'text-dark',
                            style={'textAlign': 'center', 'margin-bottom': '20px'}
                        ),
                    )
                ])
        )
    ]),
    dbc.Row(
        dbc.Col(
            dcc.Graph(figure={}, id='category-performance-chart'),
            width=12
        )
    ),

    #############
    dbc.Row(html.Div(style={'height': '50px'})),
    dbc.Row(dbc.Col(html.H2("Youtuber Metrics Histograms"), width={'size': 6, 'offset': 3}, align='center')),
    dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='metrics-dropdown',
                options=[
                    {'label': 'Subscribers Count', 'value': 'subscribers'},
                    {'label': 'Monthly Earnings (in $)', 'value': 'mean_monthly_earnings'},
                    {'label': 'Yearly Earnings (in $)', 'value': 'mean_yearly_earnings'},
                    {'label': 'Video Views', 'value': 'video views'},
                    {'label': 'Videos Uploaded', 'value': 'uploads'}
                ],
                value='mean_yearly_earnings'  # default initial value
            ), width={'size': 6, 'offset': 3})
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='metrics-histogram'), width=12)
        ]),
    ]), style={'margin-top': '20px'}, 
),
    #####################
    dbc.Row(html.Div(style={'height': '50px'})),
    dbc.Row(dbc.Col(html.H2("Annual cateogory based trend analysis"),  width={'size': 6, 'offset': 3}, align='center')),
    dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='channel-type-dropdown',
                options=[{'label': channel_type, 'value': channel_type} for channel_type in df_yt['channel_type'].unique()],
                value='Music',  # default value
                clearable=False
            ), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='temporal-chart')),
        ]),
        dbc.Row([
            dbc.Col(html.Button("Previous", id="prev-button", n_clicks=0, className="btn-lg btn-large"), width={"size": 4, "offset": 2}),
            dbc.Col(html.Button("Next", id="next-button", n_clicks=0, className="btn-lg btn-large"), width=4)
        ], justify="around", style={"margin-top": "20px"}),
    dbc.Row(html.Div(id='plot-index', style={'display': 'none'}, children=0)),
    ]),
    style={'margin-top': '20px'}
),

    ##################
    dbc.Row(html.Div(style={'height': '50px'})),
    dbc.Row(dbc.Col(html.H2("Correlation analysis"), width={'size': 12, 'offset': 0})),
    dbc.Card(
    dbc.CardBody([
        dbc.Row(dbc.Col(dcc.Dropdown(
            id='feature-dropdown',
            options=[
                {'label': 'Video Views vs Yearly Earnings (in $)', 'value': 'video views'},
                {'label': 'Uploads vs Yearly Earnings (in $)', 'value': 'uploads'},
                {'label': 'Subscribers vs Yearly Earnings (in $)', 'value': 'subscribers'}
            ],
            value='video views'  # default initial value
        ), width={'size': 6, 'offset': 3})),
        dbc.Row(dbc.Col(dcc.Graph(id='correlation-graph'), width=12)),
    ]),
    style={'margin-top': '20px'}
),
    
    dbc.Row(html.Div(style={'height': '50px'})),
    dbc.Row(dbc.Col(html.H2("Global Youtube Channels Analysis"), width={'size': 12, 'offset': 0})),
    dbc.Card(
    dbc.CardBody([
        dbc.Row(dbc.Col(dcc.Dropdown(
            id='youtube-metric-dropdown',
            options=[
            {'label': 'Subscribers', 'value': 'subscribers'},
            {'label': 'Video Views', 'value': 'video views'},
            {'label': 'New Video Views in last 30 days', 'value': 'video_views_for_the_last_30_days'},
            {'label': 'New Subscribers in last 30 days', 'value': 'subscribers_for_last_30_days'},
            {'label': 'Uploads', 'value': 'uploads'}
        ],
            value='subscribers',  # Default value
            clearable=False
        ), width={'size': 6, 'offset': 3})),
        dbc.Row(dbc.Col(dcc.Graph(id='global-youtube-data-visualization')))
    ]), 
style={'margin-top': '20px'}
)
], style={"margin": "50px 50px 50px 50px", "backgroundColor": "#f8f9fa"})


@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_graph(selected_feature):
    # Create the correlation plot using Plotly Express
    fig = px.scatter(
        df_yt, 
        x=selected_feature, 
        y='mean_yearly_earnings',
        trendline='ols',  # Ordinary Least Squares regression line
        trendline_color_override='red'
    )
    
    # Fit OLS regression model
    X = sm.add_constant(df_yt[selected_feature])
    y = df_yt['mean_yearly_earnings']
    model = sm.OLS(y, X).fit()
    
    # Extract slope and correlation values
    slope = model.params[selected_feature]
    correlation = df_yt[[selected_feature, 'mean_yearly_earnings']].corr().iloc[0, 1]
    
    # Set axis titles dynamically
    x_axis_title = selected_feature.capitalize()
    y_axis_title = "Average Yearly Earnings"
    
    # Add annotation with slope and correlation values to the plot
    fig.update_layout(
        annotations=[
            dict(
                x=0.5,
                y=0.9,
                xref='paper',
                yref='paper',
                text=f'Correlation: {correlation:.2f}',
                showarrow=False,
                font=dict(
                    size=14,
                    color="black"
                )
            )
        ],
        title=f"{selected_feature.capitalize()} vs Yearly Earnings",
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title
    )
    
    return fig

@app.callback(
    Output('temporal-chart', 'figure'),
    [Input('channel-type-dropdown', 'value'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('plot-index', 'children')]
)
def update_chart(selected_channel_type, prev_clicks, next_clicks, current_index):
    # Determine which button was clicked last
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'prev-button' in changed_id:
        current_index = max(current_index - 1, 0)  # Decrease index, minimum is 0
    elif 'next-button' in changed_id:
        current_index = (current_index + 1) % 3  # Increase index, wrap around with mod 3

    filtered_data = df_yt[df_yt['channel_type'] == selected_channel_type]

    # Decide which plot to show based on index
    if current_index % 3 == 0:
        data = filtered_data.groupby('created_year').agg({'video views': 'sum'}).reset_index()
        title = "Video Views Over Time"
        y_column = 'video views'
    elif current_index % 3 == 1:
        data = filtered_data.groupby('created_year').agg({'subscribers': 'sum'}).reset_index()
        title = "Subscribers Over Time"
        y_column = 'subscribers'
    else:
        data = filtered_data.groupby('created_year').agg({'uploads': 'sum'}).reset_index()
        title = "Uploads Over Time"
        y_column = 'uploads'

    fig = px.line(data, x='created_year', y=y_column, title=title)
    fig.update_traces(line=dict(color='darkblue'), fill='tozeroy')
    fig.update_layout(template='plotly_white', xaxis=dict(title='Year', titlefont_size=16), yaxis=dict(title=y_column.capitalize(), titlefont_size=16))

    return fig

@app.callback(
    Output('plot-index', 'children'),
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('plot-index', 'children')]
)
def update_index(prev_clicks, next_clicks, current_index):
    # Determine which button was clicked last
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'prev-button' in changed_id:
        return max(current_index - 1, 0)  # Decrease index, minimum is 0
    elif 'next-button' in changed_id:
        return (current_index + 1) % 3  # Increase index, wrap around with mod 3
    return current_index  # No change in index


@app.callback(
    Output('metrics-histogram', 'figure'),
    Input('metrics-dropdown', 'value')
)
def update_graph(selected_metric):
    # Create the histogram using Plotly Express
    fig = px.histogram(
        df_yt, 
        x=selected_metric, 
        marginal='box', 
        color_discrete_sequence=['darkred']
    )

    # Update layout settings
    fig.update_layout(
        title_text=f'Distribution of youtubers by {selected_metric.replace("_", " ").title()}',
        template='plotly_white',
        xaxis=dict(
            title=selected_metric.replace("_", " ").title(),
            titlefont_size=16,
            range=[0, df_yt[selected_metric].max()]  # Set x-axis range from 0 to max value
        ),
        yaxis=dict(
            title='Count',
            titlefont_size=16
        ),
        barmode='overlay'  # Set barmode to 'overlay' to stack bars
    )


    # Update trace settings
    fig.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=0.8)

    return fig


# Callback to update the chart based on dropdown selections
@app.callback(
    Output('category-performance-chart', 'figure'),
    Output("total_revenue_country", "children"),
    Output("total_channel_country", "children"),
    [Input('country-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_chart(selected_country, selected_metric):
    if selected_country == 'Global':
        filtered_data = df_yt.groupby('category').sum().reset_index()
    else:
        filtered_data = df_yt[df_yt['Country'] == selected_country].groupby('category').sum().reset_index()

    fig = px.treemap(
        filtered_data,
        path=['category'],
        values=selected_metric,
        title=f"{selected_country} - Category Performance by {selected_metric}"
    )
    
    
    hover_template = '<b>%{label}</b><br>%{value}<extra></extra>'
    filtered_data_1 = df_yt[df_yt['Country'] == selected_country]
    total_earning = filtered_data_1['highest_yearly_earnings'].sum()
    total_channels = filtered_data_1['Youtuber'].count()

    fig.update_traces(hovertemplate=hover_template, hoverlabel=dict(font_size=22, font_family="Arial"))

    return fig, str(round(total_earning,0)), str(total_channels)


# @app.callback(
#     Output('global-youtube-data-visualization', 'figure'),
#     Input('youtube-metric-dropdown', 'value')
# )
# def update_map(selected_metric):
#     # Create a choropleth map
#     fig = px.choropleth(
#         df_country,
#         locations='Country',  # assuming ISO_A2 or ISO_A3 country codes, if not, adjust this
#         locationmode='country names',
#         color=selected_metric,  # Dynamically use the metric chosen from the dropdown
#         hover_name='Country',
#         hover_data={'subscribers': True, 'video views': True, 'Population': True, 'Unemployment rate': True},
#         projection='natural earth',
#         title=f"Global Distribution of YouTube {selected_metric.capitalize()}"
#     )
@app.callback(
    Output('global-youtube-data-visualization', 'figure'),
    Input('youtube-metric-dropdown', 'value')
)
def update_map(selected_metric):
    # print(selected_metric)
    if selected_metric=='subscribers_for_last_30_days':
        df=df_yt_global.dropna(subset=['subscribers_for_last_30_days']).copy()
    elif selected_metric=='video_views_for_the_last_30_days':
        df=df_yt_global.dropna(subset=['video_views_for_the_last_30_days']).copy()
    else:
        df=df_yt_global.copy()
    fig = px.scatter_geo(df,
        lat='lat',
        lon='lon',
        color='category',
        size=selected_metric,
        hover_name='Youtuber',
        hover_data=['Title','Country', 'subscribers','highest_yearly_earnings','mean_yearly_earnings','created_year','created_month'],
        projection="natural earth",
        title=f"Global Distribution of Youtube Channels based on {selected_metric.capitalize()}"
    )
    fig.update_layout(
        margin={"r":0, "t":0, "l":0, "b":0},
        height=600,  # Set the height of the plot
        width=1200  # Set the width of the plot
    )
    fig.update_geos(
        visible=True,
        showcountries=True,
        countrycolor="Lightgrey",
        showland=True,
        landcolor='whitesmoke',
        lakecolor='LightBlue',
        showocean=True,
        oceancolor='Azure'
    )
    
    fig.update_layout(
        geo=dict(
            bgcolor= 'rgba(255,255,255,0)',  # Transparent background
            lakecolor='LightBlue',  # Light blue for water bodies
            showocean=True,  # Show ocean
            oceancolor='Azure'  # Very light blue for ocean
        )
    )
    fig.update_layout()
    return fig
    
    # Update layout for better visibility
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth'
        ),
        coloraxis_colorbar=dict(
            title=selected_metric
        ),
        height=800,  # Set the height of the figure
        width=1200   # Set the width of the figure
    )

    return fig

@app.callback(
    Output("attribute-bar-graph", "figure"),
    Output("attribute-pie-chart", "figure"),  
    [Input("attribute-radios", "value"),
     Input("country-dropdown_1", "value")]
)
def update_barpiegraph(selected_attribute, selected_country):
    df_no_nan = (df_yt.dropna(subset=[selected_attribute])).dropna(subset=['Country'])
    if selected_country == "Global":
        filtered_data = df_no_nan
    else:
        filtered_data = df_no_nan[df_no_nan['Country'] == selected_country]
    sorted_df = filtered_data.nlargest(10,selected_attribute)
   # Create the horizontal bar chart
    bar_fig = px.bar(
        sorted_df, 
        y='Youtuber', 
        x=selected_attribute, 
        color=selected_attribute,
        color_continuous_scale='sunset',
        orientation='h',
    )

    # Adding annotations for Youtuber names
    annotations = []
    for i, row in sorted_df.iterrows():
        annotations.append(
            dict(
                x=row[selected_attribute], y=row['Youtuber'], text=row['Country'], showarrow=False,
                textangle=0, xanchor='left', yanchor='middle', font=dict(size=12)
            )
        )

    bar_fig.update_layout(
        annotations=annotations,
        xaxis_title=f'{selected_attribute.capitalize()}',
        yaxis_title='Youtuber'
    )

    # Calculate total per country for the selected feature
    country_totals = df_no_nan.groupby('Country')[selected_attribute].sum().reset_index()
    country_totals = country_totals.sort_values(by=selected_attribute, ascending=False).head(5)
    
    # Create the pie chart
    pie_fig = px.pie(
        country_totals, 
        names='Country', 
        values=selected_attribute, 
        title=f'Distribution of channels based on its {selected_attribute} by Country'
    )

    return bar_fig, pie_fig




if __name__ == '__main__':
    app.run_server(debug=False)
