# CS661 Project : Analysis of Global Youtube Video Data

This project involves the creation of an Analytics Dashboard using the __Trending Youtube Video Statistics__ kaggle dataset. 

Dashboard deployed on Render : [https://yt-data-dashboard-1.onrender.com/](https://yt-data-dashboard-1.onrender.com/)

## Dataset Description

The chosen dataset is a global dataset sourced
from Kaggle. It is about YouTube (the world-
famous video-sharing website) which maintains
a list of the top trending videos on the platform.
This dataset is a daily record of the top trending
YouTube videos. It was collected originally using
the YouTube API. The dataset has a large number of features ranging around 50 countries including subscribers, uploads, views, publish date, created data, region wise stats etc, channel type etc. 

Link to Dataset : [Dataset Link](https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023/data)

## Experiments

As of now, we are using the `Global Youtube Statistics Data` filtered on 10 different regions. The plots are created via groupings of data based on `Country`, `Youtuber`, `channel_type`. The dashboard also contains correlation analysis, trend analysis and other plots on user actions like `[subscribers, uploads, views, monthly, yearly revenue]`. 

Some of the main plots of the dashboard inclues:

1. Annual channel based trend analysis
2. Category based plots
3. Youtuber metrics analysis
4. Region based summary
5. Correlation plots

## About the Dashboard


> The global dataset is provided under the `./data` directory

> The main dash application is present in `app.py` file

> All the notebooks are present in the `./notebooks` directory

> The dashboard is deployed using Render and the deployed link is provided above in the README file. 


## Libraries used

1. plotly express
2. dash
3. dash-bootstrap


