# CS661 Project : Analysis of Global Youtube Video Data

This project involves the creation of an Analytics Dashboard using the __Trending Youtube Video Statistics__ kaggle dataset. 

## Dataset Description

The chosen dataset is a secondary dataset sourced
from Kaggle. It is about YouTube (the world-
famous video-sharing website) which maintains
a list of the top trending videos on the platform.
This dataset is a daily record of the top trending
YouTube videos. It was collected originally using
the YouTube API. The dataset comprises several
CSV files, each representing a specific country’s
YouTube video statistics. The countries included
cover a diverse range of geographic areas, providing a broad perspective on YouTube content consumption globally. 

Link to Dataset : [Dataset Link](https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023/data)

## Experiments

As of now, we are using the `Global Youtube Statistics Data` filtered on 10 different regions. The plots are created via grouping data based on `Youtube category`, `Country`, `Youtuber`. The dashboard also contains correlation analysis on use actions like `[subscribers, uploads, views, monthly, yearly revenue]`. 

Other plots inclues:

1. Annual channel based trend analysis
2. Category based plots
3. Youtuber metrics analysis
4. Region based summary
5. Correlation plots

## Instructions to use


> The dataset is provided under the `./data` directory

> The main dash application is present in main.py file

> 

## Libraries used

1. plotly express
2. dash
3. dash-bootstrap
