# CS661 Project : Analysis of Trending Youtube Video Data

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
cover a diverse range of geographic areas, provid-
ing a broad perspective on YouTube content con-
sumption globally. 

Link to Dataset : [Dataset Link](https://www.kaggle.com/datasets/datasnaek/youtube-new/data)

## Experiments

As of now, we have merged the region wise csv files and performed some basic analysis on the data which involves identifying average videos per `category ID`, number of unique videos present per region, correlation analysis on use actions like `[likes, dislikes, views, comment_count]` on the merged data. 

Some future plans include region based analysis of the data, sentiment prediction based on user comments and video tags, determining what factors contribute to video popularity and analyzing the correlation between engagement metrics and popularity, identifying key predictors of success. 

## Instructions to use

There is only a jupyter notebook present named `initial.ipynb` for generating those basic plots.

> Download the dataset from kaggle from the link provided above.

> Unzip the `.csv` and `.json` files inside this repo.

> Run the `initial.ipynb` notebook which creates a merged dataframe and produces the desired initial plots. 

## Libraries used

1. numpy
2. pandas
3. matplotlib
4. seaborn