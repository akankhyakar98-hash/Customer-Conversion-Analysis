# Customer-Conversion-Analysis
This project analyzes maternity clothing e-commerce clickstream data using machine learning. It employs classification to predict purchases, regression to forecast revenue, and clustering for customer segmentation. The goal is to enhance marketing and sales through data-driven insights. 
# ABSTRACT
This project examines user behavior on a maternity clothing e-commerce platform using web mining techniques on clickstream data. The analysis is structured around three core machine learning tasks: classification to predict purchasing intent, regression to forecast revenue, and clustering to identify customer segments. The insights derived from these models are designed to enhance marketing effectiveness, improve consumer satisfaction, and drive significant sales growth.
## INTRODUCTION 
E-commerce has transformed retail, necessitating a deep understanding of user behavior. This study explores user interactions within an online maternity clothing store, leveraging clickstream data analysis. It employs a multi-faceted machine learning approach including classification to predict purchasing intent, regression to forecast revenue, and clustering to identify customer segments. The goal is to provide actionable insights for website optimization and personalized marketing. 
### Variable Table
| Variable       | Role       | Type       | Description                              | Units         | Missing Values |
|----------------|------------|------------|------------------------------------------|---------------|----------------|
| year           | Feature    | Date       | 2008                                     |               | No             |
| month          | Feature    | Date       | April (4) to August (8)                  |               | No             |
| day            | Feature    | Date       | Day number of the month                  |               | No             |
| order          | Feature    | Integer    | Sequence of clicks during one session    |               | No             |
| country        | Feature    | Categorical| Country of origin of the IP address      |               | No             |
| session ID     | Feature    | Integer    | Session ID (short record)                |               | No             |
| page 1         | Feature    | Categorical| Main product category                    |               | No             |
| page 2         | Feature    | Categorical| Product code for each item               |               | No             |
| color          | Feature    | Categorical| Color of product                         |               | No             |
| location       | Feature    | Categorical| Photo location on the page               |               | No             |
| model photography | Feature  | Categorical| Style of product photography             |               | No             |
| price          | Feature    | Integer    | Price in USD                             | USD           | No             |
| price 2        | Feature    | Binary     | Indicates if price exceeds category average|               | No             |
| page           | Feature    | Integer    | Page number within the e-store website   |               | No             |
## Dataset Characteristics
- **Multivariate, Sequential**: The dataset contains multiple variables and tracks sequential user interactions.
- **Subject Area**: Business, focusing on e-commerce user behavior analysis.
- **Associated Tasks**: Classification, Regression, Clustering.
- **Feature Types**: Integer, Real.
- **Number of Instances**: 165474.
- **Number of Features**: 14.
## Conclusion
Understanding user behavior in e-commerce is crucial for business success. This study provides insights into consumer preferences and decision-making processes, aiding in website optimization and marketing strategies. Further research could explore advanced analytical techniques and expand the analysis to different regions and languages, enriching the understanding of e-commerce dynamics.
