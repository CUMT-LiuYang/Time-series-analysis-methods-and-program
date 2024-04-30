# Time-series-analysis-methods-and-program
The editor can customize time series analysis to explore the temporal changes influenced by different factors. This project will use built-in air quality data as demonstration content to show how to conduct data analysis using this program and assist you in developing more in-depth analytical content.

## 1.Introduction of Document:

This document outlines the content of my undergraduate graduation project. 
It briefly summarizes key aspects and considerations of the data analysis and processing conducted during the research phase of the project, along with various analytical methods and patterns derived from the program. 
The aim is to assist others in reproducing and applying the program effectively. 

>Tips: Due to my limited proficiency in English, I've used ChatGPT 3.5 to assist in translation. 
While the translated text may not be colloquial, I hope readers will understand and pardon any discrepancies.



## 2.Entire Document Structure:
Whole-project/  
│  
├── python/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── project/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──.idea/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Myenv/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── .idea/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Air_quality.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Yourdata.csv  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Detrendofyourdata.csv  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Sup_pics/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Figofyourdata.pdf  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── requirements.txt  
├── Intermediate data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Time_integration.py  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── outputofyourdata.csv  
├── osfstorage-archive.zip  
└── Readme.txt  

## 3.Overview:
The main executable program is the Air_quality.py file. Currently, the program is modified for air quality data analysis, based on data from a specific site in Xuzhou, China, analyzing the factors affecting air time series variation (the source of data is unknown, and its authenticity is questionable). This program accepts specific data formats, detailed format specifications can be found in section 4. Document Description. The program references the paper "Fluctuations of water quality time series in rivers follow superstatistics," along with its accompanying program code, detailed references can be found in section 5. References. To facilitate the program's conversion into a time series processing and analysis program that readers need, this document will briefly explain some data processing considerations, for details please refer to section 3. Considerations.

## 4.Considerations:
### Ⅰ. Update file paths & formats
When changing data files, attention should be paid to modifying the file path. The default file path is "./data/2023_Xuzhou_Air_Indicators_Data_Every_One_Hour.csv", and the file input format is CSV. If you need to input files in other formats, please modify the input file type accordingly.

### Ⅱ. Mind time gaps
Attention should be paid to the possibility of non-continuous time distribution during the generation process of time series, which may result in inconsistencies between data and time. Therefore, generating a new time series directly may lead to mismatched data and time. Instead, it is advisable to process the original data to avoid this issue.

### Ⅲ. Standardize data formats
Standardizing data and formatting data: During the data processing process, standardize the data format according to the date, time, and various gas column data arrangement format as standardized data. Standardized data ensures correspondence between data and eliminates unnecessary interference from irrelevant data. Convert data text format, date format, and names according to the format requirements for readability into formatted data. For details, refer to .\project\Intermediate data\Xuzhou data processing integration. Chinese datasets are used here, and data set content names will be labeled in Chinese.

## 5.Document Description:
Specific data formats are followed for the introduced data. Correspond data by columns: the first column of data is the time series, and the next four columns correspond to different air quality data for the same time. The title and matrix dimensions for displaying patterns have been fixed when generating graphs. If new data needs to be introduced, modify the corresponding titles for the new data, graph dimensions, and names in the legend.

## 6.References:











<pre>-----------------------------------------------------------------------------
--            Author: Liu Yang                                             --
--                                                                         --
--            Location: China University of Mining and Technology          --
--                                                                         --
--            Contact Information:                                         --
--                E-mail_01: 1642577223@qq.com                             --
--                E-mail_02: 10204410@cumt.edu.cn                          --
--                Phone_01: +86-16675860876                                --
--                Phone_02: +86-19852497128                                --
--                                                                         --
--            Last Edited Date: April 28, 2024                             --
-----------------------------------------------------------------------------</pre>


