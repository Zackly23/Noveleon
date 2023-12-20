# Noveleon
## Genre Classification and Novel Recommendation Project

## Overview
This project aims to classify the genre of novels based on their synopses and provide genre-specific recommendations. The dataset used for this project underwent thorough exploratory data analysis (EDA) to gain insights into the distribution of genres, sequence lengths, and other relevant features.

## Exploratory Data Analysis (EDA)
### Genre Distribution
The genre distribution analysis provides insights into the balance or imbalance of the representation of Romance, History, Horror, and Fantasy genres within the dataset. Visualizations, such as histograms or pie charts, showcase the relative proportions of each genre. This analysis aids in understanding how well the classification model can be trained to recognize each genre effectively.

![Distribusi Genre](https://github.com/Zackly23/Noveleon/assets/65446701/05584502-2664-424d-a933-9b27d74b2152)

### Sequence Length Distribution
The distribution of sequence lengths in synopses for each genre is visualized through histograms or kernel density plots. This visualization highlights the variations in narrative lengths between genres. This information is crucial for decisions regarding the maximum sequence length during preprocessing and model architecture design.

![Distribusi Panjang Sinopsis](https://github.com/Zackly23/Noveleon/assets/65446701/8c448ee6-c467-4dee-b614-b75a093bf1a4)

### Genre-Specific Boxplots
Genre-specific boxplots offer a detailed view of the distribution of synopsis lengths, particularly emphasizing quartile ranges and outliers between genres. For instance, there may be genres with synopses that tend to be shorter or longer compared to other genres. This analysis helps determine whether sequence length handling needs to be genre-specific.

![Sequence Length Masin masing Genre](https://github.com/Zackly23/Noveleon/assets/65446701/f5f0091f-385a-4701-b1cc-88a92d77f3c2)

### Wordcloud 
Wordclouds visually represent the most frequently occurring words in the synopses of Romance, History, Horror, and Fantasy genres. These wordclouds help identify key words that encapsulate the distinctive characteristics of each genre. For example, in the Romance genre, words like "love," "relationships," and "romance" might be more dominant compared to other genres

![Edited_Fantasi](https://github.com/Zackly23/Noveleon/assets/65446701/a1f2085c-eb06-4230-bee9-07061288d07c)
![Edited_Sejarah](https://github.com/Zackly23/Noveleon/assets/65446701/38bdcb7c-545b-4304-bb85-d87007a4f08e)
![Edited_Romance](https://github.com/Zackly23/Noveleon/assets/65446701/a8e5bea0-0f75-48fb-9b2a-afc545101783)
![Edited_Horor](https://github.com/Zackly23/Noveleon/assets/65446701/8b012729-3705-4f03-af52-ef6f07e5e420)



