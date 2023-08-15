<div align="center">
  <img src="images/lifelog_logo.png" alt="Image Description">
</div>


## Introduction
After listening to Datacamp's Dataframed podcast episode discussing how people can use their own personal data for all sorts of things (with Gary Wolf as guest, link [here](https://www.datacamp.com/podcast/data-driven-thinking-for-the-everyday-life)), I got inspired, and decided to use my newly acquired data skills to do something with all the data that's floating around me.

More information about what I actually did with all this data can be found in [this Medium article](mock_link), 

1. Open all the different url of the websites where I can retrieve my personnal data, before I manually download them 
2. Arrange the export files from the different sources in the correct places.
3. Get the data contained in the files into dataframes containing only the data I'm interested in for my dashboard (with eventual input from the user requested in the process)
4. Upload the processed files to a Google Drive location, which is linked to my PowerBI report.

Feel free to reuse some parts of this code if this proves useful for your project, below is a general description of how each source is processed ⬇️

**Attention point**: I removed the folders containing my own files and the different secrets I need to use throughout the code from this repo, so the code won't work as is and you'll need to replace certain paths for it to work. 

## Description of the different python files

### Apple Health
1. Given that the .xml file given by Apple Health seems to have faulty data each time I export it, the first step is to remove the faulty rows
2. I then create dataframes for each type of data I'm interested in.
3. These dataframes are then expanded, so that I get one row per minute, instead of aggregations over longer periods, which will then simplify merging the data with other sources. Each dataframe is then saved into a .csv file. 
4. The expanded dfs are then finally merged together, and saved into one .csv file.

### Garmin
