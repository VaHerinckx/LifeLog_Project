![The Lifelog Project](images/lifelog_logo.png)
<div align="center">
  <img src="images/lifelog_logo.png" alt="Image Description">
</div>


## Introduction
After listening to Datacamp's Dataframed podcast episode discussing how people can use their own personal data for all sorts of things (with Gary Wolf as guest, link [here](https://www.datacamp.com/podcast/data-driven-thinking-for-the-everyday-life)), I got inspired, and decided to use my newly acquired data skills to do something with all the data that's floating around me. 

What I wanted to do with it was not very clear at first, but as I do a lot of dashboarding for my work, I thought a cool personnal dashboard would be a good start. This repo contains all the python code I used to :

1. Open all the different url of the websites where I can retrieve my personnal data, before I manually download them 
2. Arrange the export files from the different sources in the correct places.
3. Get the data contained in the files into dataframes containing only the data I'm interested in for my dashboard
4. Upload the processed files to a Google Drive location, which is linked to my PowerBI report.

Feel free to reuse some parts of this code if this proves useful for your project, below is the description of how the code works in general, followed by the description of what each file does ⬇️

## General flow of the code
