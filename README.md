<div align="center">
  <img src="Design/images/lifelog_logo.png" alt="The Lifelog Project">
</div>

After listening to Datacamp's Dataframed podcast episode discussing how people can use their own personal data for all sorts of things (with Gary Wolf as guest, link [here](https://www.datacamp.com/podcast/data-driven-thinking-for-the-everyday-life)), I got inspired, and decided to use my newly acquired data skills to do something with all the data that's floating around me.

More information about what I actually did with all this data can be found in [this Medium article](mock_link), Should you have any interest in re-using some of this code for a personal project of yours, feel free! The general flow of the code in this repo is the following, when process_exports.py is executed:

1. Open all the different url of the websites where I can retrieve my personnal data, before I manually download them
2. Arrange the export files from the different sources in the correct places.
3. Get the data contained in the files into dataframes containing only the data I'm interested in for my dashboard (with eventual input from the user requested in the process)
4. Upload the processed files to a Google Drive location, which is linked to my PowerBI report.

For more details, please refer to the comments in the different pyton files!
