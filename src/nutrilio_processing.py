import pandas as pd
import numpy as np
import regex as re
import sys
import subprocess
import warnings
from drive_storage import update_file, update_drive
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.", category=PerformanceWarning)

age = 28
height_cm = 182
weight_kg = 79
activity_factor = 2.0  # using a PAL of 2.0 for highly active individuals
col_unnecessary_qty = ['Places', 'Origin', 'Work - location', 'Screen before sleep', 'Sleep - waking up', 'Sleep - night time']
dict_extract_data = {"Food" : "food",
                     "Drinks" : "drinks",
                     "Body sensations" : "body_sensations",
                     "Sleep - dreams" : "dreams",
                     "Work - content" : "work_content",
                     "Self improvement" : "self_improvement",
                     "Social activity" : "social_activity"}

dict_work_duration = {"0-2h" : 1, "0-4h" : 2, "2-3h" : 2.5, "3-4h" : 3.5, "4-6h" : 5, "4h" : 4, "6-8h" : 7,
                      "8-10h" : 9, "8h" : 8}

def generate_calory_needs():
    # Calculate the BMR using the Harris-Benedict equation
    bmr = 88.36 + (13.4 * weight_kg) + (4.8 * height_cm) - (5.7 * age)
    # Calculate the daily calorie needs by multiplying the BMR by the activity factor
    daily_calorie_needs = round(bmr * activity_factor)
    # Calculate the calorie breakdown for each meal and snack
    breakfast_calories = round(0.2 * daily_calorie_needs)
    morning_snack_calories = round(0.1 * daily_calorie_needs)
    lunch_calories = round(0.3 * daily_calorie_needs)
    afternoon_snack_calories = round(0.1 * daily_calorie_needs)
    dinner_calories = round(0.3 * daily_calorie_needs)
    night_snack_calories = round(0.05 * daily_calorie_needs)
    # Create a dictionary to store the calorie breakdown
    calorie_breakdown = {
        'Breakfast': breakfast_calories,
        'Morning snack': morning_snack_calories,
        'Lunch': lunch_calories,
        'Afternoon snack': afternoon_snack_calories,
        'Dinner': dinner_calories,
        'Night snack': night_snack_calories}
    return calorie_breakdown

def generate_dict_ingredients():
    meal_input_df = pd.read_excel('files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx', sheet_name='Sheet1')
    dict_ingredients = {}
    for _, row in meal_input_df.iterrows():
        dict_ingredients[row['Ingredient']] = {'Score' : row['Score'],
                                               'kcal' : row['kcal'],
                                               'Category' : row['Category'],
                                               'Quantity': {'A Little' : row['A little'],
                                                            'Medium' : row['Medium'],
                                                            'A Lot' : row['A lot']}}
    return dict_ingredients

def prompt_new_ingredients(df):
    chat_gpt_prompt = ("\n\n\n" + "Please give me, for each of these ingredients below, your best assessment for the different informations:\n"
    "1. The overall healthiness of this ingredient, from 1 (worst) to 10 (best)\n"
    "2. The kcal intake for 100g of this ingredient\n"
    "3. The ingredient category of the ingredient, within this list : ['Meat','Vegetables','Fruits','Carbs','Dairy','Sauces/Spices',"
    "'Veggie alternative', 'Fish', 'Meal category', 'Sweets']\n"
    "Please just give me the output and nothing else, and in the format of an excel table, with 4 columns 'Ingredient', 'Score', 'kcal' and 'Category'\n"
    "Here are the ingredients: \n")
    new_ingredients = ""
    dict_ingredients = generate_dict_ingredients()
    dict_new_ingredients = {}
    for _, row in df.iterrows():
        if (row['food_list'] != row['food_list']) | (not row['food_list']):
            continue
        ingredients = [ingredient.strip()[1:-1] for ingredient in row['food_list'].split(',')]
        for ingredient in ingredients:
            if ingredient not in dict_ingredients.keys():
                dict_new_ingredients[ingredient] = 1
    new_ingredients_counter = len(dict_new_ingredients)
    for ing in dict_new_ingredients.keys():
        new_ingredients += f"{ing} \n"
    if new_ingredients_counter == 0:
        return "OK"
    else:
        print(chat_gpt_prompt + new_ingredients)
        subprocess.Popen(['open', 'files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx'])
        return "NOK"

def ingredient_category(df):
    meal_input_df = pd.read_excel('files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx', sheet_name='Sheet1')
    dict_ingredients = generate_dict_ingredients()
    for cat in list(meal_input_df.Category.unique()):
        df[cat] = 0
    for index,row in df.iterrows():
        if (row['food_list'] != row['food_list']) | (row['food_list'] == "") :
            continue
        ingredients = [ingredient.strip()[1:-1] for ingredient in row['food_list'].split(',')]
        for ingredient in ingredients:
            df.loc[index, dict_ingredients[ingredient]["Category"]] +=1
    return df

def score_meal(meal, ingredients, quantity):
    dict_ingredients = generate_dict_ingredients()
    calorie_needs = generate_calory_needs()
    total_kcal = 0
    total_score = 0
    if (ingredients != ingredients) | (not ingredients):
        return None, None
    if meal not in calorie_needs.keys():
        return None, "Meal info missing"
    qty_divider = 1
    ingredients = [ingredient.strip()[1:-1] for ingredient in ingredients.split(',')]
    for ingredient in ingredients:
        if dict_ingredients[ingredient]["Category"] == "Meal category":
            qty_divider = 2
            continue
        if meal[-5:] == "snack":
            ing_quantity = dict_ingredients[ingredient]["Quantity"]["Medium"] / 2
        elif quantity != quantity:
            ing_quantity = dict_ingredients[ingredient]["Quantity"]["Medium"]
        else:
            ing_quantity = dict_ingredients[ingredient]["Quantity"][quantity]
        ing_kcal = dict_ingredients[ingredient]["kcal"]
        ing_score = dict_ingredients[ingredient]["Score"]
        total_kcal += ing_quantity/100 * ing_kcal
        total_score += (10 - ing_score) * (ing_quantity)/100 * ing_kcal
    if total_kcal == 0:
        return None, "missing data"
    avg_score = total_score / total_kcal
    total_kcal = total_kcal/qty_divider
    if total_kcal > calorie_needs[meal]:
        penalty = (total_kcal - calorie_needs[meal])
        avg_score = avg_score - penalty/1000
    return round((10-avg_score),2), None

def extract_data_count(df):
    pattern = r'([\w ]+) \((\d+)x\)'
    list_col = []
    for column, indicator in dict_extract_data.items():
        for index, row in df.iterrows():
            if row[column] != row[column]:
                continue
            list_elements = []
            matches = re.findall(pattern, row[column])
            start_point = 0
            if indicator == "food":
                start_point +=1
                meal = matches[0][0].strip()
                df.loc[index, "Meal"] = meal
            for match in matches[start_point:]:
                word = match[0].strip()
                value = int(match[1])
                df.at[index, f"{word}_{indicator}"] = value
                list_col.append(f"{word}_{indicator}")
                list_elements.append(word)
            df.loc[index, f"{indicator}_list"] = str(list_elements)[1:-1]
        df.drop(column, axis = 1, inplace = True)
    return df, list_col

def extract_data(column):
    if column != column:
        return None
    return str(column).split('(')[0].strip()

def drinks_category(drinks):
    df_drinks = pd.read_excel('files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx')
    chat_gpt_prompt = ("\n\n\n" + "Please give me your best assessment of the category for each of the drinks below.\n"
                       "Use these 4 categories: soda/soft drinks, strong alcohol, light alcohol, healthy drinks. \n"
                       "Give me as only output a table with 2 columns: Drink and Category, so that I can easily copy paste it in Excel. \n"
                       "Here are the drinks: \n")
    new_drinks_count = 0
    new_drinks_list = ""
    for drink in drinks:
        if drink in list(df_drinks.Drink.unique()):
            continue
        else:
            new_drinks_count += 1
            new_drinks_list += f"{drink} \n"
    if new_drinks_count == 0:
        return "OK"
    if new_drinks_count > 0:
        print(chat_gpt_prompt + new_drinks_list)
        subprocess.Popen(['open', 'files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx'])
    answer = input("Chat GPT output integrated in work file + file saved & closed? (Y/N) \n")
    while answer != 'Y':
        answer = input("Chat GPT output integrated in work file + file saved & closed? (Y/N) \n")

def generate_pbi_files(df, indicator):
    index_list = []
    for col in df.columns:
        if len(col.split('_')) > 1:
            if '_'.join(col.split('_')[1:]) == indicator:
                index_list.append(df.columns.get_loc(col))
        elif col == "Full Date":
            index_list.append(df.columns.get_loc(col))
    sliced_df = df.iloc[:, index_list]
    sliced_df.columns = [col.split('_')[0] for col in sliced_df.columns]
    melted_df = sliced_df.melt(id_vars='Full Date', value_vars=sliced_df.columns[1:], var_name=indicator, value_name='Value')
    melted_df = melted_df[melted_df['Value'] >= 1]
    melted_df.sort_values("Full Date", ascending = False).to_csv(f"files/processed_files/nutrilio_{indicator}_pbi_processed_file.csv", sep = '|', index = False)
    if indicator == "drinks":
        drinks_category(list(melted_df.drinks.unique()))

def process_nutrilio_export():
    df = pd.read_csv(f'files/exports/nutrilio_exports/nutrilio_export.csv', sep = ',')
    #Remove quantity when unnecessary
    for col in col_unnecessary_qty:
        df[col] = df[col].apply(lambda x: extract_data(x))
    #Extract all values & quantities
    df, list_col = extract_data_count(df)
    #Put the newly created columns at the end of the df to simplify checks
    columns_to_concat = []
    for _, val in dict_extract_data.items():
        column = df.pop(f"{val}_list")
        columns_to_concat.append(column)
    df = pd.concat([df.copy()] + columns_to_concat, axis=1)
    if prompt_new_ingredients(df) == "NOK":
        answer = input("Chat GPT output integrated in work file? (Y/N) \n")
        while answer != 'Y':
            answer = input("Chat GPT output integrated in work file? (Y/N) \n")
    #Compute each meal's healthiness score
    df['Score_meal'] = df.apply(lambda x : score_meal(x.Meal, x.food_list, x.Amount_text)[0], axis = 1)
    df['Meal_data_warning'] = df.apply(lambda x : score_meal(x.Meal, x.food_list, x.Amount_text)[1], axis = 1)
    df['Source'] = 'Nutrilio'
    df_daylio = pd.read_csv('files/processed_files/daylio_processed.csv', sep = '|')
    df = pd.concat([df, df_daylio], ignore_index=True).sort_values('Full Date')
    df = ingredient_category(df)
    df['Work_duration_est'] = df["Work - duration_text"].apply(lambda x: dict_work_duration[x] if x in dict_work_duration.keys() else None)
    df['Work - good day_text'] = df['Work - good day_text'].apply(lambda x: "Average" if x =="Ok" else x)
    df.drop(list_col, axis = 1).to_csv("files/processed_files/nutrilio_processed.csv", sep = '|', index = False)
    drive_list = ["files/processed_files/nutrilio_processed.csv",
                  "files/work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx",
                  "files/work_files/nutrilio_work_files/nutrilio_drinks_category.xlsx"]
    for _, value in dict_extract_data.items():
        generate_pbi_files(df, value)
        drive_list.append(f"files/processed_files/nutrilio_{value}_pbi_processed_file.csv")
    return drive_list

#process_nutrilio_export()
drive_list = ['files/processed_files/nutrilio_processed.csv', 'files/processed_files/nutrilio_food_pbi_processed_file.csv', 'files/processed_files/nutrilio_drinks_pbi_processed_file.csv',
              'files/processed_files/nutrilio_body_sensations_pbi_processed_file.csv', 'files/processed_files/nutrilio_dreams_pbi_processed_file.csv',
              'files/processed_files/nutrilio_work_content_pbi_processed_file.csv', 'files/processed_files/nutrilio_self_improvement_pbi_processed_file.csv',
              'files/processed_files/nutrilio_social_activity_pbi_processed_file.csv']
#update_drive(drive_list)
#update_file("work_files/nutrilio_work_files/nutrilio_meal_score_input.xlsx")
process_nutrilio_export()
#update_file("work_files/Objectives.xlsx")
#process_nutrilio_export()
#update_file('processed_files/offscreen_processed.csv')
