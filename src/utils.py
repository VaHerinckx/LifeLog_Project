import pandas as pd
import datetime
from datetime import datetime

time_diff_excel = pd.read_excel('files/work_files/GMT_timediff.xlsx')


def today_export():
    current_date = datetime.now()
    date_string = current_date.strftime("%d%m%y")
    return date_string

def time_difference_correction(date, timezone_source='GMT'):
    """Corrects timezone differences"""
    time_diff_excel["Date"] = pd.to_datetime(time_diff_excel["Date"], utc = True)
    for index, row_datefile in time_diff_excel.iterrows():
        if date >= row_datefile['Date']:
            continue
        elif index == 0:
            return date + pd.Timedelta(hours=time_diff_excel.loc[index][f'Time_diff_{timezone_source}'])
        else:
            return date + pd.Timedelta(hours=time_diff_excel.loc[index-1][f'Time_diff_{timezone_source}'])
    return date + pd.Timedelta(hours=time_diff_excel.loc[time_diff_excel.index[-1]][f'Time_diff_{timezone_source}'])

def get_response(client, system_prompt, user_prompt):
  # Assign the role and content for each message
  messages = [{"role": "system", "content": system_prompt},
      		  {"role": "user", "content": user_prompt}]
  response = client.chat.completions.create(
      model="gpt-3.5-turbo", messages= messages, temperature=0)
  return response.choices[0].message.content
