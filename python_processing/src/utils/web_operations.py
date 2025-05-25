import subprocess


def open_web_urls(urls):
    """Opens all the web urls in Firefox"""
    for url in urls:
        subprocess.run(['open', '-a', 'Firefox', '-g', url])


def prompt_user_download_status(data_source_name):
    """Prompts user if the download was completed"""
    response = input(f"Did {data_source_name} export get downloaded? (Y/N) ")
    return response.upper() == 'Y'


def prompt_user_request_status(data_source_name):
    """Prompts user if the data request was submitted (for sources that send data later)"""
    response = input(f"Was {data_source_name} data requested? (Y/N) ")
    return response.upper() == 'Y'
