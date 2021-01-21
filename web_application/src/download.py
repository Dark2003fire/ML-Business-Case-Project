import requests
from io import BytesIO


def download_large_file_from_google_drive(id):
    """Downloads large file from google drive"""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id})
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params)
        resp_obj = response.content
        bytes_obj = BytesIO(resp_obj)
        return bytes_obj


def get_confirm_token(response):
    """Gets the cookie for downloading a large file"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
