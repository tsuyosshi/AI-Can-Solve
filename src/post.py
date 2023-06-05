import requests
import json
post_url = "http://127.0.0.1:80/read_pazzle" 
json = {"bytes": [2, 3, 4]} 

response = requests.post(
                   post_url,
                   json = json,
                   )
print(response.json())