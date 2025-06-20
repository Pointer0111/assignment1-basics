import requests

x = requests.get("https://www.google.com")

print(x.text)