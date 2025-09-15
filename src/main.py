import requests
r = requests.get("https://api.github.com")
print("Status:", r.status_code)
print("Keys:", list(r.json().keys())[:5])
print("Hello, Github! My first tracked change.")