import requests, base64

# API endpoint
url = "http://127.0.0.1:8000/predict/image"

# Read an image (replace with your own test image)
with open("mah.jpg", "rb") as f:
    img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# Send request
payload = {"image": img_b64}
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())
