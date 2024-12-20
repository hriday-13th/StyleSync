import requests
import json

url = "https://modelslab.com/api/v6/realtime/text2img"

payload = json.dumps({
	 "key" : "3gv7H6PUe4oVhRM1WvHZhwi5L1pRISYA4FQrqAB8nWTJxe0ngTvVyzJyQSFH",
	"prompt": "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner))",
	"negative_prompt": "bad quality",
	"width": "512",
	"height": "512",
	"safety_checker": False,
	"seed": None,
	"samples":1,
	"base64":False,
	"webhook": None,
	"track_id": None
})

headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)