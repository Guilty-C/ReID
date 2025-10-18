import os, json, sys
import requests

def main():
    url = os.environ.get("CAPTION_API_URL")
    key = os.environ.get("CAPTION_API_KEY")
    model = os.environ.get("CAPTION_API_MODEL", "gpt-4o-mini")
    print("URL=", url)
    print("KEY set=", bool(key))
    print("MODEL=", model)
    if not url or not key:
        print("Missing env.")
        sys.exit(1)
    # If key has a prefix like 'api:', strip it for Bearer format
    if key.startswith("api:"):
        key = key.split(":",1)[1]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "用一句中文描述：一张行人照片"}
        ]
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        print("status=", r.status_code)
        print("resp=", r.text)
    except Exception as e:
        print("error:", repr(e))

if __name__ == "__main__":
    main()