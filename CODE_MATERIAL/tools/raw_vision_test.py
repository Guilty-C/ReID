import os, json, sys, base64, glob
import requests

def pick_image():
    cands = []
    cands += glob.glob("iso_staging/query/*.jpg")
    cands += glob.glob("iso_staging/query/*.jpeg")
    cands += glob.glob("iso_staging/query/*.png")
    if cands:
        return cands[0]
    cands += glob.glob("data/mock_market/images/**/*.jpg", recursive=True)
    cands += glob.glob("data/mock_market/images/**/*.jpeg", recursive=True)
    cands += glob.glob("data/mock_market/images/**/*.png", recursive=True)
    return cands[0] if cands else None

def main():
    url = os.environ.get("CAPTION_API_URL")
    key = os.environ.get("CAPTION_API_KEY")
    model = os.environ.get("CAPTION_API_MODEL", "gpt-4o-mini")
    img = pick_image()
    print("URL=", url)
    print("KEY set=", bool(key))
    print("MODEL=", model)
    print("IMG=", img)
    if not url or not key or not img:
        print("Missing env or image.")
        sys.exit(1)
    if key.startswith("api:"):
        key = key.split(":",1)[1]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    with open(img, "rb") as f:
        enc = base64.b64encode(f.read()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{enc}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用中文简要描述这张行人照片的属性（性别、年龄段、上衣、下衣、颜色、背包、是否戴帽/眼镜）"},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}
                ]
            }
        ]
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        print("status=", r.status_code)
        print("resp=", r.text[:500])
    except Exception as e:
        print("error:", repr(e))

if __name__ == "__main__":
    main()