import requests
import json
import base64
import os

img_path = 'raw_combined/raw_combined/0/3447_right.jpeg'
url = 'http://127.0.0.1:8000/predict/image'

with open(img_path, 'rb') as f:
    r = requests.post(url, files={'file': f})

print('STATUS', r.status_code)
ct = r.headers.get('Content-Type', '')
print('CONTENT-TYPE', ct)

try:
    obj = r.json()
    print('KEYS:', list(obj.keys()))
    out_dir = 'saved_outputs'
    os.makedirs(out_dir, exist_ok=True)
    def save_data_url(name: str, value: str):
        if not isinstance(value, str):
            return False
        if "," in value:
            prefix, b64 = value.split(",", 1)
        else:
            b64 = value
        try:
            data = base64.b64decode(b64)
            path = os.path.join(out_dir, name)
            with open(path, 'wb') as wf:
                wf.write(data)
            print(f'Saved {name} ->', path)
            return True
        except Exception:
            return False

    if 'grad_cam_overlay' in obj:
        if not save_data_url('grad_cam_overlay.png', obj['grad_cam_overlay']):
            # fallback: save raw
            path = os.path.join(out_dir, 'grad_cam_overlay.txt')
            with open(path, 'w', encoding='utf-8') as wf:
                wf.write(str(obj['grad_cam_overlay']))
            print('Saved grad_cam_overlay text ->', path)

    if 'grad_cam_heatmap' in obj:
        if not save_data_url('grad_cam_heatmap.png', obj['grad_cam_heatmap']):
            path = os.path.join(out_dir, 'grad_cam_heatmap.txt')
            with open(path, 'w', encoding='utf-8') as wf:
                wf.write(str(obj['grad_cam_heatmap']))
            print('Saved grad_cam_heatmap text ->', path)

    # Print prediction fields if present
    for key in ('predicted_grade','predicted_label','risk_score','screening_tier','grade_probabilities'):
        if key in obj:
            print(f'{key}:', obj.get(key))

    # Print a compact JSON with key summaries
    summary = {k: (obj[k] if k in ('predicted_grade','grade_probabilities','predicted_label','risk_score','screening_tier') else type(obj[k]).__name__) for k in obj}
    print(json.dumps(summary, indent=2))
except Exception as e:
    txt = r.text
    print('RESPONSE TEXT (truncated):')
    print(txt[:20000])
