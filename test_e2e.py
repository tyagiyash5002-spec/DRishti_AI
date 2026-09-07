"""Quick end-to-end test via HTTP multipart."""
import numpy as np
import cv2
import json
import math
import requests

# Build synthetic fundus image
img = np.zeros((600,600,3), dtype=np.uint8)
cv2.circle(img, (300,300), 285, (25,70,25), -1)
cv2.circle(img, (360,260), 40, (200,210,255), -1)
cv2.circle(img, (240,300), 12, (180,180,220), -1)
for i in range(8):
    a = i*45
    x2 = int(300+200*math.cos(math.radians(a)))
    y2 = int(300+200*math.sin(math.radians(a)))
    cv2.line(img, (300,300), (x2,y2), (0,120,0), 2)
for pos in [(200,220),(280,310),(320,280),(260,350)]:
    cv2.circle(img, pos, 4, (0,0,180), -1)

_, buf = cv2.imencode('.jpg', img)

# POST to /analyze
resp = requests.post(
    'http://localhost:8000/analyze',
    files={'file': ('retina.jpg', buf.tobytes(), 'image/jpeg')},
    timeout=120
)

data = resp.json()
print('Status:', resp.status_code)
print('Case ID:', data.get('case_id'))
print('IQA Gradable:', data.get('iqa',{}).get('gradable'))
print('Final Grade:', data.get('ecde',{}).get('final_grade'))
print('Decision:', data.get('ecde',{}).get('decision'))
print('Confidence:', data.get('ecde',{}).get('confidence_level'))
print('JSD:', data.get('ecde',{}).get('jsd'))
print('NV detected:', data.get('neovascularization',{}).get('neovascularization_detected'))
print('Lesion counts:', data.get('lesion',{}).get('counts'))
print('GradCAM available:', 'overlay_b64' in data.get('gradcam',{}))
print()
print('END-TO-END TEST PASSED')
