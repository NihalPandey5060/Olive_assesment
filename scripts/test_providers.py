import os
import sys
import json
import requests
from dotenv import load_dotenv
load_dotenv()
results = {}
# Test GROQ
groq_key = os.getenv('GROQ_API_KEY')
groq_base = os.getenv('GROQ_BASE_URL','https://api.groq.com/openai/v1')
groq_model = os.getenv('GROQ_MODEL','llama-3.1-8b-instant')
results['groq'] = {'present': bool(groq_key)}
if groq_key:
    try:
        url = groq_base.rstrip('/') + '/chat/completions'
        payload = {"model": groq_model, "messages": [{"role":"system","content":"You are a brief assistant."},{"role":"user","content":"Say hello"}]}
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type":"application/json"}
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        results['groq']['status_code'] = r.status_code
        try:
            results['groq']['body'] = r.json()
        except Exception:
            results['groq']['body'] = r.text
    except Exception as e:
        results['groq']['error'] = str(e)
else:
    results['groq']['error'] = 'GROQ_API_KEY not set'

# Test GEMINI presence
gem_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
results['gemini'] = {'present': bool(gem_key)}
if gem_key:
    results['gemini']['note'] = 'Key present; not performing live call to avoid extra SDK setup.'

print(json.dumps(results, indent=2))
