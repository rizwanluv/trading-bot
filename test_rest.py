import requests

api_key = "FAKE_KEY"
m_name = "gemini-2.5-flash"
prompt_text = "test"
system_instruction = "test system"
max_tokens = 250

url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
payload = {
    "contents": [{"parts": [{"text": prompt_text}]}],
    "systemInstruction": {"parts": [{"text": system_instruction}]},
    "generationConfig": {"maxOutputTokens": max_tokens},
}
r = requests.post(url, json=payload)
print(f"Status: {r.status_code}")
print(f"Response: {r.text}")
