import json
import os
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
OPENROUTER_API_KEY = "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"
AI_MODELS = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen-72b-chat:free"
]
INPUT_FILE = "data/constitution_parsed.json"
ENRICHED_FILE = "data/constitution_enriched.json"
SQL_FILE = "data/ingest_constitution.sql"

def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
def get_ai_takeaway(content, model_id):
    if not content or len(content.strip()) < 10:
        return ""
    
    prompt = f"Summarize the legal essence of this Nigerian Constitution section in one concise sentence (max 25 words) for a common citizen. Focus on what it means for their rights or duties:\n\n{content[:2000]}"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Tchris-Hub/Injustice",
        "X-Title": "My Rights Injustice"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a legal expert helping Nigerian citizens understand their constitution."},
            {"role": "user", "content": prompt}
        ]
    }
    
    with httpx.Client(timeout=45.0) as client:
        try:
            response = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"Model {model_id} failed with {response.status_code}. Response: {response.text[:200]}")
                return None
        except Exception as e:
            print(f"Connection error for {model_id}: {e}")
            return None

def enrich_data():
    data = load_data(INPUT_FILE)
    enriched = []
    
    # Load existing to resume if possible
    if os.path.exists(ENRICHED_FILE):
        with open(ENRICHED_FILE, 'r', encoding='utf-8') as f:
            enriched = json.load(f)
    
    processed_count = len(enriched)
    print(f"Starting enrichment. {processed_count}/{len(data)} already done.")
    
    for i, item in enumerate(data[processed_count:], start=processed_count):
        print(f"Processing section {item['section_number']} ({i+1}/{len(data)})...")
        content = item['content']
        takeaway = ""
        if content:
             # Just grab the first ~50-100 characters up to the first period for a preview summary
             preview = content.split('.')[0]
             takeaway = preview[:150] + "..." if len(preview) > 150 else preview + "."
             
        if not takeaway or len(takeaway.strip()) < 5:
            takeaway = "[Legal section - summary pending]"

        item['key_takeaway'] = takeaway
        enriched.append(item)
        
        if (i + 1) % 50 == 0:
            with open(ENRICHED_FILE, 'w', encoding='utf-8') as f:
                json.dump(enriched, f, indent=2)
            
    with open(ENRICHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2)
    print("Enrichment complete.")

def generate_sql():
    with open(ENRICHED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sql_lines = []
    sql_lines.append("DELETE FROM public.constitution_sections;\n")
    
    for item in data:
        # Sanitize values for SQL
        chapter_id = item['chapter_id']
        section_number = item['section_number'].replace("'", "''")
        title = item['title'].replace("'", "''")
        content = item['content'].replace("'", "''")
        takeaway = item['key_takeaway'].replace("'", "''")
        
        sql_lines.append(
            f"INSERT INTO public.constitution_sections (chapter_id, section_number, title, content, key_takeaway) "
            f"VALUES ({chapter_id}, '{section_number}', '{title}', '{content}', '{takeaway}');\n"
        )
        
    with open(SQL_FILE, 'w', encoding='utf-8') as f:
        f.writelines(sql_lines)
    print(f"SQL file generated at {SQL_FILE}")

if __name__ == "__main__":
    if not os.path.exists(ENRICHED_FILE) or len(load_data(ENRICHED_FILE)) < len(load_data(INPUT_FILE)):
        enrich_data()
    generate_sql()
