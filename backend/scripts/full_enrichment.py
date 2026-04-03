import json
import os
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
OPENROUTER_API_KEY = "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"
MODEL_ID = "google/gemini-2.0-flash:free"
INPUT_FILE = "c:/Users/USER/Desktop/Injustice/backend/data/constitution_parsed.json"
ENRICHED_FILE = "c:/Users/USER/Desktop/Injustice/backend/data/constitution_complete.json"
SQL_FILE = "c:/Users/USER/Desktop/Injustice/backend/data/full_ingest.sql"

def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_ai_takeaway(content, model_id):
    if not content or len(content.strip()) < 10:
        return "[Brief section]"
    
    prompt = f"Summarize the core essence of this Nigerian Constitution section in one very concise sentence (max 20 words) for a common citizen. Avoid preamble, just the essence:\n\n{content[:2000]}"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Tchris-Hub/Injustice",
        "X-Title": "My Rights Injustice Ingestion"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a concise legal guide for Nigerian citizens."},
            {"role": "user", "content": prompt}
        ]
    }
    
    with httpx.Client(timeout=60.0) as client:
        try:
            response = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip().replace('"', '')
            elif response.status_code == 429:
                print(f"Rate limit hit. Waiting...")
                time.sleep(5)
                raise Exception("Rate limit")
            else:
                print(f"Error {response.status_code}: {response.text[:200]}")
                return None
        except Exception as e:
            print(f"Connection error: {e}")
            raise e

def run_enrichment():
    data = load_data(INPUT_FILE)
    if not data:
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    # Filter for standard sections (1 to 320, plus those with A, B, C suffixes)
    # The current parsed data might contain more, but we want a high quality set.
    # We will just process all sections that have content and a title.
    
    enriched = []
    if os.path.exists(ENRICHED_FILE):
        enriched = load_data(ENRICHED_FILE)
        print(f"Loaded {len(enriched)} existing enriched sections.")
    
    processed_titles = {item['title'] for item in enriched}
    
    count = 0
    total = len(data)
    
    for i, item in enumerate(data):
        if item['title'] in processed_titles:
            continue
            
        print(f"[{i+1}/{total}] Processing Section {item.get('section_number', 'Unknown')}: {item.get('title', 'Unknown')}...")
        
        takeaway = get_ai_takeaway(item['content'], MODEL_ID)
        if takeaway:
            item['key_takeaway'] = takeaway
            enriched.append(item)
            processed_titles.add(item['title'])
            count += 1
            
            # Local checkpoint
            if count % 10 == 0:
                with open(ENRICHED_FILE, 'w', encoding='utf-8') as f:
                    json.dump(enriched, f, indent=2)
        
        # Respect rate limits for free models
        time.sleep(1.5)

    with open(ENRICHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2)
    print(f"Enrichment complete. Total: {len(enriched)} sections.")

def generate_sql():
    data = load_data(ENRICHED_FILE)
    if not data:
        print("No enriched data to generate SQL.")
        return

    sql_lines = [
        "BEGIN;\n",
        "TRUNCATE TABLE public.constitution_sections RESTART IDENTITY;\n"
    ]
    
    for item in data:
        chapter_id = item.get('chapter_id', 0)
        section_number = str(item.get('section_number', '')).replace("'", "''")
        title = str(item.get('title', '')).replace("'", "''")
        content = str(item.get('content', '')).replace("'", "''")
        takeaway = str(item.get('key_takeaway', '')).replace("'", "''")
        
        sql_lines.append(
            f"INSERT INTO public.constitution_sections (chapter_id, section_number, title, content, key_takeaway) "
            f"VALUES ({chapter_id}, '{section_number}', '{title}', '{content}', '{takeaway}');\n"
        )
        
    sql_lines.append("COMMIT;\n")
    
    with open(SQL_FILE, 'w', encoding='utf-8') as f:
        f.writelines(sql_lines)
    print(f"Full SQL script generated at {SQL_FILE}")

if __name__ == "__main__":
    run_enrichment()
    generate_sql()
