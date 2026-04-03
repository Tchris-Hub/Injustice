import json
import os
import re
from fastembed import TextEmbedding

# Paths
INPUT_PATH = r"C:\Users\USER\.gemini\antigravity\brain\248a7c34-92a2-41b2-bd0f-7df209868a18\.system_generated\steps\2888\output.txt"
OUTPUT_SQL = r"C:\Users\USER\Desktop\Injustice\backend\scripts\update_embeddings.sql"

def parse_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result_str = data.get('result', '')
    # Extract the JSON array inside <untrusted-data-...> tags
    match = re.search(r'<untrusted-data-[^>]+>\n(.*?)\n</untrusted-data-', result_str, re.DOTALL)
    if not match:
        # Fallback to simple list extraction if tags are different
        match = re.search(r'\[.*\]', result_str, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group(1))
        except:
             # Try group 0 for fallback
             return json.loads(match.group(0))
    return []

def generate_embeddings():
    sections = parse_input(INPUT_PATH)
    if not sections:
        print("❌ No sections found.")
        return

    print(f"🔋 Loading model BAAI/bge-base-en-v1.5 (768-dim) for {len(sections)} sections...")
    model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
    
    contents = [s['content'] for s in sections]
    embeddings = list(model.embed(contents))
    
    print(f"✅ Generated {len(embeddings)} embeddings. Writing SQL...")
    
    with open(OUTPUT_SQL, 'w', encoding='utf-8') as f:
        for section, emb in zip(sections, embeddings):
            # emb is a numpy array
            emb_list = emb.tolist()
            # PostgreSQL vector format: '[0.1, 0.2, ...]'
            emb_str = str(emb_list)
            sql = f"UPDATE constitution_sections SET embedding = '{emb_str}' WHERE id = {section['id']};\n"
            f.write(sql)
    
    print(f"🚀 SQL saved to {OUTPUT_SQL}")

if __name__ == "__main__":
    generate_embeddings()
