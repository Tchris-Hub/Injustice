import json
import psycopg2
import os

DB_URL = "postgresql://postgres:SupabaseDB2026@db.fdjltnfkmskeqtiaqlou.supabase.co:5432/postgres"

def run_ingest():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        with open('data/constitution_enriched.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Starting import of {len(data)} sections into DB...")
        
        for idx, item in enumerate(data):
            ch_id = item['chapter_id']
            sec_num = item['section_number']
            title = item['title'].replace("'", "''")[:255]
            content = item['content'].replace("'", "''")
            takeaway = item['key_takeaway'].replace("'", "''")
            
            cur.execute(f'''
            INSERT INTO constitution_sections (chapter_id, section_number, section_title, content, key_takeaway)
            VALUES ({ch_id}, '{sec_num}', '{title}', '{content}', '{takeaway}')
            ON CONFLICT (chapter_id, section_number) 
            DO UPDATE SET section_title=EXCLUDED.section_title, content=EXCLUDED.content, key_takeaway=EXCLUDED.key_takeaway;
            ''')
            
            if (idx + 1) % 50 == 0:
                print(f"Imported {idx + 1} sections.")
                
        conn.commit()
        cur.close()
        conn.close()
        print("Finished mapping JSON to PostgreSQL!")
    except Exception as e:
        print(f"Error during import: {e}")

if __name__ == "__main__":
    run_ingest()
