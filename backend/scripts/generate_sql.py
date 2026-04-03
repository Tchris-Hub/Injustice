import json

def run_ingest():
    with open('data/constitution_enriched.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sql = 'INSERT INTO constitution_sections (chapter_id, section_number, section_title, content, key_takeaway) VALUES \n'
    values = []
    
    for item in data:
        ch_id = item['chapter_id']
        sec_num = str(item['section_number']).replace("'", "''")
        title = str(item['title']).replace("'", "''")[:255]
        content = str(item['content']).replace("'", "''")
        takeaway = str(item['key_takeaway']).replace("'", "''")
        values.append(f"({ch_id}, '{sec_num}', '{title}', '{content}', '{takeaway}')")
        
    sql += ',\n'.join(values) + '\nON CONFLICT (chapter_id, section_number) DO UPDATE SET section_title=EXCLUDED.section_title, content=EXCLUDED.content, key_takeaway=EXCLUDED.key_takeaway;'
    
    with open('data/ingest_constitution_prwcbruqvywakcvoknai.sql', 'w', encoding='utf-8') as f:
        f.write('BEGIN;\n')
        f.write('''
CREATE TABLE IF NOT EXISTS constitution_sections (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    chapter_id SMALLINT NOT NULL,
    section_number VARCHAR(10) NOT NULL,
    section_title VARCHAR(255),
    content TEXT NOT NULL,
    key_takeaway TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
    UNIQUE(chapter_id, section_number)
);
''')
        f.write(sql)
        f.write('\nCOMMIT;\n')
        
    print('Done writing SQL file.')

if __name__ == "__main__":
    run_ingest()
