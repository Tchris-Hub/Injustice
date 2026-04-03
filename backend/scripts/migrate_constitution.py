import re
import json
import os
from pathlib import Path

def clean_line(line):
    """Remove page headers, footers, and common noise."""
    # Remove the standard footer/header text
    line = re.sub(r'The Constitution of the Federal Republic of Nigeria Updated with the First, Second, Third, Fourth and Fifth Alterations', '', line)
    # Remove page numbers at the end or start of lines (often preceded by the text above)
    line = re.sub(r'^\s*\d+\s*$', '', line)
    line = re.sub(r'\s*\d+\s*$', '', line)
    return line.strip()

def parse_constitution(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chapters = []
    current_chapter = None
    current_section = None
    
    # State flags
    in_body = False
    
    # Regex patterns
    chapter_re = re.compile(r'^CHAPTER\s+([IVXLCDM]+)$', re.IGNORECASE)
    section_re = re.compile(r'^(\d+[A-Z]?)\.\s+(.*)$')
    
    print(f"Total lines in file: {len(lines)}")
    
    for i, line in enumerate(lines):
        # 1. Skip TOC - Body starts after "DO HEREBY MAKE, ENACT AND GIVE TO OURSELVES"
        if not in_body:
            if "DO HEREBY MAKE, ENACT AND GIVE" in line:
                in_body = True
                print(f"Found body start at line {i+1}")
            continue
            
        cleaned = clean_line(line)
        if not cleaned:
            continue
            
        # Check for Chapter
        chap_match = chapter_re.match(cleaned)
        if chap_match:
            chap_num_roman = chap_match.group(1)
            # Find chapter title (usually the next non-empty cleaned line)
            chap_title = "Unknown"
            for next_line in lines[i+1:i+10]:
                next_cleaned = clean_line(next_line)
                if next_cleaned and not chapter_re.match(next_cleaned) and not section_re.match(next_cleaned):
                    chap_title = next_cleaned
                    break
            
            # Map Roman to Integer for DB chapter_id if needed, but keeping Roman for now or using index
            current_chapter = {
                "chapter_id": chap_num_roman,
                "title": chap_title,
                "sections": []
            }
            chapters.append(current_chapter)
            print(f"Parsed Chapter {chap_num_roman}: {chap_title}")
            continue

        # Check for Section
        sec_match = section_re.match(cleaned)
        if sec_match:
            sec_num = sec_match.group(1)
            sec_title = sec_match.group(2)
            
            current_section = {
                "section_number": sec_num,
                "title": sec_title,
                "content": "",
                "key_takeaway": ""
            }
            if current_chapter:
                current_chapter["sections"].append(current_section)
            continue
            
        # If we have a current section, accumulate content
        if current_section:
            # Skip noise like "[Section 16A is inserted by...]"
            if cleaned.startswith('[') and 'Alteration' in cleaned:
                continue
                
            if current_section["content"]:
                current_section["content"] += " " + cleaned
            else:
                current_section["content"] = cleaned

    return chapters

def save_to_json(data, output_path):
    # Flatten structure for database ingestion
    flattened = []
    chapter_map = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8
    }
    
    for chap in data:
        for sec in chap["sections"]:
            flattened.append({
                "chapter_id": chapter_map.get(chap["chapter_id"].upper(), 0),
                "section_number": sec["section_number"],
                "title": sec["title"],
                "content": sec["content"].strip(),
                "key_takeaway": ""
            })
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flattened, f, indent=2)
    print(f"Saved {len(flattened)} sections to {output_path}")

if __name__ == "__main__":
    input_file = r'c:\Users\USER\Desktop\Injustice\backend\data\nigerian_constitution_full.txt'
    output_file = r'c:\Users\USER\Desktop\Injustice\backend\data\constitution_parsed.json'
    
    if os.path.exists(input_file):
        parsed_data = parse_constitution(input_file)
        save_to_json(parsed_data, output_file)
    else:
        print(f"Error: Input file {input_file} not found.")
