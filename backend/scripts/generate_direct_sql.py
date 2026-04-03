import json
import os

INPUT_FILE = r'C:\Users\USER\Desktop\Injustice\backend\data\constitution_parsed.json'
SQL_OUTPUT = r'C:\Users\USER\Desktop\Injustice\backend\data\full_ingest_direct.sql'

def get_internal_summary(section_number, title, content):
    summaries = {
        "1": "The Constitution is the supreme law of Nigeria; any conflicting law is void.",
        "2": "Nigeria is a single, indivisible sovereign state known as the Federal Republic of Nigeria.",
        "3": "Nigeria consists of 36 States and a Federal Capital Territory (Abuja).",
        "4": "Legislative power belongs to the National Assembly (Senate and House of Representatives) and State Assemblies.",
        "5": "Executive power is vested in the President for the Federation and Governors for the States.",
        "6": "Judicial power is vested in the courts, including the Supreme Court and other established courts.",
        "7": "Local government systems by democratically elected councils are guaranteed by the Constitution.",
        "8": "Outlines the strict procedures for creating new states or adjusting boundaries.",
        "9": "The National Assembly has the power to amend the Constitution through specific voting majorities.",
        "10": "Neither the Federation nor any State shall adopt any religion as a State Religion.",
        "11": "The National Assembly can make laws for public safety and order during emergencies.",
        "12": "Treaties with other countries must be enacted into law by the National Assembly to be valid.",
        "13": "All government authorities must conform to the Fundamental Objectives of the State.",
        "14": "Nigeria is a state based on democracy and social justice; sovereignty belongs to the people.",
        "15": "The State shall promote national integration and discourage discrimination based on origin or religion.",
        "16": "The State must manage the economy to ensure maximum welfare, freedom, and happiness for all citizens.",
        "17": "The State social order is founded on ideals of Freedom, Equality, and Justice.",
        "18": "The Government shall strive to eradicate illiteracy and provide free education when possible.",
        "19": "Nigeria's foreign policy shall promote national interest, African unity, and international cooperation.",
        "20": "The State shall protect and improve the environment and safeguard water, air, land, and wildlife.",
        "21": "The State shall protect and preserve Nigerian culture and encourage technological development.",
        "22": "The press and media shall be free to uphold the responsibility and accountability of the Government.",
        "23": "The national ethics of Nigeria shall be Discipline, Integrity, Dignity of Labour, Social Justice, and Patriotism.",
        "24": "It is the duty of every citizen to respect the Constitution, the National Flag, and the National Anthem.",
        "33": "Every person has a right to life, which can only be taken in execution of a court sentence.",
        "34": "Every individual is entitled to respect for the dignity of their person; no torture or slavery.",
        "35": "Every person is entitled to personal liberty; no one shall be deprived of such liberty except by law.",
        "36": "In the determination of civil rights or criminal charges, every person is entitled to a fair hearing.",
        "37": "The privacy of citizens, their homes, correspondence, and telephone conversations is guaranteed.",
        "38": "Every person is entitled to freedom of thought, conscience, and religion.",
        "39": "Every person is entitled to freedom of expression, including freedom to hold opinions and impart ideas.",
        "40": "Every person is entitled to assemble freely and associate with others, including political parties.",
        "41": "Every citizen is entitled to move freely throughout Nigeria and reside in any part thereof.",
        "42": "No Nigerian citizen shall be discriminated against based on community, ethnic group, place of origin, sex, religion, or political opinion.",
        "43": "Every citizen shall have the right to acquire and own immovable property anywhere in Nigeria.",
        "44": "No property shall be taken possession of or acquired compulsorily except in the manner prescribed by law."
    }
    
    if section_number in summaries:
        return summaries[section_number]
    
    if "court" in content.lower():
        return f"Defines and regulates the powers and jurisdiction of {title}."
    if "power" in content.lower():
        return f"Specifies the legal powers and limitations regarding {title}."
    
    return f"Establishes the legal framework and constitutional provisions for {title}."

def run():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Generating SQL for {len(data)} sections...")
    sql_lines = [
        "BEGIN;\n",
        "TRUNCATE TABLE public.constitution_sections RESTART IDENTITY;\n"
    ]
    
    # We will build a single multi-row INSERT query to make it robust
    insert_prefix = "INSERT INTO public.constitution_sections (chapter_id, section_number, title, content, key_takeaway) VALUES "
    sql_lines.append(insert_prefix)
    
    values = []
    # Filter for standard sections 1 to 320 to avoid bloat, 
    # the parsed file actually has roughly 320 unique sections based on previous looks
    
    # Need to keep track of unique (chapter_id, section_number) because of the UNIQUE constraint
    seen = set()
    filtered_data = []
    
    for item in data:
        chapter_id = item.get('chapter_id', 1)
        section_number = str(item.get('section_number', '')).strip()
        if not section_number:
            continue
            
        key = (chapter_id, section_number)
        if key in seen:
            continue
        seen.add(key)
        filtered_data.append(item)
    
    for i, item in enumerate(filtered_data):
        chapter_id = item.get('chapter_id', 1)
        section_number = str(item.get('section_number', '')).strip()
        
        # Handle single quotes properly
        title = item.get('title', '').replace("'", "''")
        content = item.get('content', '').replace("'", "''")
        summary = get_internal_summary(section_number, title, content).replace("'", "''")
        
        value_str = f"({chapter_id}, '{section_number}', '{title}', '{content}', '{summary}')"
        values.append(value_str)
    
    sql_lines.append(",\n".join(values) + ";\n")
    sql_lines.append("COMMIT;\n")
    
    with open(SQL_OUTPUT, 'w', encoding='utf-8') as f:
        f.writelines(sql_lines)
    
    print(f"Success! SQL generated at {SQL_OUTPUT}")

if __name__ == "__main__":
    run()
