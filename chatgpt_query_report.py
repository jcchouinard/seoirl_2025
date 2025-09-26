import json
import re
import pandas as pd
from collections import Counter
from urllib.parse import urlparse

def extract_text_from_parts(parts):
    """ Helper function to recursively extract plain text from content parts. """
    if not isinstance(parts, list): return ""
    text_list = []
    for part in parts:
        if isinstance(part, str):
            text_list.append(part)
        elif isinstance(part, dict):
            for value in part.values():
                if isinstance(value, str):
                    text_list.append(value)
                elif isinstance(value, list):
                    text_list.append(extract_text_from_parts(value))
    return "".join(text_list)

def analyze_single_conversation(conversation):
    """
    Analyzes a single conversation object to extract link details.
    """
    all_links = []
    url_regex = re.compile(r'\[.*?\]\((https?://[^\s)]+)\)|(https?://[^\s)]+)')

    mapping = conversation.get('mapping', {})
    for message_id, message_data in mapping.items():
        message = message_data.get('message')
        if not message: continue

        metadata = message.get('metadata', {})
        content = message.get('content', {})
        
        def add_link(url, link_type, title=None):
            try:
                domain = urlparse(url).netloc
                all_links.append({'url': url, 'type': link_type, 'domain': domain, 'title': title})
            except Exception:
                pass

        if 'content_references' in metadata and isinstance(metadata.get('content_references'), list):
            for ref_object in metadata['content_references']:
                if ref_object.get('type') == 'grouped_webpages' and ref_object.get('items'):
                    for item in ref_object['items']:
                        if isinstance(item, dict) and (url := item.get('url')):
                            add_link(url, 'primary_citations', item.get('title'))
                
                if 'cite_map' in ref_object and isinstance(ref_object.get('cite_map'), dict):
                    for citation_obj in ref_object['cite_map'].values():
                        if isinstance(citation_obj, dict) and (url := citation_obj.get('url')):
                            add_link(url, 'sidebar_citations', citation_obj.get('title'))
                
                if ref_object.get('type') == 'businesses_map' and ref_object.get('businesses'):
                    for business in ref_object['businesses']:
                        if url := business.get('website_url'):
                            add_link(url, 'business_map', business.get('name'))

        if 'search_result_groups' in metadata:
            for group in metadata.get('search_result_groups', []):
                for entry in group.get('entries', []):
                    if url := entry.get('url'):
                        add_link(url, 'grouped_citations', entry.get('title'))

        if 'parts' in content:
            text = extract_text_from_parts(content.get('parts', []))
            for md_url, raw_url in url_regex.findall(text):
                if url := (md_url or raw_url):
                    add_link(url, 'decorated_links', None)

        if 'image_results' in metadata:
             for image in metadata.get('image_results', []):
                if url := (image.get('url') or image.get('content_url')):
                    add_link(url, 'images', image.get('title'))

    return all_links

def generate_report_by_title(input_filename="conversations.json"):
    """
    Parses conversations.json, generates a report for each conversation, and
    saves all data to a single CSV file.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{input_filename}' is not a valid JSON file.")
        return

    master_report_lines = ["="*80, "🔗 Link Visibility Analysis by Conversation Title", "="*80, ""]
    all_conversations_links = []

    for convo in conversations:
        title = convo.get("title", "Untitled Conversation")
        links_in_convo = analyze_single_conversation(convo)

        # Add conversation title to each link and append to master list for CSV
        for link in links_in_convo:
            link['conversation_title'] = title
        all_conversations_links.extend(links_in_convo)

        master_report_lines.append(f"\n{'='*20} Conversation: {title} {'='*20}\n")

        if not links_in_convo:
            master_report_lines.append("No links found in this conversation.")
            master_report_lines.append("\n" + "="*80)
            continue

        links_df = pd.DataFrame(links_in_convo).fillna({'title': 'No Title Provided'})
        
        # --- Generate Report Sections for this Conversation ---
        
        # Link Type Summary
        type_totals = links_df['type'].value_counts().to_dict()
        master_report_lines.append("## 📊 Link Type Summary ##\n")
        master_type_list = [
            'primary_citations', 'grouped_citations', 'sidebar_citations', 
            'business_map', 'decorated_links', 'images'
        ]
        for link_type in master_type_list:
            master_report_lines.append(f"{link_type:<25}: {type_totals.get(link_type, 0)} links")

        # Top 5 Domains
        domain_counts = links_df['domain'].value_counts()
        master_report_lines.append("\n## 🌐 Top 5 Domains by Link Count ##\n")
        top_5_domains = domain_counts.head(5)
        for domain, count in top_5_domains.items():
            master_report_lines.append(f"{domain:<50}: {count} links")

        # Top Domains by Title
        master_report_lines.append("\n## 📖 Top 5 Domains by Title ##\n")
        for domain in top_5_domains.index:
            master_report_lines.append(f"--- {domain} ---\n")
            domain_titles = links_df[links_df['domain'] == domain]['title'].value_counts()
            for t, count in domain_titles.items():
                 master_report_lines.append(f"  - [{count}x] {t}")
            master_report_lines.append("")
        
        master_report_lines.append("\n" + "="*80)

    # --- Save Master Report ---
    report_string = "\n".join(master_report_lines)
    report_output_filename = "report_by_title.txt"
    with open(report_output_filename, "w", encoding='utf-8') as f:
        f.write(report_string)
        
    # --- Save Master CSV ---
    csv_output_filename = "links_by_conversation_title.csv"
    if all_conversations_links:
        master_df = pd.DataFrame(all_conversations_links)
        ordered_cols = ['conversation_title', 'url', 'title', 'type', 'domain']
        master_df = master_df[ordered_cols]
        master_df.to_csv(csv_output_filename, index=False, encoding='utf-8')
    
    print(f"✅ Analysis Complete!")
    print(f"✅ Report saved to '{report_output_filename}'")
    if all_conversations_links:
        print(f"✅ Data saved to '{csv_output_filename}'")

if __name__ == "__main__":
    generate_report_by_title()