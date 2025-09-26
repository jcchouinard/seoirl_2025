import json
import re
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
import matplotlib.pyplot as plt

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

def definitive_parse_and_analyze(input_filename="conversations.json"):
    """
    Parses conversations.json, generates a report, saves a CSV, and creates
    visual plots of the link data.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        error_msg = f"Error: The input file '{input_filename}' was not found."
        print(error_msg)
        return (error_msg, None)
    except json.JSONDecodeError:
        error_msg = f"Error: The file '{input_filename}' is not a valid JSON file."
        print(error_msg)
        return (error_msg, None)

    all_links = []
    url_regex = re.compile(r'\[.*?\]\((https?://[^\s)]+)\)|(https?://[^\s)]+)')

    for convo in conversations:
        mapping = convo.get('mapping', {})
        for message_id, message_data in mapping.items():
            message = message_data.get('message')
            if not message: continue

            metadata = message.get('metadata', {})
            content = message.get('content', {})
            
            def add_link(url, link_type):
                try:
                    domain = urlparse(url).netloc
                    all_links.append({'url': url, 'type': link_type, 'domain': domain})
                except Exception:
                    pass

            if 'content_references' in metadata and isinstance(metadata.get('content_references'), list):
                for ref_object in metadata['content_references']:
                    if ref_object.get('type') == 'grouped_webpages' and ref_object.get('items'):
                        for item in ref_object['items']:
                            if isinstance(item, dict) and (url := item.get('url')):
                                add_link(url, 'primary_citations')
                    
                    if 'cite_map' in ref_object and isinstance(ref_object.get('cite_map'), dict):
                        for citation_obj in ref_object['cite_map'].values():
                            if isinstance(citation_obj, dict) and (url := citation_obj.get('url')):
                                add_link(url, 'sidebar_citations')
                    
                    if ref_object.get('type') == 'businesses_map' and ref_object.get('businesses'):
                        for business in ref_object['businesses']:
                            if url := business.get('website_url'):
                                add_link(url, 'business_map')

            if 'search_result_groups' in metadata:
                for group in metadata.get('search_result_groups', []):
                    for entry in group.get('entries', []):
                        if url := entry.get('url'):
                            add_link(url, 'grouped_citations')

            if 'parts' in content:
                text = extract_text_from_parts(content.get('parts', []))
                for md_url, raw_url in url_regex.findall(text):
                    if url := (md_url or raw_url):
                        add_link(url, 'decorated_links')
            
            if 'image_results' in metadata:
                 for image in metadata.get('image_results', []):
                    if url := (image.get('url') or image.get('content_url')):
                        add_link(url, 'images')

    if not all_links:
        print("No links found to analyze.")
        return ("No links found.", None)

    links_df = pd.DataFrame(all_links)
    type_counts = links_df['type'].value_counts()

    # --- PLOTTING ---

    # Plot 1: Link Type Distribution Bar Chart
    plt.figure(figsize=(12, 7))
    type_counts.sort_values(ascending=True).plot(kind='barh')
    plt.title('Distribution of Link Types')
    plt.xlabel('Number of Links')
    plt.ylabel('Link Type')
    plt.tight_layout()
    plt.savefig('link_type_distribution.png')
    plt.close()

    # Plot 2: Top Domains by Link Type
    domain_link_type_pivot = links_df.groupby(['domain', 'type']).size().unstack(fill_value=0)
    top_domains = domain_link_type_pivot.sum(axis=1).nlargest(10).index
    top_domains_pivot = domain_link_type_pivot.loc[top_domains]
    top_domains_pivot.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Link Type Breakdown for Top 10 Domains')
    plt.ylabel('Number of Links')
    plt.xlabel('Domain')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Link Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('top_domains_by_link_type.png')
    plt.close()

    # --- REPORT, DATAFRAME, and CSV ---
    type_totals = type_counts.to_dict()
    domain_counts = links_df['domain'].value_counts()
    
    report_lines = ["="*80, "🔗 Definitive Link Visibility Analysis Report", "="*80]
    
    report_lines.append("\n## 📊 Link Type Summary ##\n")
    master_type_list = [
        'primary_citations', 'grouped_citations', 'sidebar_citations', 
        'business_map', 'decorated_links', 'images'
    ]
    for link_type in master_type_list:
        report_lines.append(f"{link_type:<25}: {type_totals.get(link_type, 0)} links")

    report_lines.append("\n## 🌐 Top 10 Domains by Link Count ##\n")
    for domain, count in domain_counts.head(10).items():
        report_lines.append(f"{domain:<50}: {count} links")

    report_string = "\n".join(report_lines)

    with open("link_visibility_report.txt", "w", encoding='utf-8') as f:
        f.write(report_string)

    df = links_df.groupby('url')['type'].value_counts().unstack(fill_value=0)
    
    for col_name in master_type_list:
        if col_name not in df.columns:
            df[col_name] = 0
            
    df['total_count'] = df[master_type_list].sum(axis=1)
    df = df.reset_index()
    
    ordered_cols = ['url', 'total_count'] + master_type_list
    df = df[ordered_cols]
    
    df.to_csv('link_visibility_data.csv', index=False, encoding='utf-8')
    
    return report_string, df

if __name__ == "__main__":
    report, dataframe = definitive_parse_and_analyze()
    
    if dataframe is not None:
        print("✅ Analysis Complete!")
        print(f"✅ Report saved to 'link_visibility_report.txt'")
        print(f"✅ Data saved to 'link_visibility_data.csv'")
        print(f"✅ Charts saved to 'link_type_distribution.png', 'link_type_summary_pie.png', and 'top_domains_by_link_type.png'")
        print("\n--- Report Summary ---")
        print(report)
        print("\n--- DataFrame Head ---")
        print(dataframe.head())
    else:
        print("Analysis failed. Please check the error message.")