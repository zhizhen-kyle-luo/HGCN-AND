import os
import json
import argparse
import xml.etree.ElementTree as ET
from nameparser import HumanName
import requests
from collections import defaultdict
import sys

def ensure_directory(path):
    # ensure directory exists
    os.makedirs(path, exist_ok=True)

def fetch_author_data(author_name, max_results=200):
    # fetch author data from openalex api
    # author_name: name of the author to disambiguate
    # max_results: max number of results to return
    # returns: dict with author ids as keys and author data as values
    print(f"Fetching author data for {author_name}...")
    cursor = "*"
    authors_data = {}
    result_count = 0
    
    while True and result_count < max_results:
        query_url = f'https://api.openalex.org/authors?search={author_name}&per_page=100&cursor={cursor}'
        
        try:
            response = requests.get(query_url)
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code}")
                break
                
            data = response.json()
            authors = data["results"]
            
            if not authors:
                break

            for author in authors:
                name = HumanName(author.get("display_name", ""))
                
                # skip if name doesnt match
                first_name = name.first.lower()
                last_name = name.last.lower()
                target_name_parts = author_name.lower().split()

                query_first = ""
                query_last = ""

                if len(target_name_parts) > 0:
                    query_first = target_name_parts[0]

                if len(target_name_parts) > 1: 
                    query_last = target_name_parts[-1]
                elif len(target_name_parts) == 1: 
                    pass

                candidate_first_normalized = name.first.lower()
                candidate_last_normalized = name.last.lower()
                
                match = False
                if query_first and query_last:
                    if candidate_first_normalized == query_first and candidate_last_normalized == query_last:
                        match = True
                elif query_first:
                    if candidate_first_normalized == query_first:
                        match = True 
                elif query_last:
                    if candidate_last_normalized == query_last:
                        match = True
                
                if not match:
                    continue
                
                author_id = author["id"].replace("https://openalex.org/", "")
                authors_data[author_id] = {
                    "id": author_id,
                    "name": author.get("display_name", ""),
                    "name_first": name.first,
                    "name_middle": name.middle,
                    "name_last": name.last,
                    "works_count": author.get("works_count", 0),
                    "works": []
                }
                
                result_count += 1
                if result_count >= max_results:
                    break
            
            # update cursor for next page
            cursor = data["meta"].get("next_cursor")
            if not cursor:
                break
                
        except Exception as e:
            print(f"Error fetching author data: {e}")
            break
    
    print(f"Found {len(authors_data)} authors matching {author_name}")
    return authors_data

def fetch_works_for_author(author_id, max_works=100):
    # fetch works for a specific author id from openalex api
    print(f"Fetching works for author ID {author_id}...")
    cursor = "*"
    works = []
    fetched_count = 0
    
    while True and fetched_count < max_works:
        query_url = f'https://api.openalex.org/works?filter=author.id:{author_id}&per_page=100&cursor={cursor}'
        
        try:
            response = requests.get(query_url)
            if response.status_code != 200:
                print(f"Error fetching works: {response.status_code}")
                break
                
            data = response.json()
            batch_works = data["results"]
            
            if not batch_works:
                break
                
            for work in batch_works:
                # extract needed fields
                work_id = work["id"].replace("https://openalex.org/", "")
                
                # get authors
                authors = []
                for authorship in work.get("authorships", []):
                    if "author" in authorship:
                        author_name = authorship["author"].get("display_name", "")
                        author_id = authorship["author"]["id"].replace("https://openalex.org/", "")
                        authors.append({"name": author_name, "id": author_id})
                
                # get venue
                venue_name = ""
                if "primary_location" in work and work["primary_location"] and "source" in work["primary_location"]:
                    venue_name = work["primary_location"]["source"].get("display_name", "")
                
                # create work entry
                work_entry = {
                    "id": work_id,
                    "title": work.get("title", ""),  # openalex may return none for title
                    "year": work.get("publication_year", 0),
                    "authors": authors,
                    "venue": venue_name
                }
                
                # ensure title is never none
                if not work_entry["title"]:
                    work_entry["title"] = "Untitled publication"
                
                works.append(work_entry)
                fetched_count += 1
                
                if fetched_count >= max_works:
                    break
            
            # update cursor for next page
            cursor = data["meta"].get("next_cursor")
            if not cursor:
                break
                
        except Exception as e:
            print(f"Error fetching works: {e}")
            break
    
    print(f"Found {len(works)} works for author ID {author_id}")
    return works

def create_xml_file(author_name, author_data, works_data, author_id_to_label=None):
    # create xml file in the format expected by hgcn name disambiguation
    print(f"Creating XML file for {author_name}...")
    
    # create mapping from author id to label if not provided
    if author_id_to_label is None:
        author_id_to_label = {}
        for i, author_id in enumerate(author_data.keys()):
            author_id_to_label[author_id] = str(i)
    
    # helper function to escape xml special characters
    def escape_xml(text):
        if text is None:
            return ""
        # replace xml special characters
        text = str(text)
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace("\"", "&quot;")
        text = text.replace("'", "&apos;")
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        return text
    
    xml_str = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml_str += '<person>\n'
    
    # use the first author id as the personid
    first_author_id = next(iter(author_data.keys()))
    xml_str += f'\t<personID>{escape_xml(first_author_id)}</personID>\n'
    
    xml_str += f'\t<FullName>{escape_xml(author_name)}</FullName>\n'
    xml_str += f'\t<FirstName>{escape_xml(author_name.split()[0])}</FirstName>\n'
    xml_str += f'\t<LastName>{escape_xml(author_name.split()[-1])}</LastName>\n'
    
    unique_works = {}
    
    for author_id, author in author_data.items():
        for work_id in author["works"]:
            if work_id in works_data and work_id not in unique_works:
                work = works_data[work_id]
                unique_works[work_id] = work
                
                # ensure title is never none or empty
                title_text = work["title"] if work["title"] else "Untitled publication"
                
                # add publication element
                xml_str += '\t<publication>\n'
                xml_str += f'\t\t<title>{escape_xml(title_text)}</title>\n'
                xml_str += f'\t\t<year>{escape_xml(work["year"])}</year>\n'
                
                # join authors
                authors_text = ", ".join([a["name"] for a in work["authors"]])
                xml_str += f'\t\t<authors>{escape_xml(authors_text)}</authors>\n'
                
                # add venue
                venue_text = work["venue"] if work["venue"] else "Unknown"
                xml_str += f'\t\t<jconf>{escape_xml(venue_text)}</jconf>\n'
                
                # add id
                xml_str += f'\t\t<id>{escape_xml(work_id)}</id>\n'
                
                # add label (which author id this publication belongs to)
                # we'll map each unique author id to a unique integer for the label
                xml_str += f'\t\t<label>{escape_xml(author_id_to_label.get(author_id, "0"))}</label>\n'
                
                # add organization (use openalex author institution if available, otherwise "null")
                xml_str += f'\t\t<organization>{escape_xml("null")}</organization>\n'
                
                xml_str += '\t</publication>\n'
    
    xml_str += '</person>'
    
    # create directory if it doesn't exist
    ensure_directory("raw-data-temp")
    
    # write xml file
    file_path = os.path.join("raw-data-temp", f"{author_name}.xml")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"XML file created: {file_path}")
    return unique_works

def create_author_pair_file(author_name, works_data):
    # create author pair file in the format expected by hgcn name disambiguation
    print(f"Creating author pair file for {author_name}...")
    
    # map from publication id to index
    pub_to_idx = {}
    for i, (pub_id, _) in enumerate(works_data.items()):
        pub_to_idx[pub_id] = i
    
    # track co-author relationships
    co_author_pairs = []
    
    # for each publication
    for pub_id, pub in works_data.items():
        pub_idx = pub_to_idx[pub_id]
        
        # for each author in the publication
        for i in range(len(pub["authors"])):
            for j in range(i+1, len(pub["authors"])):
                author_i = pub["authors"][i]
                author_j = pub["authors"][j]
                
                co_author_pairs.append((pub_idx, pub_idx, author_i["name"], author_j["name"]))
    
    # create directory if it doesn't exist
    ensure_directory(os.path.join("experimental-results", "authors"))
    
    # write author pair file
    file_path = os.path.join("experimental-results", "authors", f"{author_name}_authorlist.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for pair in co_author_pairs:
            f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")
    
    print(f"Author pair file created: {file_path}")

def create_venue_pair_file(author_name, works_data):
    # create venue pair file in the format expected by hgcn name disambiguation
    print(f"Creating venue pair file for {author_name}...")
    
    # map from publication id to index
    pub_to_idx = {}
    for i, (pub_id, _) in enumerate(works_data.items()):
        pub_to_idx[pub_id] = i
    
    # group publications by venue
    venues = defaultdict(list)
    for pub_id, pub in works_data.items():
        venue = pub["venue"]
        venues[venue].append(pub_id)
    
    # create venue pairs
    venue_pairs = []
    for venue, pubs in venues.items():
        for i in range(len(pubs)):
            for j in range(i+1, len(pubs)):
                pub_i = pubs[i]
                pub_j = pubs[j]
                idx_i = pub_to_idx[pub_i]
                idx_j = pub_to_idx[pub_j]
                venue_pairs.append((idx_i, idx_j, venue, venue))
    
    # create directory if it doesn't exist
    ensure_directory("experimental-results")
    
    # write venue pair file
    file_path = os.path.join("experimental-results", f"{author_name}_jconfpair.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for pair in venue_pairs:
            f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")
    
    print(f"Venue pair file created: {file_path}")

def save_data_to_json(author_name, author_data, works_data, author_id_to_label):
    # save the fetched data to json for future use
    print(f"Saving data for {author_name} to JSON...")
    
    data = {
        "author_name": author_name,
        "author_data": author_data,
        "works_data": works_data,
        "author_id_to_label": author_id_to_label
    }
    
    # create directory if it doesn't exist
    ensure_directory("cache")
    
    # write json file
    file_path = os.path.join("cache", f"{author_name}_data.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to: {file_path}")

def load_data_from_json(author_name):
    # load data from json if available
    file_path = os.path.join("cache", f"{author_name}_data.json")
    
    if os.path.exists(file_path):
        print(f"Loading cached data for {author_name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data["author_data"], data["works_data"], data["author_id_to_label"]
    
    return None, None, None

def fetch_works_only(author_id, author_name, max_works=100):
    # function to fetch works only for a specific author id
    works = fetch_works_for_author(author_id, max_works)
    
    # save to cache
    cache_dir = os.path.join("cache", "works")
    ensure_directory(cache_dir)
    
    file_path = os.path.join(cache_dir, f"{author_name}_{author_id}_works.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(works, f, indent=2)
    
    print(f"Works for author {author_id} saved to: {file_path}")
    return works

def create_files_from_cache(author_name):
    # create xml and pair files from cached data
    # load author data
    author_data = {}
    works_data = {}
    author_id_to_label = {}
    
    # check if we have the main cache file
    main_cache = os.path.join("cache", f"{author_name}_data.json")
    if os.path.exists(main_cache):
        with open(main_cache, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        author_data = data["author_data"]
        works_data = data.get("works_data", {})
        author_id_to_label = data["author_id_to_label"]
    
    # check for individual works cache files
    works_cache_dir = os.path.join("cache", "works")
    if os.path.exists(works_cache_dir):
        for file in os.listdir(works_cache_dir):
            if file.startswith(f"{author_name}_") and file.endswith("_works.json"):
                with open(os.path.join(works_cache_dir, file), 'r', encoding='utf-8') as f:
                    author_works = json.load(f)
                
                # add works to the works_data dictionary
                for work in author_works:
                    works_data[work["id"]] = work
    
    if not author_data or not works_data:
        print(f"No cached data found for {author_name}")
        return False
    
    # create files
    unique_works = create_xml_file(author_name, author_data, works_data, author_id_to_label)
    create_author_pair_file(author_name, unique_works)
    create_venue_pair_file(author_name, unique_works)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract OpenAlex data for HGCN name disambiguation')
    parser.add_argument('--name', type=str, help='Name to disambiguate (e.g., "John Smith")')
    parser.add_argument('--max_authors', type=int, default=30, help='Maximum number of authors to fetch')
    parser.add_argument('--max_works', type=int, default=100, help='Maximum number of works per author')
    parser.add_argument('--use_cache', action='store_true', help='Use cached data if available')
    
    # new arguments for batch processing
    parser.add_argument('--fetch_works_only', action='store_true', help='Only fetch works for a specific author ID')
    parser.add_argument('--author_id', type=str, help='Author ID to fetch works for (use with --fetch_works_only)')
    parser.add_argument('--create_files_only', action='store_true', help='Create XML and pair files from cached data')
    
    args = parser.parse_args()
    
    # check for required arguments
    if args.fetch_works_only:
        if not args.author_id or not args.name:
            print("Error: --author_id and --name are required with --fetch_works_only")
            sys.exit(1)
        fetch_works_only(args.author_id, args.name, args.max_works)
        sys.exit(0)
    
    if args.create_files_only:
        if not args.name:
            print("Error: --name is required with --create_files_only")
            sys.exit(1)
        success = create_files_from_cache(args.name)
        sys.exit(0 if success else 1)
    
    if not args.name:
        print("Error: --name is required")
        sys.exit(1)
    
    # check for cached data
    if args.use_cache:
        author_data, works_data, author_id_to_label = load_data_from_json(args.name)
        if author_data and works_data and author_id_to_label:
            # create files from cached data
            unique_works = create_xml_file(args.name, author_data, works_data, author_id_to_label)
            create_author_pair_file(args.name, unique_works)
            create_venue_pair_file(args.name, unique_works)
            print(f"Data extraction and formatting complete for {args.name} (using cached data)")
            print(f"Found {len(author_data)} authors and {len(unique_works)} unique publications")
            print(f"Run name_disambiguation.py to perform disambiguation")
            sys.exit(0)
    
    # 1. fetch author data from openalex
    author_data = fetch_author_data(args.name, args.max_authors)
    
    # create mapping from author id to label (integer)
    author_id_to_label = {}
    for i, author_id in enumerate(author_data.keys()):
        author_id_to_label[author_id] = str(i)
    
    # 2. fetch works for each author
    works_data = {}
    for author_id, author in author_data.items():
        author_works = fetch_works_for_author(author_id, args.max_works)
        author["works"] = [w["id"] for w in author_works]
        
        # add works to global works data
        for work in author_works:
            works_data[work["id"]] = work
    
    # save data to json for future use
    save_data_to_json(args.name, author_data, works_data, author_id_to_label)
    
    # 3. create xml file
    unique_works = create_xml_file(args.name, author_data, works_data, author_id_to_label)
    
    # 4. create author pair file
    create_author_pair_file(args.name, unique_works)
    
    # 5. create venue pair file
    create_venue_pair_file(args.name, unique_works)
    
    print(f"Data extraction and formatting complete for {args.name}")
    print(f"Found {len(author_data)} authors and {len(unique_works)} unique publications")
    print(f"Run name_disambiguation.py to perform disambiguation")