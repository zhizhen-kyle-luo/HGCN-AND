#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import sys
from tqdm import tqdm
import time
import logging

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_disambiguation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('batch_disambiguation')

def run_command(cmd, verbose=False):
    # run a shell command and return the output
    if verbose:
        logger.info(f"Running command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Command failed with error: {result.stderr}")
        return False, result.stderr
    
    if verbose:
        logger.info(f"Command output: {result.stdout}")
    
    return True, result.stdout

def process_single_name(name, max_authors=30, max_works=100, verbose=False):
    # process a single name through the entire disambiguation workflow
    logger.info(f"Processing name: {name}")
    
    # step 1: fetch author data from openalex and create hgcn input files
    # ensure name is quoted properly for the command line
    quoted_name = json.dumps(name) # use json.dumps for robust quoting
    cmd_fetch = f"python openAlex_to_HGCN.py --name {quoted_name} --max_authors {max_authors} --max_works {max_works}"
    success_fetch, output_fetch = run_command(cmd_fetch, verbose)
    
    if not success_fetch:
        logger.error(f"Failed to fetch data for {name}: {output_fetch}")
        return False
    
    # step 2: run the name disambiguation in openalex mode
    # call name_disambiguation.py directly with --openalex flag
    cmd_disambiguate = f"python name_disambiguation.py --openAlex --name {quoted_name}"
    success_disambiguate, output_disambiguate = run_command(cmd_disambiguate, verbose)
    
    if not success_disambiguate:
        logger.error(f"Failed to run disambiguation for {name}: {output_disambiguate}")
        return False
    else:
        # print final json output from captured stdout
        try:
            json_header = f"\n{name} OpenAlex author ID clusters:"
            start_index = output_disambiguate.find(json_header)
            if start_index != -1:
                # print the header and the json that follows it
                print(output_disambiguate[start_index:].strip())
            else:
                # log if the expected header wasn't found, but command succeeded
                logger.warning(f"Could not find final JSON output in stdout for {name}. Command output was:\n{output_disambiguate}")
        except Exception as e:
            logger.error(f"Error extracting JSON output for {name}: {e}")

    logger.info(f"Successfully processed {name}")
    return True

def process_from_file(filename, max_authors=30, max_works=100, verbose=False):
    # process names from a text file
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        return False
    
    logger.info(f"Processing {len(names)} names from {filename}")
    
    success_count = 0
    with tqdm(total=len(names), desc="Processing names") as pbar:
        for name in names:
            result = process_single_name(name, max_authors, max_works, verbose)
            if result:
                success_count += 1
            pbar.update(1)
            # add a small delay to avoid overwhelming the api
            time.sleep(1)
    
    logger.info(f"Completed processing {len(names)} names. {success_count} successful, {len(names) - success_count} failed.")
    return success_count > 0

def process_from_json(json_file, name_field, max_authors=30, max_works=100, verbose=False):
    # process names from a json file
    if not os.path.exists(json_file):
        logger.error(f"JSON file not found: {json_file}")
        return False
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file}: {e}")
        return False
    
    # extract names from json data
    names = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and name_field in item:
                names.append(item[name_field])
            elif isinstance(item, str):
                names.append(item)
    elif isinstance(data, dict):
        if name_field in data:
            if isinstance(data[name_field], list):
                names = data[name_field]
            else:
                names = [data[name_field]]
    
    if not names:
        logger.error(f"No names found in JSON file using field '{name_field}'")
        return False
    
    logger.info(f"Processing {len(names)} names from JSON file {json_file}")
    
    success_count = 0
    with tqdm(total=len(names), desc="Processing names") as pbar:
        for name in names:
            result = process_single_name(name, max_authors, max_works, verbose)
            if result:
                success_count += 1
            pbar.update(1)
            # add a small delay to avoid overwhelming the api
            time.sleep(1)
    
    logger.info(f"Completed processing {len(names)} names. {success_count} successful, {len(names) - success_count} failed.")
    return success_count > 0

def process_existing_openAlex_data(directory='openAlex_scraper', max_authors=30, max_works=100, verbose=False):
    # process existing data from the openalex scraper directory
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return False
    
    # try to find existing data sources
    sources = []
    
    # check for the json data file
    json_file = os.path.join(directory, 'asci_aap_dataJSON.json')
    if os.path.exists(json_file):
        sources.append(('json', json_file))
    
    # check for openalex ids
    ids_file = os.path.join(directory, 'openAlex_ids.json')
    if os.path.exists(ids_file):
        sources.append(('ids', ids_file))
    
    if not sources:
        logger.error(f"No usable data sources found in {directory}")
        return False
    
    # process each source
    for source_type, source_path in sources:
        logger.info(f"Processing source: {source_path}")
        
        if source_type == 'json':
            # try to identify the name field by examining the file
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # check if it's a list of items
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    # look for common name fields
                    for field in ['name', 'author_name', 'display_name', 'full_name']:
                        if isinstance(sample, dict) and field in sample:
                            logger.info(f"Found name field: {field}")
                            process_from_json(source_path, field, max_authors, max_works, verbose)
                            break
                    else:
                        logger.warning(f"Could not identify name field in {source_path}")
            except Exception as e:
                logger.error(f"Error processing JSON file {source_path}: {e}")
        
        elif source_type == 'ids':
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    ids_data = json.load(f)
                
                # check if this is a dictionary mapping names to ids
                if isinstance(ids_data, dict):
                    names = list(ids_data.keys())
                    logger.info(f"Found {len(names)} names in {source_path}")
                    
                    success_count = 0
                    with tqdm(total=len(names), desc="Processing names") as pbar:
                        for name in names:
                            result = process_single_name(name, max_authors, max_works, verbose)
                            if result:
                                success_count += 1
                            pbar.update(1)
                            # add a small delay to avoid overwhelming the api
                            time.sleep(1)
                    
                    logger.info(f"Completed processing {len(names)} names. {success_count} successful, {len(names) - success_count} failed.")
            except Exception as e:
                logger.error(f"Error processing IDs file {source_path}: {e}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process names for disambiguation')
    
    # input sources
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument('--names', nargs='+', help='List of names to process')
    input_group.add_argument('--file', type=str, help='Text file containing names (one per line)')
    input_group.add_argument('--json', type=str, help='JSON file containing names')
    input_group.add_argument('--json_field', type=str, default='name', help='Field name containing the name in the JSON file')
    input_group.add_argument('--use_existing', action='store_true', help='Use existing data from the OpenAlex scraper directory')
    
    # processing options
    options_group = parser.add_argument_group('Processing Options')
    options_group.add_argument('--max_authors', type=int, default=30, help='Maximum number of authors to fetch per name')
    options_group.add_argument('--max_works', type=int, default=100, help='Maximum number of works per author')
    options_group.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # check if at least one input source is provided
    if not (args.names or args.file or args.json or args.use_existing):
        parser.print_help()
        print("\nError: At least one input source must be provided")
        sys.exit(1)
    
    # process names from command line
    if args.names:
        logger.info(f"Processing {len(args.names)} names from command line")
        for name in args.names:
            process_single_name(name, args.max_authors, args.max_works, args.verbose)
    
    # process names from text file
    if args.file:
        process_from_file(args.file, args.max_authors, args.max_works, args.verbose)
    
    # process names from json file
    if args.json:
        process_from_json(args.json, args.json_field, args.max_authors, args.max_works, args.verbose)
    
    # process existing openalex data
    if args.use_existing:
        process_existing_openAlex_data('openAlex_scraper', args.max_authors, args.max_works, args.verbose)
    
    logger.info("Batch processing complete")
