#!/usr/bin/env python3
"""
Browser Bookmark Management Tool
Manages bookmarks with duplicate removal, link validation, backup, and organization features.
"""

import json
import os
import shutil
import argparse
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date
from tqdm import tqdm
import time
import hashlib

# LLM integration imports
from llm_client import create_llm_client, convert_llm_suggestions_to_internal_format
from llm_templates import LLMConfig

class BookmarkManager:
    def __init__(self, bookmark_file: str):
        self.bookmark_file = bookmark_file
        self.bookmarks = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Cache and state management
        self.cache_file = f"{bookmark_file}.cache.json"
        self.cache = self.load_cache()
        
    def load_cache(self) -> Dict:
        """Load cache from file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'link_validation': {},
            'folder_analysis': {},
            'processing_history': []
        }
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def load_processing_state(self, state_file: str) -> Optional[Dict]:
        """Load processing state from intermediate file"""
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load processing state: {e}")
        return None
    
    def save_processing_state(self, state: Dict, state_file: str):
        """Save processing state to intermediate file"""
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving processing state: {e}")
            raise
        
    def load_bookmarks(self) -> Dict:
        """Load bookmarks from Chrome/Edge bookmark file"""
        with open(self.bookmark_file, 'r', encoding='utf-8') as f:
            self.bookmarks = json.load(f)
        return self.bookmarks
    
    def create_backup(self) -> str:
        """Create a backup of the original bookmark file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{self.bookmark_file}.backup_{timestamp}"
        shutil.copy2(self.bookmark_file, backup_file)
        return backup_file
    
    def extract_all_bookmarks(self, node: Dict, path: str = "") -> List[Dict]:
        """Recursively extract all bookmarks with their folder paths"""
        bookmarks = []
        
        if node.get('type') == 'url':
            bookmark = {
                'name': node['name'],
                'url': node['url'],
                'guid': node.get('guid'),
                'date_added': node.get('date_added'),
                'date_modified': node.get('date_modified'),
                'folder_path': path
            }
            bookmarks.append(bookmark)
        elif node.get('type') == 'folder':
            folder_name = node['name']
            new_path = f"{path}/{folder_name}" if path else folder_name
            
            for child in node.get('children', []):
                bookmarks.extend(self.extract_all_bookmarks(child, new_path))
        
        return bookmarks
    
    def find_duplicates(self) -> List[List[Dict]]:
        """Find duplicate bookmarks based on URL"""
        all_bookmarks = []
        
        # Extract from bookmarks bar
        if 'roots' in self.bookmarks and 'bookmark_bar' in self.bookmarks['roots']:
            bar_bookmarks = self.extract_all_bookmarks(
                self.bookmarks['roots']['bookmark_bar'], 
                "Bookmarks Bar"
            )
            all_bookmarks.extend(bar_bookmarks)
        
        # Extract from other bookmarks
        if 'roots' in self.bookmarks and 'other' in self.bookmarks['roots']:
            other_bookmarks = self.extract_all_bookmarks(
                self.bookmarks['roots']['other'], 
                "Other Bookmarks"
            )
            all_bookmarks.extend(other_bookmarks)
        
        # Group by URL
        url_groups = {}
        for bookmark in all_bookmarks:
            url = bookmark['url'].lower().strip()
            if url not in url_groups:
                url_groups[url] = []
            url_groups[url].append(bookmark)
        
        # Return groups with duplicates
        duplicates = [group for group in url_groups.values() if len(group) > 1]
        return duplicates
    
    def validate_links(self, bookmarks: List[Dict], timeout: int = 5) -> List[Dict]:
        """Test bookmark URLs and mark invalid ones (with caching)"""
        results = []
        cache_hits = 0
        cache_misses = 0
        
        # Get URLs that need validation
        urls_to_validate = []
        for bookmark in bookmarks:
            url = bookmark['url']
            if url in self.cache['link_validation']:
                cache_hits += 1
            else:
                urls_to_validate.append(bookmark)
                cache_misses += 1
        
        if cache_hits > 0:
            print(f"Using cached results for {cache_hits} URLs, validating {cache_misses} new URLs")
        
        # Validate uncached URLs
        if urls_to_validate:
            for bookmark in tqdm(urls_to_validate, desc="Validating links"):
                url = bookmark['url']
                try:
                    response = self.session.head(url, timeout=timeout, allow_redirects=True)
                    validation_result = {
                        'status_code': response.status_code,
                        'is_valid': response.status_code < 400,
                        'final_url': response.url,
                        'validated_at': datetime.now().isoformat()
                    }
                except Exception as e:
                    validation_result = {
                        'status_code': None,
                        'is_valid': False,
                        'error': str(e),
                        'validated_at': datetime.now().isoformat()
                    }
                
                # Cache the result
                self.cache['link_validation'][url] = validation_result
                time.sleep(0.1)  # Be respectful
        
        # Build results using cache
        for bookmark in bookmarks:
            result = bookmark.copy()
            cached = self.cache['link_validation'][bookmark['url']]
            result.update(cached)
            results.append(result)
        
        # Save cache after validation
        if urls_to_validate:
            self.save_cache()
        
        return results
    
    def get_folder_stats(self, bookmarks: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract folder statistics without similarity analysis"""
        folder_stats = {}
        
        for bookmark in bookmarks:
            folder_path = bookmark['folder_path']
            if folder_path not in ['Bookmarks Bar', 'Other Bookmarks', '']:
                if folder_path not in folder_stats:
                    folder_stats[folder_path] = []
                folder_stats[folder_path].append(bookmark)
        
        return folder_stats
    
    def identify_outdated_bookmarks(self, bookmarks: List[Dict], days_threshold: int = 365) -> List[Dict]:
        """Identify potentially outdated bookmarks"""
        outdated = []
        current_time = int(time.time() * 1000000)  # Chrome timestamp format
        threshold_microseconds = days_threshold * 24 * 60 * 60 * 1000000
        
        for bookmark in bookmarks:
            date_added = bookmark.get('date_added', 0)
            date_modified = bookmark.get('date_modified', date_added)
            
            # Handle None values and convert to int
            date_added = int(date_added) if date_added is not None else 0
            date_modified = int(date_modified) if date_modified is not None else date_added
            
            # Use the more recent of the two dates
            last_activity = max(date_added, date_modified)
            
            if (current_time - last_activity) > threshold_microseconds:
                bookmark_copy = bookmark.copy()
                bookmark_copy['days_old'] = (current_time - last_activity) // (24 * 60 * 60 * 1000000)
                outdated.append(bookmark_copy)
        
        return sorted(outdated, key=lambda x: x['days_old'], reverse=True)
    
    def generate_cleanup_plan(self) -> Dict:
        """Generate a comprehensive cleanup plan"""
        all_bookmarks = []
        if 'roots' in self.bookmarks:
            for root_name, root_node in self.bookmarks['roots'].items():
                if root_name == 'bookmark_bar':
                    bookmarks = self.extract_all_bookmarks(root_node, "Bookmarks Bar")
                else:
                    bookmarks = self.extract_all_bookmarks(root_node, root_name.replace('_', ' ').title())
                all_bookmarks.extend(bookmarks)
        
        # Analyze all aspects
        duplicates = self.find_duplicates()
        folder_stats = self.get_folder_stats(all_bookmarks)
        
        # Test all links for broken ones
        print("Testing all links for broken ones...")
        sample_bookmarks = all_bookmarks
        validated_sample = self.validate_links(sample_bookmarks)
        broken_links = [b for b in validated_sample if not b['is_valid']]
        
        outdated = self.identify_outdated_bookmarks(all_bookmarks, 365)
        
        # Count orphaned bookmarks
        orphaned_count = sum(1 for b in all_bookmarks if b['folder_path'] in ['Bookmarks Bar', 'Other Bookmarks', ''])
        
        plan = {
            'total_bookmarks': len(all_bookmarks),
            'duplicates': duplicates,
            'folder_stats': folder_stats,
            'broken_links_sample': broken_links,
            'outdated_bookmarks': outdated[:20],  # Show first 20
            'orphaned_count': orphaned_count,
            'proposed_actions': []
        }
        
        # Generate proposed actions
        if duplicates:
            plan['proposed_actions'].append({
                'action': 'remove_duplicates',
                'description': f'Remove {sum(len(group)-1 for group in duplicates)} duplicate bookmarks',
                'details': f'Found {len(duplicates)} sets of duplicates'
            })
        
        if orphaned_count > 0:
            plan['proposed_actions'].append({
                'action': 'organize_orphaned',
                'description': f'Organize {orphaned_count} orphaned bookmarks into appropriate folders using LLM analysis',
                'details': f'Move bookmarks from Bookmarks Bar and Other Bookmarks into organized folders'
            })
        
        if broken_links:
            estimated_broken = int(len(broken_links) * len(all_bookmarks) / sample_size)
            plan['proposed_actions'].append({
                'action': 'remove_broken_links',
                'description': f'Remove approximately {estimated_broken} broken links',
                'details': f'Based on sample of {sample_size} bookmarks, found {len(broken_links)} broken'
            })
        
        if len(outdated) > 10:
            plan['proposed_actions'].append({
                'action': 'archive_outdated',
                'description': f'Archive {len(outdated)} bookmarks older than 1 year',
                'details': 'Move to an "Archived" folder for review'
            })
        
        return plan
    
    def create_reorganized_bookmarks(self, plan_choices: Dict, folder_mapping: Optional[Dict] = None, alphabetize: bool = False) -> Dict:
        """Create a new bookmark structure based on user choices"""
        new_bookmarks = {
            "checksum": self.bookmarks.get("checksum", ""),
            "roots": {
                "bookmark_bar": {"children": [], "date_added": str(int(time.time() * 1000000)), "date_modified": str(int(time.time() * 1000000)), "id": "1", "name": "Bookmarks bar", "type": "folder"},
                "other": {"children": [], "date_added": str(int(time.time() * 1000000)), "date_modified": str(int(time.time() * 1000000)), "id": "2", "name": "Other bookmarks", "type": "folder"},
                "synced": {"children": [], "date_added": str(int(time.time() * 1000000)), "date_modified": str(int(time.time() * 1000000)), "id": "3", "name": "Mobile bookmarks", "type": "folder"}
            },
            "version": 1
        }
        
        # Extract all bookmarks
        all_bookmarks = []
        if 'roots' in self.bookmarks:
            for root_name, root_node in self.bookmarks['roots'].items():
                if root_name == 'bookmark_bar':
                    bookmarks = self.extract_all_bookmarks(root_node, "Bookmarks Bar")
                else:
                    bookmarks = self.extract_all_bookmarks(root_node, root_name.replace('_', ' ').title())
                all_bookmarks.extend(bookmarks)
        
        # Remove duplicates if chosen
        if plan_choices.get('remove_duplicates', False):
            duplicates = self.find_duplicates()
            urls_to_remove = set()
            for group in duplicates:
                # Keep the first one, mark others for removal
                for bookmark in group[1:]:
                    urls_to_remove.add(bookmark['url'])
            all_bookmarks = [b for b in all_bookmarks if b['url'] not in urls_to_remove]
        
        # Remove broken links if chosen
        if plan_choices.get('remove_broken_links', False):
            print("Validating all links for removal...")
            validated = self.validate_links(all_bookmarks)
            all_bookmarks = [b for b in validated if b['is_valid']]
        
        # Folder reorganization will be handled by LLM analysis in the main create_plan flow
        # This method now focuses on basic bookmark structure creation
        
        # Build new folder structure
        folders_created = {}
        
        def create_nested_folder_structure(path: str) -> Dict:
            """Create nested folder structure for a given path"""
            if path in folders_created:
                return folders_created[path]
            
            # Split path into components
            path_parts = [part.strip() for part in path.replace('\\', '/').split('/') if part.strip()]
            
            if not path_parts:
                return None
            
            # Create or get root folder
            root_folder_name = path_parts[0]
            root_path = root_folder_name
            
            if root_path not in folders_created:
                folders_created[root_path] = {
                    "children": [],
                    "date_added": str(int(time.time() * 1000000)),
                    "date_modified": str(int(time.time() * 1000000)),
                    "id": str(len(folders_created) + 10),
                    "name": root_folder_name,
                    "type": "folder"
                }
            
            current_folder = folders_created[root_path]
            current_path = root_path
            
            # Create nested subfolders
            for i, part in enumerate(path_parts[1:], 1):
                current_path += '/' + part
                
                # Check if this nested folder already exists
                if current_path not in folders_created:
                    subfolder = {
                        "children": [],
                        "date_added": str(int(time.time() * 1000000)),
                        "date_modified": str(int(time.time() * 1000000)),
                        "id": str(len(folders_created) + 10),
                        "name": part,
                        "type": "folder"
                    }
                    
                    # Add to parent's children
                    current_folder["children"].append(subfolder)
                    folders_created[current_path] = subfolder
                    current_folder = subfolder
                else:
                    current_folder = folders_created[current_path]
            
            return folders_created[path]
        
        # Place bookmarks in appropriate folders
        for bookmark in all_bookmarks:
            folder_path = bookmark['folder_path']
            
            # Apply folder mapping if exists
            if folder_path in folder_mapping:
                folder_path = folder_mapping[folder_path]
            
            # Create bookmark node
            bookmark_node = {
                "date_added": bookmark.get('date_added', str(int(time.time() * 1000000))),
                "date_modified": bookmark.get('date_modified', str(int(time.time() * 1000000))),
                "id": str(hash(bookmark['url']) % 1000000),
                "name": bookmark['name'],
                "type": "url",
                "url": bookmark['url']
            }
            
            # Place in appropriate root folder
            if folder_path == 'Bookmarks Bar':
                new_bookmarks['roots']['bookmark_bar']['children'].append(bookmark_node)
            elif folder_path in ['Other Bookmarks', '']:
                new_bookmarks['roots']['other']['children'].append(bookmark_node)
            else:
                # Create nested folder structure and place bookmark
                folder_node = create_nested_folder_structure(folder_path)
                if folder_node:
                    folder_node['children'].append(bookmark_node)
                else:
                    # Fallback to Other Bookmarks if folder creation fails
                    new_bookmarks['roots']['other']['children'].append(bookmark_node)
        
        # Add only root-level folders to other bookmarks (subfolders are already nested)
        root_folders = {}
        for path, folder_node in folders_created.items():
            root_name = path.split('/')[0] if '/' in path else path
            if root_name not in root_folders:
                root_folders[root_name] = folders_created.get(root_name)
        
        for folder_name, folder_node in root_folders.items():
            if folder_node and folder_name not in ['Bookmarks Bar', 'Other Bookmarks', '']:
                new_bookmarks['roots']['other']['children'].append(folder_node)
        
        # Apply alphabetical sorting if requested
        if alphabetize:
            self._alphabetize_folder_structure(new_bookmarks)
        
        return new_bookmarks
    
    def create_bookmarks_from_complete_structure(self, folder_structure: Dict, all_bookmarks: List[Dict], alphabetize: bool = False) -> Dict:
        """Create bookmarks file from complete LLM-provided folder structure"""
        # Create GUID to bookmark mapping for fast lookup
        guid_to_bookmark = {}
        for bookmark in all_bookmarks:
            guid = bookmark.get('guid')
            if guid:
                guid_to_bookmark[guid] = bookmark
        
        # Track which GUIDs have been processed by the LLM
        processed_guids = set()
        
        # Create new bookmark structure
        new_bookmarks = {
            "checksum": self.bookmarks.get("checksum", ""),
            "roots": {
                "bookmark_bar": {
                    "children": [], 
                    "date_added": str(int(time.time() * 1000000)), 
                    "date_modified": str(int(time.time() * 1000000)), 
                    "id": "1", 
                    "name": "Bookmarks bar", 
                    "type": "folder"
                },
                "other": {
                    "children": [], 
                    "date_added": str(int(time.time() * 1000000)), 
                    "date_modified": str(int(time.time() * 1000000)), 
                    "id": "2", 
                    "name": "Other bookmarks", 
                    "type": "folder"
                },
                "synced": {
                    "children": [], 
                    "date_added": str(int(time.time() * 1000000)), 
                    "date_modified": str(int(time.time() * 1000000)), 
                    "id": "3", 
                    "name": "Mobile bookmarks", 
                    "type": "folder"
                }
            },
            "version": 1
        }
        
        # Counter for generating unique IDs
        next_id = 10
        
        def create_folder_structure(structure: Dict, parent_children: List, path: str = "") -> None:
            """Recursively create folder structure and place bookmarks"""
            nonlocal next_id
            
            for folder_name, folder_data in structure.items():
                current_path = f"{path}/{folder_name}" if path else folder_name
                
                # Create folder node
                folder_node = {
                    "children": [],
                    "date_added": str(int(time.time() * 1000000)),
                    "date_modified": str(int(time.time() * 1000000)),
                    "id": str(next_id),
                    "name": folder_name,
                    "type": "folder"
                }
                next_id += 1
                
                # Add bookmarks to this folder if any
                if isinstance(folder_data, dict) and 'bookmark_guids' in folder_data:
                    for guid in folder_data['bookmark_guids']:
                        if guid in guid_to_bookmark:
                            bookmark = guid_to_bookmark[guid]
                            bookmark_node = {
                                "date_added": bookmark.get('date_added', str(int(time.time() * 1000000))),
                                "date_modified": bookmark.get('date_modified', str(int(time.time() * 1000000))),
                                "id": str(next_id),
                                "name": bookmark['name'],
                                "type": "url",
                                "url": bookmark['url']
                            }
                            next_id += 1
                            folder_node["children"].append(bookmark_node)
                            processed_guids.add(guid)  # Track that this GUID was processed
                        else:
                            print(f"⚠️ Warning: GUID {guid} not found in bookmark collection")
                
                # Recursively create subfolders
                if isinstance(folder_data, dict):
                    for key, value in folder_data.items():
                        if key != 'bookmark_guids' and isinstance(value, dict):
                            create_folder_structure({key: value}, folder_node["children"], current_path)
                
                parent_children.append(folder_node)
        
        # Build folder structure under "Other Bookmarks"
        create_folder_structure(folder_structure, new_bookmarks['roots']['other']['children'])
        
        # Preserve bookmarks that weren't included in the LLM response
        unprocessed_bookmarks = []
        for bookmark in all_bookmarks:
            guid = bookmark.get('guid')
            # Include bookmarks without GUIDs or with GUIDs not processed by LLM
            if not guid or (guid and guid not in processed_guids):
                unprocessed_bookmarks.append(bookmark)
        
        if unprocessed_bookmarks:
            print(f"📌 Preserving {len(unprocessed_bookmarks)} bookmarks not included in LLM response in original structure")
            print(f"\n📋 Details of preserved bookmarks:")
            
            # Group by folder for cleaner output
            folder_groups = {}
            for bookmark in unprocessed_bookmarks:
                folder_path = bookmark.get('folder_path', 'Other Bookmarks')
                if folder_path not in folder_groups:
                    folder_groups[folder_path] = []
                folder_groups[folder_path].append(bookmark)
            
            # Print details for each folder
            for folder_path, bookmarks_in_folder in sorted(folder_groups.items()):
                print(f"\n  📁 {folder_path} ({len(bookmarks_in_folder)} bookmarks):")
                for bookmark in bookmarks_in_folder:
                    name = bookmark.get('name', 'Unnamed')
                    url = bookmark.get('url', 'No URL')
                    guid = bookmark.get('guid', 'No GUID')
                    # Truncate long names and URLs for readability
                    display_name = name[:60] + "..." if len(name) > 60 else name
                    display_url = url[:80] + "..." if len(url) > 80 else url
                    print(f"    • {display_name}")
                    print(f"      URL: {display_url}")
                    if guid != 'No GUID':
                        print(f"      GUID: {guid}")
                    print()
            
            print(f"{'='*60}")
            
            # Group unprocessed bookmarks by their original folder path
            unprocessed_folder_structure = {}
            for bookmark in unprocessed_bookmarks:
                folder_path = bookmark.get('folder_path', 'Other Bookmarks')
                if folder_path not in unprocessed_folder_structure:
                    unprocessed_folder_structure[folder_path] = []
                unprocessed_folder_structure[folder_path].append(bookmark)
            
            # Recreate the exact original folder structure for unprocessed bookmarks
            def create_preserved_folder_structure(path: str) -> Dict:
                """Create nested folder structure preserving exact original paths"""
                nonlocal next_id
                if not path or path in ['Bookmarks Bar', 'Other Bookmarks', '']:
                    return None
                
                # Split path into components (handle both / and \ separators)
                path_parts = [part.strip() for part in path.replace('\\', '/').split('/') if part.strip()]
                if not path_parts:
                    return None
                
                # Find existing folder in new structure or create it
                current_children = new_bookmarks['roots']['other']['children']
                current_path = ""
                
                for i, part in enumerate(path_parts):
                    current_path = f"{current_path}/{part}" if current_path else part
                    
                    # Look for existing folder with this name
                    existing_folder = None
                    for child in current_children:
                        if child.get('type') == 'folder' and child.get('name') == part:
                            existing_folder = child
                            break
                    
                    if existing_folder:
                        current_children = existing_folder['children']
                    else:
                        # Create new folder with original name (no modifications)
                        new_folder = {
                            "children": [],
                            "date_added": str(int(time.time() * 1000000)),
                            "date_modified": str(int(time.time() * 1000000)),
                            "id": str(next_id),
                            "name": part,  # Original name, no "Preserved" prefix
                            "type": "folder"
                        }
                        next_id += 1
                        current_children.append(new_folder)
                        current_children = new_folder['children']
                
                # Return the final folder's children list where bookmarks should go
                return current_children
            
            # Add unprocessed bookmarks to their exact original locations
            for folder_path, bookmarks_in_folder in unprocessed_folder_structure.items():
                if folder_path == 'Bookmarks Bar':
                    target_children = new_bookmarks['roots']['bookmark_bar']['children']
                elif folder_path in ['Other Bookmarks', '']:
                    target_children = new_bookmarks['roots']['other']['children']
                else:
                    target_children = create_preserved_folder_structure(folder_path)
                
                # Add bookmarks to target location
                if target_children is not None:
                    for bookmark in bookmarks_in_folder:
                        bookmark_node = {
                            "date_added": bookmark.get('date_added', str(int(time.time() * 1000000))),
                            "date_modified": bookmark.get('date_modified', str(int(time.time() * 1000000))),
                            "id": str(next_id),
                            "name": bookmark['name'],
                            "type": "url",
                            "url": bookmark['url']
                        }
                        next_id += 1
                        target_children.append(bookmark_node)
        
        # Apply alphabetical sorting if requested
        if alphabetize:
            self._alphabetize_folder_structure(new_bookmarks)
        
        return new_bookmarks
    
    def _alphabetize_folder_structure(self, bookmarks_data: Dict):
        """Recursively sort folders and bookmarks alphabetically"""
        def sort_children(children_list: List[Dict]):
            """Sort a list of bookmark/folder children alphabetically"""
            # Separate folders and bookmarks
            folders = [child for child in children_list if child.get('type') == 'folder']
            bookmarks = [child for child in children_list if child.get('type') == 'url']
            
            # Sort each type alphabetically by name (case-insensitive)
            folders.sort(key=lambda x: x.get('name', '').lower())
            bookmarks.sort(key=lambda x: x.get('name', '').lower())
            
            # Recursively sort folder contents
            for folder in folders:
                if 'children' in folder and isinstance(folder['children'], list):
                    sort_children(folder['children'])
            
            # Update the children list: folders first, then bookmarks (both sorted)
            children_list.clear()
            children_list.extend(folders)
            children_list.extend(bookmarks)
        
        # Sort each root section
        for root_name in ['bookmark_bar', 'other', 'synced']:
            if root_name in bookmarks_data['roots'] and 'children' in bookmarks_data['roots'][root_name]:
                sort_children(bookmarks_data['roots'][root_name]['children'])
    
    def load_llm_json_from_file(self, json_file: str) -> Dict:
        """Load and validate LLM JSON output from file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
            
            print(f"📋 Loaded LLM JSON from: {json_file}")
            
            # Convert to internal format
            converted_result = convert_llm_suggestions_to_internal_format(llm_data)
            
            # Validate format type
            format_type = converted_result.get('format_type', 'unknown')
            print(f"🔍 Detected format type: {format_type}")
            
            if format_type == 'complete_restructure':
                folder_structure = converted_result.get('folder_structure', {})
                folder_count = self._count_folders_recursive(folder_structure)
                guid_count = len(self._extract_all_guids_from_structure(folder_structure))
                print(f"📁 Structure contains {folder_count} folders with {guid_count} bookmark assignments")
                
            elif format_type == 'incremental_suggestions':
                consolidations = len(converted_result.get('consolidation_suggestions', []))
                hierarchies = len(converted_result.get('new_hierarchies', []))
                print(f"📋 Contains {consolidations} consolidation suggestions and {hierarchies} hierarchy suggestions")
            
            else:
                raise ValueError(f"Unknown or invalid LLM format type: {format_type}")
            
            return converted_result
            
        except FileNotFoundError:
            raise FileNotFoundError(f"LLM JSON file not found: {json_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {json_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading LLM JSON from {json_file}: {e}")
    
    def _count_folders_recursive(self, structure: Dict) -> int:
        """Recursively count folders in structure"""
        count = 0
        for folder_name, folder_data in structure.items():
            count += 1  # Count this folder
            if isinstance(folder_data, dict):
                for key, value in folder_data.items():
                    if key != 'bookmark_guids' and isinstance(value, dict):
                        count += self._count_folders_recursive({key: value})
        return count
    
    def _extract_all_guids_from_structure(self, folder_structure: Dict) -> List[str]:
        """Extract all bookmark GUIDs from folder structure"""
        all_guids = []
        
        def extract_guids_recursive(structure: Dict):
            for folder_name, folder_data in structure.items():
                if isinstance(folder_data, dict):
                    # Check for bookmark_guids at this level
                    if 'bookmark_guids' in folder_data and isinstance(folder_data['bookmark_guids'], list):
                        all_guids.extend(folder_data['bookmark_guids'])
                    
                    # Recursively check subfolders
                    for key, value in folder_data.items():
                        if key != 'bookmark_guids' and isinstance(value, dict):
                            extract_guids_recursive({key: value})
        
        extract_guids_recursive(folder_structure)
        return all_guids
    
    def save_bookmarks(self, new_bookmarks: Dict, output_file: str):
        """Save the reorganized bookmarks to a new file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_bookmarks, f, indent=2, ensure_ascii=False)
    
    def process_duplicates_phase(self, output_file: str) -> Dict:
        """Phase 1: Remove duplicates and save intermediate result"""
        print("🔍 Phase 1: Processing duplicates...")
        
        # Extract all bookmarks
        all_bookmarks = []
        if 'roots' in self.bookmarks:
            for root_name, root_node in self.bookmarks['roots'].items():
                if root_name == 'bookmark_bar':
                    bookmarks = self.extract_all_bookmarks(root_node, "Bookmarks Bar")
                else:
                    bookmarks = self.extract_all_bookmarks(root_node, root_name.replace('_', ' ').title())
                all_bookmarks.extend(bookmarks)
        
        # Find and remove duplicates
        duplicates = self.find_duplicates()
        urls_to_remove = set()
        
        if duplicates:
            print(f"Found {len(duplicates)} sets of duplicates")
            for group in duplicates:
                # Keep the first one, mark others for removal
                for bookmark in group[1:]:
                    urls_to_remove.add(bookmark['url'])
            
            deduplicated_bookmarks = [b for b in all_bookmarks if b['url'] not in urls_to_remove]
            print(f"Removed {len(all_bookmarks) - len(deduplicated_bookmarks)} duplicate bookmarks")
        else:
            deduplicated_bookmarks = all_bookmarks
            print("No duplicates found")
        
        # Preserve folder context in state
        folder_stats = {}
        for bookmark in deduplicated_bookmarks:
            folder_path = bookmark['folder_path']
            if folder_path not in ['Bookmarks Bar', 'Other Bookmarks', '']:
                if folder_path not in folder_stats:
                    folder_stats[folder_path] = []
                folder_stats[folder_path].append(bookmark)
        
        # Create intermediate state with folder context preserved
        state = {
            'bookmarks': deduplicated_bookmarks,
            'folder_stats': folder_stats,  # Preserve original folder categorization
            'original_count': len(all_bookmarks),
            'processing_state': {
                'duplicates_removed': True,
                'links_validated': False,
                'folders_reorganized': False
            },
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'original_file': self.bookmark_file,
                'duplicates_removed': len(all_bookmarks) - len(deduplicated_bookmarks),
                'folder_count': len(folder_stats)
            }
        }
        
        # Save intermediate result
        self.save_processing_state(state, output_file)
        return state
    
    def ask_resume_or_restart(self, state_file: str, original_bookmark_count: int) -> bool:
        """Ask user if they want to resume from existing state file"""
        state = self.load_processing_state(state_file)
        if not state:
            return False
        
        print(f"\n📁 Found existing processing state: {state_file}")
        print(f"Progress:")
        proc_state = state.get('processing_state', {'duplicates_removed': False, 'links_validated': False, 'folders_reorganized': False})
        print(f"  ✅ Duplicates removed: {proc_state['duplicates_removed']}")
        print(f"  {'✅' if proc_state['links_validated'] else '⏳'} Links validated: {proc_state['links_validated']}")
        print(f"  {'✅' if proc_state['folders_reorganized'] else '⏳'} Folders reorganized: {proc_state['folders_reorganized']}")
        
        metadata = state.get('metadata', {})
        if 'duplicates_removed' in metadata:
            print(f"  📊 {metadata['duplicates_removed']} duplicates removed")
        print(f"  📊 Current bookmark count: {len(state.get('bookmarks', []))}")
        
        while True:
            response = input("\nResume from this state? (y)es / (n)o - restart from original: ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def enhanced_folder_similarity_with_llm(self, folder_stats: Dict[str, List[Dict]], output_raw_response: bool = False, raw_output_file: str = None, debug_prompt: bool = False) -> Dict:
        """Use LLM to analyze and suggest folder organization improvements"""
        
        llm_client = create_llm_client(output_raw_response=output_raw_response, raw_output_file=raw_output_file, debug_prompt=debug_prompt)
        
        print("🤖 Analyzing folder relationships with LLM...")
        llm_result = llm_client.analyze_folders(folder_stats)
        
        # Handle raw output mode (llm_result will be None)
        if llm_result is None:
            if output_raw_response:
                print("✅ Raw response output completed")
                return None  # Return None for raw output mode
            else:
                print("❌ LLM analysis failed or returned no results")
                return {}
        
        print("✅ LLM analysis completed successfully")
        
        # Display the LLM response
        self.display_llm_response(llm_result)
        
        return convert_llm_suggestions_to_internal_format(llm_result)
    
    def output_llm_prompts(self, folder_stats: Dict[str, List[Dict]], output_file: Optional[str] = None):
        """Generate and output LLM prompts for experimentation"""
        try:
            from llm_client import LLMClient
            from llm_templates import LLMConfig, prepare_folder_structure_for_llm, get_template
        except ImportError:
            print("❌ Error: LLM dependencies not available. Please install required packages.")
            return
        
        print(f"🔍 Generating LLM prompts for {len(folder_stats)} folders...")
        
        # Set up default config for prompt generation (doesn't need API keys)
        config = LLMConfig(
            provider="openai",  # Default, but won't be used for API calls
            model="gpt-4.1",      # Default, but won't be used for API calls
            custom_template=os.environ.get('BOOKMARK_LLM_TEMPLATE')
        )
        
        try:
            # Single prompt for all folders (no batching)
            print(f"\n{'='*60}")
            print(f"LLM PROMPT ({len(folder_stats)} folders)")
            print(f"{'='*60}")
            self._output_single_prompt(folder_stats, config)
            
            if output_file:
                self._save_prompt_to_file(folder_stats, config, output_file)
                print(f"💾 Prompt saved to: {output_file}")
        
        except Exception as e:
            print(f"❌ Error generating prompts: {e}")
            import traceback
            print(f"Details: {traceback.format_exc()}")
    
    def _output_single_prompt(self, folder_stats: Dict[str, List[Dict]], config: LLMConfig, chunk_name: str = ""):
        """Generate and display a single prompt"""
        try:
            from llm_templates import prepare_folder_structure_for_llm, get_template
            
            # Prepare structured data for LLM
            folder_structure = prepare_folder_structure_for_llm(folder_stats)
            template = get_template(config)
            
            # Use safe string replacement to avoid format conflicts
            prompt = template.replace('{folder_structure}', folder_structure)
            
            print(f"Prompt length: {len(prompt)} characters")
            print(f"\n{'-'*60}")
            print("FULL PROMPT:")
            print(f"{'-'*60}")
            print(prompt)
            print(f"{'-'*60}")
            
        except FileNotFoundError as e:
            print(f"❌ Template file not found: {e}")
            print("Please ensure llm_template.txt exists or provide a custom template")
        except Exception as e:
            print(f"❌ Error generating prompt: {e}")
    
    def _save_prompt_to_file(self, folder_stats: Dict[str, List[Dict]], config: LLMConfig, output_file: str, chunk_name: str = ""):
        """Save prompt to file"""
        try:
            from llm_templates import prepare_folder_structure_for_llm, get_template
            
            # Prepare structured data for LLM
            folder_structure = prepare_folder_structure_for_llm(folder_stats)
            template = get_template(config)
            
            # Use safe string replacement to avoid format conflicts
            prompt = template.replace('{folder_structure}', folder_structure)
            
            # Create output with metadata
            output_content = f"""# LLM Prompt for Bookmark Folder Analysis
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Prompt length: {len(prompt)} characters
{f'# Chunk: {chunk_name}' if chunk_name else ''}

{'-'*60}
FULL PROMPT:
{'-'*60}
{prompt}
{'-'*60}
"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_content)
                
        except Exception as e:
            print(f"❌ Error saving prompt to file: {e}")
    
    def display_llm_response(self, llm_result: Dict):
        """Display the raw LLM analysis response to user"""
        print(f"\n🤖 LLM ANALYSIS RESPONSE")
        print(f"{'='*50}")
        
        consolidations = llm_result.get('consolidations', [])
        renames = llm_result.get('renames', [])
        new_hierarchies = llm_result.get('new_hierarchies', [])
        orphan_suggestions = llm_result.get('orphan_suggestions', [])
        
        if consolidations:
            print(f"\n📁 FOLDER CONSOLIDATIONS ({len(consolidations)}):")
            for i, cons in enumerate(consolidations, 1):
                print(f"{i}. Merge {', '.join(cons['merge_folders'])} → {cons['primary_folder']}")
                print(f"   Confidence: {cons.get('confidence', 0):.1%}")
                print(f"   Reasoning: {cons.get('reasoning', 'No reasoning provided')}")
                print()
        
        if renames:
            print(f"\n✏️ FOLDER RENAMES ({len(renames)}):")
            for i, rename in enumerate(renames, 1):
                print(f"{i}. '{rename['old_name']}' → '{rename['new_name']}'")
                print(f"   Reasoning: {rename.get('reasoning', 'No reasoning provided')}")
                print()
        
        if new_hierarchies:
            print(f"\n🗂️ NEW HIERARCHIES ({len(new_hierarchies)}):")
            for i, hier in enumerate(new_hierarchies, 1):
                if 'parent' in hier:
                    children = ', '.join(hier.get('children', []))
                    print(f"{i}. {hier['parent']}/")
                    print(f"   Subfolders: {children}")
                    print(f"   Reasoning: {hier.get('reasoning', 'No reasoning provided')}")
                    print()
                else:
                    print(f"⚠️ Skipping malformed hierarchy: {hier}")
                    print()
        
        if orphan_suggestions:
            print(f"\n📌 ORPHAN BOOKMARK SUGGESTIONS ({len(orphan_suggestions)}):")
            for i, orphan in enumerate(orphan_suggestions, 1):
                if 'bookmark_pattern' in orphan and 'suggested_folder' in orphan:
                    print(f"{i}. {orphan['bookmark_pattern']} → {orphan['suggested_folder']}")
                    print(f"   Reasoning: {orphan.get('reasoning', 'No reasoning provided')}")
                    print()
                else:
                    print(f"⚠️ Skipping malformed orphan suggestion: {orphan}")
                    print()
        
        print(f"{'='*50}")
    
    def display_complete_restructure(self, llm_result: Dict):
        """Display the complete restructure response from LLM"""
        print(f"\n🤖 LLM COMPLETE RESTRUCTURE RESPONSE")
        print(f"{'='*60}")
        
        folder_structure = llm_result.get('folder_structure', {})
        reasoning = llm_result.get('reasoning', {})
        summary = llm_result.get('summary', {})
        
        if summary:
            print(f"\n📊 SUMMARY:")
            print(f"  Total folders: {summary.get('total_folders', 'Unknown')}")
            print(f"  Total bookmarks: {summary.get('total_bookmarks', 'Unknown')}")
            print(f"  Max depth: {summary.get('max_depth', 'Unknown')}")
            if 'major_categories' in summary:
                print(f"  Major categories: {', '.join(summary['major_categories'])}")
        
        print(f"\n🗂️ PROPOSED FOLDER STRUCTURE:")
        self._display_folder_tree(folder_structure, "", show_guids=False)
        
        if reasoning:
            print(f"\n💭 LLM REASONING:")
            for folder_path, reason in reasoning.items():
                print(f"  {folder_path}: {reason}")
        
        print(f"{'='*60}")
    
    def _display_folder_tree(self, structure: Dict, indent: str = "", show_guids: bool = False):
        """Recursively display folder tree structure"""
        for folder_name, folder_data in structure.items():
            print(f"{indent}📁 {folder_name}/")
            
            if isinstance(folder_data, dict):
                # Show bookmark count if bookmark_guids present
                if 'bookmark_guids' in folder_data:
                    bookmark_count = len(folder_data['bookmark_guids'])
                    print(f"{indent}  📄 {bookmark_count} bookmarks")
                    
                    if show_guids and bookmark_count <= 5:  # Show GUIDs for small folders
                        for guid in folder_data['bookmark_guids'][:5]:
                            print(f"{indent}    • {guid}")
                
                # Recursively display subfolders
                for key, value in folder_data.items():
                    if key != 'bookmark_guids' and isinstance(value, dict):
                        self._display_folder_tree({key: value}, indent + "  ", show_guids)
    
    def preview_folder_structure(self, folder_stats: Dict[str, List[Dict]], llm_suggestions: Dict) -> Dict[str, List[str]]:
        """Generate and display preview of proposed folder structure"""
        print(f"\n🌳 PROPOSED FOLDER STRUCTURE PREVIEW")
        print(f"{'='*50}")
        
        # Start with current folders
        proposed_structure = {}
        folder_mapping = {}
        
        # Apply consolidations
        for suggestion in llm_suggestions.get('consolidation_suggestions', []):
            primary = suggestion['primary_folder']
            for merge_folder in suggestion['merge_folders']:
                folder_mapping[merge_folder] = primary
        
        # Apply renames from LLM
        llm_result = llm_suggestions.get('llm_analysis', {})
        for rename in llm_result.get('renames', []):
            old_name = rename['old_name']
            new_name = rename['new_name']
            folder_mapping[old_name] = new_name
        
        # Build proposed structure
        for folder_name, bookmarks in folder_stats.items():
            # Apply mapping if exists
            final_folder_name = folder_mapping.get(folder_name, folder_name)
            
            if final_folder_name not in proposed_structure:
                proposed_structure[final_folder_name] = []
            
            # Add bookmark info (just counts and samples)
            for bookmark in bookmarks:
                proposed_structure[final_folder_name].append(bookmark['name'])
        
        # Display structure as tree
        def display_folder_tree(structure: Dict[str, List[str]], indent: str = ""):
            for folder_name, bookmark_names in sorted(structure.items()):
                # Handle nested folders
                if '/' in folder_name:
                    continue  # Skip, will be handled by parent
                
                print(f"{indent}📁 {folder_name}/ ({len(bookmark_names)} bookmarks)")
                
                # Show sample bookmarks
                for bookmark_name in bookmark_names[:3]:
                    display_name = bookmark_name[:60] + "..." if len(bookmark_name) > 60 else bookmark_name
                    print(f"{indent}   📄 {display_name}")
                
                if len(bookmark_names) > 3:
                    print(f"{indent}   ... and {len(bookmark_names) - 3} more bookmarks")
                
                # Look for subfolders
                subfolders = {}
                for other_folder, other_bookmarks in structure.items():
                    if other_folder.startswith(folder_name + '/'):
                        subfolder_name = other_folder[len(folder_name) + 1:]
                        if '/' not in subfolder_name:  # Direct child only
                            subfolders[subfolder_name] = other_bookmarks
                
                if subfolders:
                    for subfolder_name, subfolder_bookmarks in sorted(subfolders.items()):
                        print(f"{indent}  📁 {subfolder_name}/ ({len(subfolder_bookmarks)} bookmarks)")
                        for bookmark_name in subfolder_bookmarks[:2]:
                            display_name = bookmark_name[:50] + "..." if len(bookmark_name) > 50 else bookmark_name
                            print(f"{indent}     📄 {display_name}")
                        if len(subfolder_bookmarks) > 2:
                            print(f"{indent}     ... and {len(subfolder_bookmarks) - 2} more")
                
                print()
        
        # Show current vs proposed
        print(f"\n📊 CURRENT STRUCTURE ({len(folder_stats)} folders):")
        current_structure = {name: [b['name'] for b in bookmarks] for name, bookmarks in folder_stats.items()}
        display_folder_tree(current_structure)
        
        print(f"\n🎯 PROPOSED STRUCTURE ({len(proposed_structure)} folders):")
        display_folder_tree(proposed_structure)
        
        # Show changes summary
        print(f"\n📈 CHANGES SUMMARY:")
        original_folders = set(folder_stats.keys())
        proposed_folders = set(proposed_structure.keys())
        
        removed_folders = original_folders - proposed_folders
        new_folders = proposed_folders - original_folders
        
        if removed_folders:
            print(f"❌ Folders to be merged/removed: {', '.join(sorted(removed_folders))}")
        if new_folders:
            print(f"✅ New/renamed folders: {', '.join(sorted(new_folders))}")
        
        consolidations = len(llm_suggestions.get('consolidation_suggestions', []))
        renames = len(llm_result.get('renames', []))
        print(f"🔄 Total consolidations: {consolidations}")
        print(f"✏️ Total renames: {renames}")
        
        return proposed_structure
    
    def remove_duplicates_interactive(self, duplicates: List[List[Dict]]):
        """Interactively remove duplicates"""
        print(f"\nFound {len(duplicates)} sets of duplicate bookmarks:")
        
        for i, group in enumerate(duplicates, 1):
            print(f"\n--- Duplicate Set {i} ---")
            print(f"URL: {group[0]['url']}")
            
            for j, bookmark in enumerate(group):
                print(f"  {j+1}. {bookmark['name']} (in {bookmark['folder_path']})")
            
            while True:
                try:
                    choice = input(f"Keep which bookmark? (1-{len(group)}, 'a' for all, 's' to skip): ").strip().lower()
                    if choice == 's':
                        break
                    elif choice == 'a':
                        break
                    else:
                        keep_idx = int(choice) - 1
                        if 0 <= keep_idx < len(group):
                            # Mark others for removal
                            for k, bookmark in enumerate(group):
                                if k != keep_idx:
                                    bookmark['_marked_for_removal'] = True
                            break
                        else:
                            print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number, 'a', or 's'.")

def main():
    parser = argparse.ArgumentParser(description="Browser Bookmark Management Tool")
    parser.add_argument("bookmark_file", help="Path to browser bookmark file")
    parser.add_argument("--model", help="LLM model to use (e.g., gpt-4.1, claude-3-sonnet-20240229)")
    parser.add_argument("--output-prompts", action="store_true", help="Output generated prompt only (for experimentation)")
    parser.add_argument("--output-llm-response", action="store_true", help="Output raw LLM response for debugging")
    parser.add_argument("--debug-prompt", action="store_true", help="Save the prompt being sent to LLM for debugging")
    parser.add_argument("--skip-link-check", action="store_true", help="Skip broken link validation")
    parser.add_argument("--skip-duplicate-check", action="store_true", help="Skip duplicate bookmark detection")
    parser.add_argument("--no-alphabetize", action="store_true", help="Skip alphabetical sorting of folders and bookmarks")
    parser.add_argument("--output", "-o", help="Output file for reorganized bookmarks")
    
    # Keep essential legacy options for backwards compatibility
    parser.add_argument("--backup", action="store_true", help="Create backup before processing")
    parser.add_argument("--create-plan", action="store_true", help="Generate cleanup plan and execute with user approval")
    parser.add_argument("--apply-llm-json", help="Apply LLM JSON output from file (skips validation/analysis, directly restructures bookmarks)")
    
    # Hidden legacy options (still functional but not shown in help)
    parser.add_argument("--find-duplicates", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate-links", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suggest-folders", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--find-outdated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--days-threshold", type=int, default=365, help=argparse.SUPPRESS)
    parser.add_argument("--llm-provider", choices=["openai", "anthropic", "none"], help=argparse.SUPPRESS)
    parser.add_argument("--llm-model", help=argparse.SUPPRESS)
    parser.add_argument("--llm-template", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bookmark_file):
        print(f"Error: Bookmark file '{args.bookmark_file}' not found.")
        return
    
    # Default behavior: if no specific action is requested, run create-plan
    if not (args.output_prompts or args.apply_llm_json or args.find_duplicates or 
           args.validate_links or args.suggest_folders or args.find_outdated or args.create_plan):
        args.create_plan = True
    
    # Set up LLM configuration from CLI args if provided
    if args.model:
        os.environ['BOOKMARK_LLM_MODEL'] = args.model
    
    # Handle legacy args for backwards compatibility
    if hasattr(args, 'llm_provider') and args.llm_provider:
        os.environ['BOOKMARK_LLM_PROVIDER'] = args.llm_provider
    if hasattr(args, 'llm_model') and args.llm_model:
        os.environ['BOOKMARK_LLM_MODEL'] = args.llm_model
    if hasattr(args, 'llm_template') and args.llm_template:
        os.environ['BOOKMARK_LLM_TEMPLATE'] = args.llm_template
    
    manager = BookmarkManager(args.bookmark_file)
    manager.load_bookmarks()
    
    # Handle direct LLM JSON application
    if args.apply_llm_json:
        if not os.path.exists(args.apply_llm_json):
            print(f"Error: LLM JSON file '{args.apply_llm_json}' not found.")
            return
        
        try:
            # Load and validate LLM JSON
            llm_analysis = manager.load_llm_json_from_file(args.apply_llm_json)
            
            # Create backup
            backup_file = manager.create_backup()
            print(f"📄 Backup created: {backup_file}")
            
            # Extract all bookmarks for processing
            all_bookmarks = []
            if 'roots' in manager.bookmarks:
                for root_name, root_node in manager.bookmarks['roots'].items():
                    if root_name == 'bookmark_bar':
                        bookmarks = manager.extract_all_bookmarks(root_node, "Bookmarks Bar")
                    else:
                        bookmarks = manager.extract_all_bookmarks(root_node, root_name.replace('_', ' ').title())
                    all_bookmarks.extend(bookmarks)
            
            print(f"📚 Extracted {len(all_bookmarks)} bookmarks from original file")
            
            # Determine output file
            output_file = args.output
            if not output_file:
                base_name = os.path.splitext(args.bookmark_file)[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{base_name}_restructured_{timestamp}.json"
            
            # Process based on format type
            if llm_analysis.get('format_type') == 'complete_restructure':
                # Display the complete restructure proposal
                manager.display_complete_restructure(llm_analysis['llm_analysis'])
                
                # Ask for approval
                print(f"\n{'='*60}")
                while True:
                    approval = input("Apply this complete folder restructure to your bookmarks? (y/n): ").lower().strip()
                    if approval in ['y', 'yes']:
                        print("✅ Restructure approved - applying changes...")
                        break
                    elif approval in ['n', 'no']:
                        print("❌ Restructure cancelled")
                        return
                    else:
                        print("Please enter 'y' or 'n'")
                
                # Apply the complete structure
                print(f"\n🔧 Applying complete restructure...")
                folder_structure = llm_analysis['folder_structure']
                new_bookmarks = manager.create_bookmarks_from_complete_structure(folder_structure, all_bookmarks, not args.no_alphabetize)
                
                # Save the result
                manager.save_bookmarks(new_bookmarks, output_file)
                
                print(f"\n✅ Complete restructure applied successfully!")
                print(f"📄 New bookmarks saved: {output_file}")
                
                # Summary
                processed_guids = manager._extract_all_guids_from_structure(folder_structure)
                print(f"\n📊 Summary:")
                print(f"  Original bookmarks: {len(all_bookmarks)}")
                print(f"  Restructured bookmarks: {len(processed_guids)}")
                print(f"  New folder structure: {manager._count_folders_recursive(folder_structure)} folders")
                
            elif llm_analysis.get('format_type') == 'incremental_suggestions':
                print(f"\n⚠️ Legacy incremental format detected.")
                print(f"This workflow is designed for complete restructure JSON files.")
                print(f"Use --create-plan for incremental suggestions, or provide a complete restructure JSON.")
                return
                
            else:
                print(f"❌ Error: Unknown LLM format type: {llm_analysis.get('format_type', 'unknown')}")
                return
                
        except Exception as e:
            print(f"❌ Error processing LLM JSON: {e}")
            return
        
        return
    
    if args.backup:
        backup_file = manager.create_backup()
        print(f"Backup created: {backup_file}")
    
    # Extract all bookmarks for processing
    all_bookmarks = []
    if 'roots' in manager.bookmarks:
        for root_name, root_node in manager.bookmarks['roots'].items():
            if root_name == 'bookmark_bar':
                bookmarks = manager.extract_all_bookmarks(root_node, "Bookmarks Bar")
            else:
                bookmarks = manager.extract_all_bookmarks(root_node, root_name.replace('_', ' ').title())
            all_bookmarks.extend(bookmarks)
    
    # Handle output-prompts first - it may need to apply cleanup operations
    if args.output_prompts:
        # Apply cleanup operations based on skip flags (default is to do cleanup)
        cleaned_bookmarks = all_bookmarks.copy()
        
        # Remove duplicates unless explicitly skipped
        if not args.skip_duplicate_check:
            print("🔍 Removing duplicates before generating prompt...")
            duplicates = manager.find_duplicates()
            if duplicates:
                urls_to_remove = set()
                for group in duplicates:
                    # Keep the first one, mark others for removal
                    for bookmark in group[1:]:
                        urls_to_remove.add(bookmark['url'])
                cleaned_bookmarks = [b for b in cleaned_bookmarks if b['url'] not in urls_to_remove]
                print(f"✅ Removed {len(all_bookmarks) - len(cleaned_bookmarks)} duplicate bookmarks")
            else:
                print("✅ No duplicate bookmarks found.")
        else:
            print("⏩ Skipping duplicate check")
        
        # Validate links unless explicitly skipped  
        if not args.skip_link_check:
            print("🔍 Removing invalid links before generating prompt...")
            validated = manager.validate_links(cleaned_bookmarks)
            valid_bookmarks = [b for b in validated if b['is_valid']]
            removed_count = len(cleaned_bookmarks) - len(valid_bookmarks)
            cleaned_bookmarks = valid_bookmarks
            if removed_count > 0:
                print(f"✅ Removed {removed_count} invalid bookmarks")
            else:
                print("✅ All bookmarks are valid!")
        else:
            print("⏩ Skipping link validation")
        
        # Generate folder stats from cleaned data
        folder_stats = manager.get_folder_stats(cleaned_bookmarks)
        
        if not folder_stats:
            print("❌ No organized folders found. Cannot generate prompts.")
            print("This tool works best with bookmarks that are already organized into folders.")
            return
        
        # Set up output file if specified
        output_file = None
        if args.output:
            output_file = args.output
            if not output_file.endswith('.txt'):
                output_file += '.txt'
        
        print(f"\n🤖 Generating LLM prompt with {len(cleaned_bookmarks)} cleaned bookmarks...")
        manager.output_llm_prompts(folder_stats, output_file)
        return
    
    # Handle individual analysis operations (when not generating prompts)
    if args.find_duplicates:
        duplicates = manager.find_duplicates()
        if duplicates:
            print(f"\nFound {len(duplicates)} sets of duplicate bookmarks:")
            for i, group in enumerate(duplicates, 1):
                print(f"\nSet {i}: {group[0]['url']}")
                for bookmark in group:
                    print(f"  - {bookmark['name']} (in {bookmark['folder_path']})")
        else:
            print("No duplicate bookmarks found.")
    
    if args.validate_links:
        print(f"\nValidating {len(all_bookmarks)} bookmarks...")
        validated = manager.validate_links(all_bookmarks)
        invalid = [b for b in validated if not b['is_valid']]
        
        if invalid:
            print(f"\nFound {len(invalid)} invalid bookmarks:")
            for bookmark in invalid:
                error_info = bookmark.get('error', f"HTTP {bookmark.get('status_code')}")
                print(f"  - {bookmark['name']}: {bookmark['url']} ({error_info})")
        else:
            print("All bookmarks are valid!")
    
    if args.suggest_folders:
        folder_stats = manager.get_folder_stats(all_bookmarks)
        orphaned_count = sum(1 for b in all_bookmarks if b['folder_path'] in ['Bookmarks Bar', 'Other Bookmarks', ''])
        
        print("\nFolder Structure Analysis:")
        
        print(f"\nExisting Folders ({len(folder_stats)}):")
        for folder, bookmarks in sorted(folder_stats.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  - {folder}: {len(bookmarks)} bookmarks")
        
        if orphaned_count > 0:
            print(f"\nOrphaned Bookmarks: {orphaned_count}")
            print("Use --create-plan with LLM integration for folder reorganization suggestions.")
    
    if args.create_plan:
        # Determine output file for intermediate processing
        intermediate_file = args.output
        if not intermediate_file:
            base_name = os.path.splitext(args.bookmark_file)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = f"{base_name}_processing_{timestamp}.json"
        
        # Check if we should resume from existing state
        current_state = None
        if os.path.exists(intermediate_file):
            original_bookmark_count = len(manager.extract_all_bookmarks(manager.bookmarks['roots']['bookmark_bar'], "")) + \
                                   len(manager.extract_all_bookmarks(manager.bookmarks['roots'].get('other', {}), ""))
            
            if manager.ask_resume_or_restart(intermediate_file, original_bookmark_count):
                current_state = manager.load_processing_state(intermediate_file)
                print("📂 Resuming from existing state...")
            else:
                print("🔄 Starting fresh processing...")
        
        # Phase 1: Remove duplicates (if not skipped and not already done)
        if not args.skip_duplicate_check and (not current_state or not current_state['processing_state']['duplicates_removed']):
            # Create backup if not already done
            if not args.backup:
                backup_file = manager.create_backup()
                print(f"Backup created: {backup_file}")
            
            current_state = manager.process_duplicates_phase(intermediate_file)
        elif args.skip_duplicate_check:
            print("⏩ Skipping duplicate removal")
            # Create a basic state without duplicate processing
            all_bookmarks = []
            if 'roots' in manager.bookmarks:
                for root_name, root_node in manager.bookmarks['roots'].items():
                    if root_name == 'bookmark_bar':
                        bookmarks = manager.extract_all_bookmarks(root_node, "Bookmarks Bar")
                    else:
                        bookmarks = manager.extract_all_bookmarks(root_node, root_name.replace('_', ' ').title())
                    all_bookmarks.extend(bookmarks)
            
            current_state = {
                'bookmarks': all_bookmarks,
                'original_count': len(all_bookmarks),
                'processing_state': {'duplicates_removed': False, 'links_validated': False, 'folders_reorganized': False},
                'metadata': {'processed_at': datetime.now().isoformat(), 'original_file': args.bookmark_file}
            }
        else:
            print("✅ Phase 1 already completed: Duplicates removed")
        
        # Generate plan based on current state
        bookmarks = current_state['bookmarks']
        print(f"\n🔍 Analyzing {len(bookmarks)} bookmarks for further cleanup...")
        
        # Check for broken links (sample first) unless skipped
        broken_links_sample = []
        if not args.skip_link_check and not current_state['processing_state']['links_validated']:
            sample_size = min(25, len(bookmarks))
            print(f"Testing sample of {sample_size} links...")
            sample_bookmarks = bookmarks[:sample_size]
            validated_sample = manager.validate_links(sample_bookmarks)
            broken_links_sample = [b for b in validated_sample if not b['is_valid']]
        elif args.skip_link_check:
            print("⏩ Skipping link validation")
        
        # Use preserved folder stats or rebuild if not available
        if 'folder_stats' in current_state:
            folder_stats = current_state['folder_stats']
            print(f"📁 Using preserved folder context from Phase 1 ({len(folder_stats)} folders)")
        else:
            # Fallback: rebuild folder stats
            folder_stats = {}
            for bookmark in bookmarks:
                folder_path = bookmark['folder_path']
                if folder_path not in ['Bookmarks Bar', 'Other Bookmarks', '']:
                    if folder_path not in folder_stats:
                        folder_stats[folder_path] = []
                    folder_stats[folder_path].append(bookmark)
        
        # Count orphaned bookmarks  
        orphaned_bookmarks = [b for b in bookmarks if b['folder_path'] in ['Bookmarks Bar', 'Other Bookmarks', '']]
        
        # Enhanced folder analysis with bookmark context
        llm_analysis = {}
        complete_restructure_approved = False
        
        if folder_stats:
            raw_output_file = args.output if args.output_llm_response and args.output else None
            llm_analysis = manager.enhanced_folder_similarity_with_llm(folder_stats, args.output_llm_response, raw_output_file, args.debug_prompt)
            
            # If we're just outputting raw response, exit early
            if args.output_llm_response:
                print("🔍 Raw LLM response output completed")
                return
            
            # Check if this is the new complete restructure format
            if llm_analysis.get('format_type') == 'complete_restructure':
                # Display the complete restructure proposal
                manager.display_complete_restructure(llm_analysis['llm_analysis'])
                
                # Ask for approval of the complete restructure
                print(f"\n{'='*60}")
                while True:
                    approval = input("Do you approve this complete folder restructure? (y/n): ").lower().strip()
                    if approval in ['y', 'yes']:
                        print("✅ Complete restructure approved")
                        complete_restructure_approved = True
                        break
                    elif approval in ['n', 'no']:
                        print("❌ Complete restructure declined")
                        llm_analysis = {'consolidation_suggestions': [], 'folder_analysis': {}}
                        break
                    else:
                        print("Please enter 'y' or 'n'")
            
            # Legacy format handling
            elif llm_analysis.get('llm_analysis') and llm_analysis.get('consolidation_suggestions'):
                proposed_structure = manager.preview_folder_structure(folder_stats, llm_analysis)
                
                # Ask for approval of the incremental reorganization
                print(f"\n{'='*60}")
                while True:
                    approval = input("Do you approve this folder structure reorganization? (y/n): ").lower().strip()
                    if approval in ['y', 'yes']:
                        print("✅ Folder reorganization approved")
                        break
                    elif approval in ['n', 'no']:
                        print("❌ Folder reorganization declined")
                        llm_analysis = {'consolidation_suggestions': [], 'folder_analysis': {}}
                        break
                    else:
                        print("Please enter 'y' or 'n'")
        else:
            llm_analysis = {'consolidation_suggestions': [], 'folder_analysis': {}}
        
        # Present enhanced plan
        print(f"\n📊 ENHANCED BOOKMARK CLEANUP PLAN")
        print(f"{'='*60}")
        print(f"Current status: {len(bookmarks)} bookmarks")
        if current_state['metadata'].get('duplicates_removed', 0) > 0:
            print(f"✅ Already removed {current_state['metadata']['duplicates_removed']} duplicates")
        
        proposed_actions = []
        
        if broken_links_sample:
            estimated_broken = int(len(broken_links_sample) * len(bookmarks) / min(25, len(bookmarks)))
            proposed_actions.append({
                'action': 'validate_and_remove_broken',
                'description': f'Validate all links and remove ~{estimated_broken} broken ones',
                'details': f'Based on sample, found {len(broken_links_sample)}/{min(25, len(bookmarks))} broken links'
            })
        
        if llm_analysis.get('consolidation_suggestions'):
            proposed_actions.append({
                'action': 'consolidate_semantic_folders',
                'description': f'Consolidate {len(llm_analysis["consolidation_suggestions"])} sets of semantically similar folders',
                'details': llm_analysis['consolidation_suggestions'][:3]
            })
        
        if orphaned_bookmarks:
            proposed_actions.append({
                'action': 'organize_orphaned_enhanced',
                'description': f'Organize {len(orphaned_bookmarks)} orphaned bookmarks using semantic analysis',
                'details': 'Move bookmarks from root folders into semantically appropriate folders'
            })
        
        if not proposed_actions:
            if llm_analysis.get('consolidation_suggestions'):
                print("\n✅ Your bookmarks are well organized!")
                print("LLM found no significant improvements needed.")
            else:
                print("\n❌ Unable to analyze folder structure.")
                print("Please check LLM configuration and try again.")
            return
        
        print(f"\n📋 Additional Proposed Actions:")
        for i, action in enumerate(proposed_actions, 1):
            print(f"{i}. {action['description']}")
            if isinstance(action['details'], list):
                for detail in action['details'][:2]:  # Show first 2 details
                    if isinstance(detail, dict):
                        print(f"   • {detail.get('category', 'Category')}: merge {', '.join(detail.get('merge_folders', []))} into {detail.get('primary_folder', 'primary')}")
                    else:
                        print(f"   • {detail}")
            else:
                print(f"   {action['details']}")
        
        print(f"\n{'='*60}")
        print("Choose which additional actions to apply:")
        
        choices = {}
        for action in proposed_actions:
            while True:
                response = input(f"\n{action['description']}? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    choices[action['action']] = True
                    break
                elif response in ['n', 'no']:
                    break
                else:
                    print("Please enter 'y' or 'n'")
        
        if not any(choices.values()):
            print("No additional changes selected.")
            if current_state['metadata'].get('duplicates_removed', 0) > 0:
                print("✅ Duplicates have been removed. Processing complete!")
            return
        
        # Phase 2: Apply selected actions
        print(f"\n🔧 Phase 2: Applying selected improvements...")
        
        processed_bookmarks = bookmarks.copy()
        
        # Validate and remove broken links
        if choices.get('validate_and_remove_broken'):
            print("🔍 Validating all links...")
            validated = manager.validate_links(processed_bookmarks)
            valid_bookmarks = [b for b in validated if b['is_valid']]
            removed_broken = len(processed_bookmarks) - len(valid_bookmarks)
            processed_bookmarks = valid_bookmarks
            print(f"✅ Removed {removed_broken} broken links")
            
            # Update state
            current_state['bookmarks'] = processed_bookmarks
            current_state['processing_state']['links_validated'] = True
            current_state['metadata']['broken_links_removed'] = removed_broken
            manager.save_processing_state(current_state, intermediate_file)
        
        # Apply folder consolidations and organization
        if choices.get('consolidate_semantic_folders') or choices.get('organize_orphaned_enhanced'):
            folder_mapping = {}
            
            # Apply semantic consolidations
            if choices.get('consolidate_semantic_folders'):
                for suggestion in llm_analysis['consolidation_suggestions']:
                    primary = suggestion['primary_folder']
                    for merge_folder in suggestion['merge_folders']:
                        folder_mapping[merge_folder] = primary
                        print(f"📁 Consolidating '{merge_folder}' into '{primary}'")
            
            # Apply folder mappings and organize orphaned
            reorganized_bookmarks = []
            for bookmark in processed_bookmarks:
                new_bookmark = bookmark.copy()
                folder_path = bookmark['folder_path']
                
                # Apply consolidation mapping
                if folder_path in folder_mapping:
                    new_bookmark['folder_path'] = folder_mapping[folder_path]
                
                # Handle orphaned bookmarks with semantic suggestions
                elif choices.get('organize_orphaned_enhanced') and folder_path in ['Bookmarks Bar', 'Other Bookmarks', '']:
                    # Simple semantic categorization based on URL/title
                    url_domain = urlparse(bookmark['url']).netloc.lower()
                    title_lower = bookmark['name'].lower()
                    
                    suggested_folder = None
                    for category, folders in llm_analysis.get('folder_analysis', {}).items():
                        if category in folder_stats:  # Only suggest existing folders
                            continue
                    
                    # Fallback to domain-based suggestions
                    if 'github.com' in url_domain or 'stackoverflow.com' in url_domain:
                        suggested_folder = next((f for f in folder_stats.keys() if 'dev' in f.lower() or 'code' in f.lower()), None)
                    elif 'youtube.com' in url_domain or 'netflix.com' in url_domain:
                        suggested_folder = next((f for f in folder_stats.keys() if 'entertainment' in f.lower() or 'video' in f.lower()), None)
                    elif any(word in title_lower for word in ['news', 'article', 'blog']):
                        suggested_folder = next((f for f in folder_stats.keys() if 'news' in f.lower() or 'read' in f.lower()), None)
                    
                    if suggested_folder:
                        new_bookmark['folder_path'] = suggested_folder
                        print(f"📌 Moved '{bookmark['name'][:30]}...' to '{suggested_folder}'")
                
                reorganized_bookmarks.append(new_bookmark)
            
            processed_bookmarks = reorganized_bookmarks
            current_state['bookmarks'] = processed_bookmarks
            current_state['processing_state']['folders_reorganized'] = True
            manager.save_processing_state(current_state, intermediate_file)
        
        # Final step: Create clean bookmark file
        final_output = intermediate_file.replace('_processing_', '_final_')
        
        # Handle complete restructure format
        if complete_restructure_approved and llm_analysis.get('format_type') == 'complete_restructure':
            print(f"\n🔧 Phase 2: Applying complete restructure...")
            
            # Use the complete structure from LLM
            folder_structure = llm_analysis['folder_structure']
            new_bookmarks = manager.create_bookmarks_from_complete_structure(folder_structure, processed_bookmarks, not args.no_alphabetize)
            
            print(f"✅ Complete restructure applied successfully!")
            
        else:
            # Legacy incremental approach
            # Convert to Chrome bookmark format
            new_bookmarks = manager.create_reorganized_bookmarks({
                'remove_duplicates': False,  # Already done
                'remove_broken_links': False,  # Already done if selected
                'reorganize_folders': True
            }, alphabetize=not args.no_alphabetize)
            
            # Use processed bookmarks instead of re-processing
            new_bookmarks = {
                "checksum": manager.bookmarks.get("checksum", ""),
                "roots": {
                    "bookmark_bar": {"children": [], "date_added": str(int(time.time() * 1000000)), "date_modified": str(int(time.time() * 1000000)), "id": "1", "name": "Bookmarks bar", "type": "folder"},
                    "other": {"children": [], "date_added": str(int(time.time() * 1000000)), "date_modified": str(int(time.time() * 1000000)), "id": "2", "name": "Other bookmarks", "type": "folder"},
                    "synced": {"children": [], "date_added": str(int(time.time() * 1000000)), "date_modified": str(int(time.time() * 1000000)), "id": "3", "name": "Mobile bookmarks", "type": "folder"}
                },
                "version": 1
            }
            
            # Only continue with legacy processing if not using complete restructure
            if not complete_restructure_approved:
                # Organize bookmarks into proper nested folder structure
                folder_tree = {}
                
                def create_nested_folder_structure(path_parts: List[str], tree: Dict, depth: int = 0) -> Dict:
                    """Recursively create nested folder structure"""
                    if not path_parts:
                        return tree
                        
                    current_folder = path_parts[0]
                    remaining_parts = path_parts[1:]
                    
                    if current_folder not in tree:
                        tree[current_folder] = {
                            "children": [],
                            "date_added": str(int(time.time() * 1000000)),
                            "date_modified": str(int(time.time() * 1000000)),
                            "id": str(hash(current_folder + str(depth)) % 1000000),
                            "name": current_folder,
                            "type": "folder",
                            "subfolders": {}
                        }
                    
                    if remaining_parts:
                        tree[current_folder]["subfolders"] = create_nested_folder_structure(
                            remaining_parts, 
                            tree[current_folder].get("subfolders", {}),
                            depth + 1
                        )
                        
                    return tree
        
                def add_bookmark_to_tree(folder_path: str, bookmark_node: Dict):
                    """Add bookmark to the appropriate location in folder tree"""
                    nonlocal folder_tree  # Access the outer folder_tree variable
                    
                    if folder_path in ['Bookmarks Bar', '']:
                        new_bookmarks['roots']['bookmark_bar']['children'].append(bookmark_node)
                    elif folder_path in ['Other Bookmarks']:
                        new_bookmarks['roots']['other']['children'].append(bookmark_node)
                    else:
                        # Split path into components (handle both / and \ separators)
                        path_parts = [part.strip() for part in folder_path.replace('\\', '/').split('/') if part.strip()]
                        
                        if path_parts:
                            # Create nested structure
                            folder_tree = create_nested_folder_structure(path_parts, folder_tree)
                            
                            # Navigate to the deepest folder and add bookmark
                            current_tree = folder_tree
                            for part in path_parts[:-1]:
                                if part in current_tree:
                                    current_tree = current_tree[part]["subfolders"]
                            
                            # Add to final folder
                            final_folder = path_parts[-1]
                            if final_folder in current_tree:
                                current_tree[final_folder]["children"].append(bookmark_node)
                        else:
                            # Fallback to Other Bookmarks if path parsing fails
                            new_bookmarks['roots']['other']['children'].append(bookmark_node)
        
                # Process each bookmark
                for bookmark in processed_bookmarks:
                    folder_path = bookmark['folder_path']
                    
                    # Create bookmark node
                    bookmark_node = {
                        "date_added": bookmark.get('date_added', str(int(time.time() * 1000000))),
                        "date_modified": bookmark.get('date_modified', str(int(time.time() * 1000000))),
                        "id": str(hash(bookmark['url']) % 1000000),
                        "name": bookmark['name'],
                        "type": "url",
                        "url": bookmark['url']
                    }
                    
                    add_bookmark_to_tree(folder_path, bookmark_node)
        
                def flatten_folder_tree_to_chrome_format(tree: Dict, parent_children: List):
                    """Convert our tree structure to Chrome's flat children format"""
                    for folder_name, folder_data in tree.items():
                        chrome_folder = {
                            "children": folder_data["children"].copy(),  # Start with bookmarks
                            "date_added": folder_data["date_added"],
                            "date_modified": folder_data["date_modified"], 
                            "id": folder_data["id"],
                            "name": folder_name,
                            "type": "folder"
                        }
                        
                        # Recursively add subfolders
                        if "subfolders" in folder_data and folder_data["subfolders"]:
                            flatten_folder_tree_to_chrome_format(folder_data["subfolders"], chrome_folder["children"])
                        
                        parent_children.append(chrome_folder)
                
                # Add all top-level folders to "Other Bookmarks"
                flatten_folder_tree_to_chrome_format(folder_tree, new_bookmarks['roots']['other']['children'])
        
        manager.save_bookmarks(new_bookmarks, final_output)
        
        print(f"\n✅ Enhanced cleanup complete!")
        print(f"📁 Processing state saved: {intermediate_file}")
        print(f"📄 Final bookmarks saved: {final_output}")
        
        # Summary
        final_count = len(processed_bookmarks)
        original_count = current_state['original_count']
        total_removed = original_count - final_count
        
        print(f"\n📊 Summary:")
        print(f"  Original bookmarks: {original_count}")
        print(f"  Final bookmarks: {final_count}")
        if total_removed > 0:
            print(f"  Total removed: {total_removed}")
            if current_state['metadata'].get('duplicates_removed', 0) > 0:
                print(f"    - Duplicates: {current_state['metadata']['duplicates_removed']}")
            if current_state['metadata'].get('broken_links_removed', 0) > 0:
                print(f"    - Broken links: {current_state['metadata']['broken_links_removed']}")
        
        return
    
    if args.find_outdated:
        outdated = manager.identify_outdated_bookmarks(all_bookmarks, args.days_threshold)
        if outdated:
            print(f"\nFound {len(outdated)} bookmarks older than {args.days_threshold} days:")
            for bookmark in outdated[:10]:  # Show first 10
                print(f"  - {bookmark['name']} ({bookmark['days_old']} days old)")
            if len(outdated) > 10:
                print(f"  ... and {len(outdated) - 10} more")
        else:
            print(f"No bookmarks older than {args.days_threshold} days found.")

if __name__ == "__main__":
    main()
