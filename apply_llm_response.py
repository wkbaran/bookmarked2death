#!/usr/bin/env python3

"""
Apply a saved LLM response to bookmarks without re-querying the LLM
"""

import json
import sys
import os
from bookmark_manager import BookmarkManager

def main():
    if len(sys.argv) != 4:
        print("Usage: python apply_llm_response.py <bookmarks_file> <raw_llm_response.json> <output_file>")
        print("Example: python apply_llm_response.py Bookmarks raw.json reorganized_bookmarks.json")
        sys.exit(1)
    
    bookmarks_file = sys.argv[1]
    llm_response_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Load the LLM response
    print(f"📄 Loading LLM response from: {llm_response_file}")
    with open(llm_response_file, 'r', encoding='utf-8') as f:
        llm_response = json.load(f)
    
    # Initialize bookmark manager
    print(f"📚 Loading bookmarks from: {bookmarks_file}")
    manager = BookmarkManager(bookmarks_file)
    manager.bookmarks = manager.load_bookmarks()
    
    if not manager.bookmarks:
        print(f"❌ Failed to load bookmarks from {bookmarks_file}")
        sys.exit(1)
    
    # Extract all bookmarks for processing
    all_bookmarks = []
    if 'roots' in manager.bookmarks:
        for root_name, root_node in manager.bookmarks['roots'].items():
            if root_name == 'bookmark_bar':
                bookmarks = manager.extract_all_bookmarks(root_node, "Bookmarks Bar")
            else:
                bookmarks = manager.extract_all_bookmarks(root_node, root_name.replace('_', ' ').title())
            all_bookmarks.extend(bookmarks)
    
    print(f"📊 Total bookmarks found: {len(all_bookmarks)}")
    
    # Remove duplicates (like the main flow does)
    duplicates = manager.find_duplicates()
    if duplicates:
        print(f"🔍 Found {len(duplicates)} sets of duplicates")
        unique_bookmarks = []
        processed_urls = set()
        
        for bookmark in all_bookmarks:
            url = bookmark['url'].lower().strip()
            if url not in processed_urls:
                unique_bookmarks.append(bookmark)
                processed_urls.add(url)
        
        print(f"✂️ After removing duplicates: {len(unique_bookmarks)} bookmarks")
        all_bookmarks = unique_bookmarks
    
    # Apply the LLM response with bookmark preservation
    print("🤖 Applying LLM reorganization with bookmark preservation...")
    try:
        new_structure = manager.create_bookmarks_from_complete_structure(
            llm_response['folder_structure'], 
            all_bookmarks
        )
        
        # Save the result
        print(f"💾 Saving reorganized bookmarks to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_structure, f, indent=2, ensure_ascii=False)
        
        print("✅ Successfully applied LLM reorganization!")
        print(f"📁 Output saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error applying LLM response: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()