"""
LLM Templates and Configuration for Bookmark Analysis
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Load .env file
from dotenv import load_dotenv
load_dotenv()  # This will load .env from current directory or parent directories

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str = "openai"  # "openai", "anthropic", or "none"
    model: str = "gpt-4.1"  # or "claude-3-sonnet-20240229"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    custom_template: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create config from environment variables"""
        provider = os.environ['BOOKMARK_LLM_PROVIDER'].lower()
        
        if provider == 'openai':
            api_key = os.environ['OPENAI_API_KEY']
            model = os.environ['BOOKMARK_LLM_MODEL']
        elif provider == 'anthropic':
            api_key = os.environ['ANTHROPIC_API_KEY']
            model = os.environ['BOOKMARK_LLM_MODEL']
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: 'openai', 'anthropic'")
            
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=float(os.environ['BOOKMARK_LLM_TEMPERATURE']),
            max_tokens=int(os.environ['BOOKMARK_LLM_MAX_TOKENS']) if 'BOOKMARK_LLM_MAX_TOKENS' in os.environ and os.environ['BOOKMARK_LLM_MAX_TOKENS'].strip() else None,
            custom_template=os.environ.get('BOOKMARK_LLM_TEMPLATE')  # This is optional
        )


# Use string template to avoid format conflicts
FOLDER_STRUCTURE_TEMPLATE = """
{folder_name}:
  - Bookmark Count: {bookmark_count}
  - Top Domains: {top_domains}
  - Sample Titles: {sample_titles}
  - Common Keywords: {keywords}
  - Subfolder Structure: {subfolders}
"""

def format_folder_for_llm(folder_name: str, folder_data: Dict) -> str:
    """Format individual folder data for LLM analysis"""
    
    # Get top domains (limit to 5)
    domains = folder_data['domains']
    top_domains = ', '.join(sorted(set(domains), key=domains.count, reverse=True)[:5]) if domains else 'None'
    
    # Get sample bookmark titles (limit to 5)
    bookmarks = folder_data['bookmarks']
    sample_titles = []
    for bookmark in bookmarks[:5]:
        title = bookmark['name']
        if len(title) > 50:
            title = title[:47] + "..."
        sample_titles.append(title)
    sample_titles_str = ', '.join(sample_titles) if sample_titles else 'None'
    
    # Get keywords (limit to 8)
    keywords = ', '.join(folder_data['keywords'][:8]) if folder_data['keywords'] else 'None'
    
    # Format subfolder structure
    subfolders = 'None'
    if folder_name.count('/') > 0 or any('/' in b['folder_path'] and b['folder_path'].startswith(folder_name + '/') for b in bookmarks):
        subfolders = 'Has subfolders'
    
    return FOLDER_STRUCTURE_TEMPLATE.format(
        folder_name=folder_name,
        bookmark_count=folder_data['bookmark_count'],
        top_domains=top_domains,
        sample_titles=sample_titles_str,
        keywords=keywords,
        subfolders=subfolders
    )

def prepare_folder_structure_for_llm(folder_stats: Dict[str, List[Dict]]) -> str:
    """Convert folder stats to structured tree format for LLM (bookmark titles with GUIDs, no URLs)"""
    
    # Build nested folder structure
    def build_tree() -> Dict:
        tree = {}
        
        for folder_path, bookmarks in folder_stats.items():
            # Split folder path into components
            path_parts = folder_path.split('/')
            
            # Navigate/create the nested structure
            current_level = tree
            for part in path_parts:
                if part not in current_level:
                    current_level[part] = {'bookmarks': [], 'subfolders': {}}
                current_level = current_level[part]['subfolders']
            
            # Add bookmarks to the final folder
            final_folder = path_parts[-1]
            if path_parts[-1] not in tree:
                # Handle case where we need to backtrack to add bookmarks
                current_level = tree
                for part in path_parts[:-1]:
                    current_level = current_level[part]['subfolders']
                if final_folder not in current_level:
                    current_level[final_folder] = {'bookmarks': [], 'subfolders': {}}
                current_level[final_folder]['bookmarks'] = bookmarks
            else:
                # Navigate to correct location to add bookmarks
                current_level = tree
                for part in path_parts:
                    if part in current_level:
                        if 'bookmarks' not in current_level[part]:
                            current_level[part]['bookmarks'] = []
                        if part == path_parts[-1]:
                            current_level[part]['bookmarks'] = bookmarks
                        else:
                            current_level = current_level[part]['subfolders']
        
        return tree
    
    def format_tree(tree: Dict, indent: str = "") -> List[str]:
        lines = []
        for folder_name, folder_data in sorted(tree.items()):
            lines.append(f"{indent}{folder_name}:")
            
            # Add bookmarks with GUID
            for bookmark in folder_data.get('bookmarks', []):
                lines.append(f"{indent}  - {bookmark['name']} (guid: {bookmark.get('guid', 'unknown')})")
            
            # Add subfolders recursively
            if folder_data.get('subfolders'):
                lines.extend(format_tree(folder_data['subfolders'], indent + "  "))
            
            if not folder_data.get('subfolders'):
                lines.append("")  # Empty line after folders with no subfolders
        
        return lines
    
    # For simple flat structure, just list folders with bookmarks
    if not any('/' in folder_path for folder_path in folder_stats.keys()):
        result_lines = []
        for folder_name, bookmarks in sorted(folder_stats.items()):
            result_lines.append(f"{folder_name}:")
            for bookmark in bookmarks:
                result_lines.append(f"  - {bookmark['name']} (guid: {bookmark.get('guid', 'unknown')})")
            result_lines.append("")  # Empty line between folders
        return '\n'.join(result_lines)
    
    # Handle nested structure
    tree = build_tree()
    lines = format_tree(tree)
    return '\n'.join(lines)

def get_template(config: LLMConfig) -> str:
    """Get the template for analysis from external file"""
    # Check for template files in order of preference
    template_files = [
        config.custom_template,         # Environment variable override
        './llm_template.txt',           # Local project config
        './template.txt',               # Alternative local config
    ]
    
    for template_path in template_files:
        if template_path and os.path.exists(template_path):
            print(f"📋 Using template: {template_path}")
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    # If custom_template is set but not a file, treat as inline template
    if config.custom_template and not os.path.exists(config.custom_template):
        return config.custom_template
    
    raise FileNotFoundError("No template file found. Please ensure llm_template.txt exists.")
