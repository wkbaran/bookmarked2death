"""
LLM Client Implementation for Bookmark Analysis
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from llm_templates import LLMConfig, prepare_folder_structure_for_llm, get_template

def setup_llm_logger() -> logging.Logger:
    """Set up dedicated logger for LLM operations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_filename = f"{logs_dir}/{timestamp}.log"
    
    logger = logging.getLogger('llm_client')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Don't propagate to root logger to avoid console spam
    logger.propagate = False
    
    logger.info(f"=== LLM Operations Log Started ===")
    logger.info(f"Log file: {log_filename}")
    
    # Print to console so user knows where logs are
    print(f"📝 Detailed LLM logging enabled: {log_filename}")
    
    return logger

class LLMClient:
    def __init__(self, config: LLMConfig, output_raw_response: bool = False, raw_output_file: str = None, debug_prompt: bool = False):
        self.config = config
        self.client = None
        self.output_raw_response = output_raw_response
        self.raw_output_file = raw_output_file
        self.debug_prompt = debug_prompt
        self.logger = setup_llm_logger()
        self.logger.info(f"Initializing LLMClient with provider: {config.provider}, model: {config.model}")
        
        if config.provider == 'openai':
            import openai
            self.client = openai.OpenAI(
                api_key=config.api_key,
                timeout=300.0  # 5 minute timeout
            )
            self.logger.info("Successfully initialized OpenAI client")
                
        elif config.provider == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=config.api_key,
                timeout=300.0  # 5 minute timeout  
            )
            self.logger.info("Successfully initialized Anthropic client")
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    def is_available(self) -> bool:
        """Check if LLM client is properly configured"""
        return self.client is not None
    
    def analyze_folders(self, folder_stats: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Send folder analysis to LLM and return structured response"""
        self.logger.info(f"Starting folder analysis for {len(folder_stats)} folders")
        
        if not self.is_available():
            raise RuntimeError("LLM client not properly configured")
            
        result = self._analyze_single_chunk(folder_stats)
        if result is None and self.output_raw_response:
            print("⏩ Analysis completed (raw output mode)")
            return None
        return result
    
    def _analyze_single_chunk(self, folder_stats: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Analyze a single chunk of folders"""
        chunk_id = f"chunk_{datetime.now().strftime('%H%M%S')}"
        self.logger.info(f"=== Starting analysis for {chunk_id} with {len(folder_stats)} folders ===")
        
        # Prepare structured data for LLM
        folder_structure = prepare_folder_structure_for_llm(folder_stats)
        template = get_template(self.config)
        
        # Use safe string replacement to avoid format conflicts
        prompt = template.replace('{folder_structure}', folder_structure)
        
        # Log the full request
        self.logger.info(f"=== REQUEST {chunk_id} ===")
        self.logger.info(f"Provider: {self.config.provider}")
        self.logger.info(f"Model: {self.config.model}")
        self.logger.info(f"Temperature: {self.config.temperature}")
        self.logger.info(f"Max tokens: {self.config.max_tokens}")
        self.logger.info(f"Folder count: {len(folder_stats)}")
        self.logger.info(f"Folder names: {list(folder_stats.keys())}")
        self.logger.info(f"Prompt length: {len(prompt)} characters")
        self.logger.debug(f"Full prompt:\n{prompt}")
        
        # Count total bookmarks being sent
        total_bookmarks = sum(len(bookmarks) for bookmarks in folder_stats.values())
        print(f"🔍 Sending {len(folder_stats)} folders with {total_bookmarks} total bookmarks to LLM for analysis...")
        print(f"Using {self.config.provider} with model {self.config.model}")
        
        # Debug: Show prompt size
        print(f"📏 Prompt size: {len(prompt):,} characters")
        if len(prompt) > 100000:
            print(f"⚠️ WARNING: Very large prompt ({len(prompt):,} chars) - may be truncated by LLM")
        
        # Save prompt for debugging if requested
        if self.debug_prompt:
            prompt_file = "debug_prompt.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"📝 Debug prompt saved to: {prompt_file}")
        
        # Show progress indicator for potentially long request
        import time
        import threading
        
        progress_active = [True]  # Use list to make it mutable in nested function
        def show_progress():
            chars = "|/-\\"
            i = 0
            while progress_active[0]:
                print(f"\r⏳ Waiting for LLM response... {chars[i % len(chars)]}", end="", flush=True)
                time.sleep(0.5)
                i += 1
        
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            if self.config.provider == 'openai':
                request_data = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert at organizing bookmark collections. Respond with the exact JSON structure requested."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                }
                
                # Only add max_tokens if it's set
                if self.config.max_tokens:
                    request_data["max_tokens"] = self.config.max_tokens
                    
                request_data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bookmark_structure",
                            "strict": False,
                            "schema": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "folder_structure": { "$ref": "#/$defs/folder" }
                                },
                                "required": ["folder_structure"],
                                "$defs": {
                                    "folder": {
                                        "type": "object",
                                        "description": "A folder may contain bookmark_guids and any number of subfolders keyed by folder name. Empty folders are allowed.",
                                        "additionalProperties": False,
                                        "properties": {
                                            "bookmark_guids": {
                                                "type": "array",
                                                "description": "List of bookmark GUIDs contained in this folder",
                                                "items": { "type": "string" }
                                            }
                                        },
                                        "patternProperties": {
                                            "^(?!bookmark_guids$).*": { "$ref": "#/$defs/folder" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                
                self.logger.info(f"OpenAI request parameters: {json.dumps({k: v for k, v in request_data.items() if k != 'messages'}, indent=2)}")
                
                # Make the API call (timeout handled at client level)
                response = self.client.chat.completions.create(**request_data)
                content = response.choices[0].message.content
                
                # Output raw response immediately if requested (before any processing)
                if self.output_raw_response:
                    if self.raw_output_file:
                        with open(self.raw_output_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ Raw LLM response saved to: {self.raw_output_file}")
                        print(f"📏 Response length: {len(content)} characters")
                    else:
                        print(f"\n{'='*80}")
                        print("🔍 FULL RAW LLM RESPONSE (OpenAI):")
                        print(f"{'='*80}")
                        print(content)  # Print the complete response without truncation
                        print(f"{'='*80}")
                        print(f"📏 Response length: {len(content)} characters")
                        print(f"{'='*80}\n")
                
                # Log response metadata
                usage = response.usage
                self.logger.info(f"=== RESPONSE {chunk_id} ===")
                self.logger.info(f"Response length: {len(content)} characters")
                self.logger.info(f"Tokens used - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, Total: {usage.total_tokens}")
                self.logger.info(f"Full response:\n{content}")
                
                print(f"✅ Received response from OpenAI ({len(content)} characters)")
            
            elif self.config.provider == 'anthropic':
                request_data = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                # Only add max_tokens if it's set (Anthropic requires it, so use a high default)
                if self.config.max_tokens:
                    request_data["max_tokens"] = self.config.max_tokens
                else:
                    request_data["max_tokens"] = 8000  # Anthropic requires max_tokens
                
                self.logger.info(f"Anthropic request parameters: {json.dumps({k: v for k, v in request_data.items() if k != 'messages'}, indent=2)}")
                
                # Make the API call (timeout handled at client level)
                response = self.client.messages.create(**request_data)
                content = response.content[0].text
                
                # Output raw response immediately if requested (before any processing)
                if self.output_raw_response:
                    if self.raw_output_file:
                        with open(self.raw_output_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ Raw LLM response saved to: {self.raw_output_file}")
                        print(f"📏 Response length: {len(content)} characters")
                    else:
                        print(f"\n{'='*80}")
                        print("🔍 FULL RAW LLM RESPONSE (Anthropic):")
                        print(f"{'='*80}")
                        print(content)  # Print the complete response without truncation
                        print(f"{'='*80}")
                        print(f"📏 Response length: {len(content)} characters")
                        print(f"{'='*80}\n")
                
                # Log response metadata
                usage = response.usage
                self.logger.info(f"=== RESPONSE {chunk_id} ===")
                self.logger.info(f"Response length: {len(content)} characters")
                self.logger.info(f"Tokens used - Input: {usage.input_tokens}, Output: {usage.output_tokens}")
                self.logger.info(f"Full response:\n{content}")
                
                print(f"✅ Received response from Anthropic ({len(content)} characters)")
            else:
                self.logger.error("Unsupported provider")
                return None
                
        except Exception as e:
            # Handle any unhandled exceptions
            print(f"\n❌ Error during LLM request: {e}")
            raise
        finally:
            # Stop the progress indicator
            progress_active[0] = False
            print(f"\r" + " " * 50 + "\r", end="", flush=True)  # Clear the progress line
        
        # If we're just outputting raw response, skip parsing entirely
        if self.output_raw_response:
            print("🔍 Skipping JSON parsing - raw output only")
            return None
        
        # Parse JSON response
        result = self._parse_llm_response(content)
        
        if result:
            self.logger.info(f"=== PARSED RESULT {chunk_id} ===")
            self.logger.info(f"Consolidations found: {len(result.get('consolidations', []))}")
            self.logger.info(f"Renames found: {len(result.get('renames', []))}")
            self.logger.info(f"New hierarchies found: {len(result.get('new_hierarchies', []))}")
            self.logger.info(f"Orphan suggestions found: {len(result.get('orphan_suggestions', []))}")
            self.logger.debug(f"Full parsed result:\n{json.dumps(result, indent=2)}")
        else:
            self.logger.error(f"Failed to parse response for {chunk_id}")
        
        return result
    
    def _analyze_folders_chunked(self, folder_stats: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Process large folder collections in chunks"""
        self.logger.info(f"=== Starting chunked analysis for {len(folder_stats)} folders ===")
        
        # Sort folders by bookmark count (prioritize larger folders)
        sorted_folders = sorted(folder_stats.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Split into chunks of ~30 folders each
        chunk_size = 30
        chunks = []
        for i in range(0, len(sorted_folders), chunk_size):
            chunk_items = sorted_folders[i:i + chunk_size]
            chunks.append(dict(chunk_items))
        
        self.logger.info(f"Split into {len(chunks)} chunks of ~{chunk_size} folders each")
        print(f"📦 Processing {len(chunks)} chunks of ~{chunk_size} folders each...")
        
        all_results = {
            'consolidations': [],
            'renames': [],
            'new_hierarchies': [],
            'orphan_suggestions': []
        }
        
        for i, chunk in enumerate(chunks, 1):
            chunk_folders = list(chunk.keys())
            self.logger.info(f"=== Processing chunk {i}/{len(chunks)} ===")
            self.logger.info(f"Chunk {i} folders: {chunk_folders}")
            
            print(f"\n📦 Processing chunk {i}/{len(chunks)} ({len(chunk)} folders)...")
            
            chunk_result = self._analyze_single_chunk(chunk)
            if chunk_result:
                # Log chunk results before merging
                self.logger.info(f"Chunk {i} results - Consolidations: {len(chunk_result['consolidations'])}, Renames: {len(chunk_result['renames'])}")
                
                # Merge results
                for key in all_results.keys():
                    if key in chunk_result:
                        all_results[key].extend(chunk_result[key])
                print(f"✅ Chunk {i} completed successfully")
            elif self.output_raw_response:
                print(f"⏩ Chunk {i} skipped (raw output mode)")
                # Continue processing other chunks in raw output mode
                continue
            else:
                raise RuntimeError(f"Chunk {i} processing failed")
        
        # Remove duplicate suggestions across chunks
        original_consolidations = len(all_results['consolidations'])
        all_results['consolidations'] = self._deduplicate_consolidations(all_results['consolidations'])
        
        if original_consolidations > len(all_results['consolidations']):
            dedupe_removed = original_consolidations - len(all_results['consolidations'])
            self.logger.info(f"Removed {dedupe_removed} duplicate consolidation suggestions")
        
        # Log final results
        self.logger.info(f"=== Final chunked results ===")
        self.logger.info(f"Total consolidations: {len(all_results['consolidations'])}")
        self.logger.info(f"Total renames: {len(all_results['renames'])}")
        self.logger.info(f"Total hierarchies: {len(all_results['new_hierarchies'])}")
        self.logger.info(f"Total orphan suggestions: {len(all_results['orphan_suggestions'])}")
        
        return all_results if any(all_results.values()) else None
    
    def _deduplicate_consolidations(self, consolidations: List[Dict]) -> List[Dict]:
        """Remove duplicate consolidation suggestions"""
        seen_pairs = set()
        unique_consolidations = []
        
        for consolidation in consolidations:
            # Create a signature for this consolidation
            primary = consolidation['primary_folder']
            merge_folders = tuple(sorted(consolidation['merge_folders']))
            signature = (primary, merge_folders)
            
            if signature not in seen_pairs:
                seen_pairs.add(signature)
                unique_consolidations.append(consolidation)
        
        return sorted(unique_consolidations, key=lambda x: x['confidence'], reverse=True)
    
    def _parse_llm_response(self, content: str) -> Optional[Dict]:
        """Parse and validate LLM response"""
        self.logger.info("=== Starting response parsing ===")
        self.logger.info(f"Content length: {len(content)} characters")
        
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            content = content.strip()
            
            # Debug: Print the actual response
            print(f"\n🔍 LLM Response (first 500 chars):")
            print(f"{content[:500]}...")
            print(f"{'='*50}")
            
            self.logger.debug(f"Raw response content (first 1000 chars): {content[:1000]}")
            
            # Try multiple parsing strategies
            json_str = None
            
            # Strategy 1: Look for code blocks
            if '```json' in content:
                start_marker = '```json'
                end_marker = '```'
                start_idx = content.find(start_marker) + len(start_marker)
                end_idx = content.find(end_marker, start_idx)
                if end_idx != -1:
                    json_str = content[start_idx:end_idx].strip()
            
            # Strategy 2: Look for bare JSON
            elif '{' in content and '}' in content:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
            
            # Strategy 3: Try to reconstruct partial JSON 
            if not json_str and content:
                # Check if it looks like partial JSON starting with a key
                cleaned = content.strip()
                # Look for new folder_structure format or old consolidations format
                if (cleaned.startswith('"folder_structure"') or cleaned.startswith('  "folder_structure"') or
                    cleaned.startswith('"consolidations"') or cleaned.startswith('  "consolidations"')):
                    # Try to reconstruct as complete JSON
                    json_str = "{" + cleaned.lstrip()
                    if not json_str.endswith('}'):
                        json_str += "}"
                
                # Strategy 4: Clean up common formatting issues
                if not json_str:
                    for prefix in ["Here's the analysis:", "```json", "```", "Response:", "Analysis:"]:
                        cleaned = cleaned.replace(prefix, "")
                    cleaned = cleaned.strip()
                    
                    # Try to find JSON in cleaned content
                    if '{' in cleaned and '}' in cleaned:
                        start_idx = cleaned.find('{')
                        end_idx = cleaned.rfind('}') + 1
                        json_str = cleaned[start_idx:end_idx]
            
            if json_str:
                print(f"\n🔍 Extracted JSON:")
                print(f"{json_str[:300]}...")
                
                self.logger.info(f"Successfully extracted JSON string ({len(json_str)} chars)")
                self.logger.debug(f"Extracted JSON: {json_str}")
                
                # Check if JSON appears to be truncated
                if not json_str.rstrip().endswith('}'):
                    raise ValueError(f"JSON appears to be truncated (doesn't end with closing brace). Length: {len(json_str)} chars")
                
                # Try to repair common JSON issues
                json_str = self._attempt_json_repair(json_str)
                
                result = json.loads(json_str)
                self.logger.info("JSON parsing successful")
                
                # Check if this is the new folder_structure format
                if 'folder_structure' in result:
                    self.logger.info("Detected new folder_structure format")
                    # Validate new format
                    required_keys = ['folder_structure']
                    for key in required_keys:
                        if key not in result:
                            raise ValueError(f"Missing required key '{key}' in LLM response")
                    
                    # Validate that folder_structure contains nested structure
                    if not isinstance(result['folder_structure'], dict):
                        raise ValueError("folder_structure must be a dictionary")
                    
                    # Optional: validate that all bookmark_guids are present and unique
                    all_guids = self._extract_all_guids(result['folder_structure'])
                    self.logger.info(f"Found {len(all_guids)} bookmark GUIDs in structure")
                    
                    return result
                
                else:
                    # Legacy format validation
                    self.logger.info("Detected legacy consolidations format")
                    required_keys = ['consolidations', 'renames', 'new_hierarchies', 'orphan_suggestions']
                    for key in required_keys:
                        if key not in result:
                            raise ValueError(f"Missing required key '{key}' in LLM response")
                    
                    # Validate and normalize new_hierarchies structure
                    if 'new_hierarchies' in result:
                        valid_hierarchies = []
                        for hierarchy in result['new_hierarchies']:
                            if isinstance(hierarchy, dict):
                                # Normalize different field names that LLM might use
                                normalized = {}
                                
                                # Handle parent field variations
                                if 'parent' in hierarchy:
                                    normalized['parent'] = hierarchy['parent']
                                elif 'parent_folder' in hierarchy:
                                    normalized['parent'] = hierarchy['parent_folder']
                                else:
                                    self.logger.warning(f"Hierarchy missing parent field: {hierarchy}")
                                    continue
                                
                                # Handle children field variations
                                if 'children' in hierarchy:
                                    normalized['children'] = hierarchy['children']
                                elif 'child_folders' in hierarchy:
                                    normalized['children'] = hierarchy['child_folders']
                                else:
                                    normalized['children'] = []
                                
                                # Copy other fields
                                for key in ['reasoning']:
                                    if key in hierarchy:
                                        normalized[key] = hierarchy[key]
                                
                                valid_hierarchies.append(normalized)
                                self.logger.info(f"Normalized hierarchy: {normalized}")
                            else:
                                self.logger.warning(f"Skipping non-dict hierarchy entry: {hierarchy}")
                        
                        result['new_hierarchies'] = valid_hierarchies
                    
                    # Validate and normalize orphan_suggestions structure
                    if 'orphan_suggestions' in result:
                        valid_orphans = []
                        for orphan in result['orphan_suggestions']:
                            if isinstance(orphan, dict):
                                normalized = {}
                                
                                # Handle bookmark_pattern field variations
                                if 'bookmark_pattern' in orphan:
                                    normalized['bookmark_pattern'] = orphan['bookmark_pattern']
                                elif 'folder' in orphan:
                                    normalized['bookmark_pattern'] = orphan['folder']
                                elif 'pattern' in orphan:
                                    normalized['bookmark_pattern'] = orphan['pattern']
                                else:
                                    self.logger.warning(f"Orphan suggestion missing pattern field: {orphan}")
                                    continue
                                
                                # Handle suggested_folder field variations
                                if 'suggested_folder' in orphan:
                                    normalized['suggested_folder'] = orphan['suggested_folder']
                                elif 'suggestion' in orphan:
                                    normalized['suggested_folder'] = orphan['suggestion']
                                elif 'target_folder' in orphan:
                                    normalized['suggested_folder'] = orphan['target_folder']
                                else:
                                    self.logger.warning(f"Orphan suggestion missing target field: {orphan}")
                                    continue
                                
                                # Copy other fields
                                for key in ['reasoning']:
                                    if key in orphan:
                                        normalized[key] = orphan[key]
                                
                                valid_orphans.append(normalized)
                                self.logger.info(f"Normalized orphan suggestion: {normalized}")
                            else:
                                self.logger.warning(f"Skipping non-dict orphan suggestion: {orphan}")
                        
                        result['orphan_suggestions'] = valid_orphans
                
                return result
            else:
                raise ValueError(f"Could not extract valid JSON from LLM response: {content[:200]}...")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing error: {e}. Attempted to parse: {json_str[:200] if json_str else 'None'}...")
        except Exception as e:
            import traceback
            raise RuntimeError(f"Error processing LLM response: {e}\nTraceback: {traceback.format_exc()}")
    
    def _attempt_json_repair(self, json_str: str) -> str:
        """Attempt to repair common JSON formatting issues"""
        if not json_str:
            return json_str
        
        # Remove any trailing commas before closing braces/brackets
        import re
        # Fix trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix missing commas between array elements (common in truncated responses)
        # This is more complex and risky, so we'll be conservative
        lines = json_str.split('\n')
        repaired_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # If this line ends with a quote and the next line starts with a quote,
            # and there's no comma, add one
            if (i < len(lines) - 1 and 
                stripped.endswith('"') and not stripped.endswith('",') and
                lines[i + 1].strip().startswith('"')):
                repaired_lines.append(line + ',')
            else:
                repaired_lines.append(line)
        
        return '\n'.join(repaired_lines)

    def _extract_all_guids(self, folder_structure: Dict, path: str = "") -> List[str]:
        """Recursively extract all bookmark GUIDs from folder structure"""
        all_guids = []
        
        for folder_name, folder_data in folder_structure.items():
            current_path = f"{path}/{folder_name}" if path else folder_name
            
            if isinstance(folder_data, dict):
                # Check for bookmark_guids at this level
                if 'bookmark_guids' in folder_data and isinstance(folder_data['bookmark_guids'], list):
                    all_guids.extend(folder_data['bookmark_guids'])
                    self.logger.debug(f"Found {len(folder_data['bookmark_guids'])} GUIDs in {current_path}")
                
                # Recursively check subfolders (any key that's not bookmark_guids)
                for key, value in folder_data.items():
                    if key != 'bookmark_guids' and isinstance(value, dict):
                        all_guids.extend(self._extract_all_guids({key: value}, current_path))
        
        return all_guids

def create_llm_client(output_raw_response: bool = False, raw_output_file: str = None, debug_prompt: bool = False) -> LLMClient:
    """Factory function to create LLM client from environment"""
    config = LLMConfig.from_env()
    return LLMClient(config, output_raw_response, raw_output_file, debug_prompt)

def convert_llm_suggestions_to_internal_format(llm_result: Dict) -> Dict:
    """Convert LLM response to internal format"""
    
    # Check if this is the new folder_structure format
    if 'folder_structure' in llm_result:
        # New format - return as complete restructuring
        return {
            'folder_structure': llm_result['folder_structure'],
            'reasoning': llm_result.get('reasoning', {}),
            'summary': llm_result.get('summary', {}),
            'llm_analysis': llm_result,
            'format_type': 'complete_restructure'
        }
    
    # Legacy format - convert to consolidations
    consolidation_suggestions = []
    
    # Convert consolidations
    for consolidation in llm_result.get('consolidations', []):
        consolidation_suggestions.append({
            'category': 'llm_suggested',
            'primary_folder': consolidation['primary_folder'],
            'merge_folders': consolidation['merge_folders'],
            'confidence': consolidation['confidence'],
            'reasoning': consolidation['reasoning'],
            'bookmark_counts': [0] * (len(consolidation['merge_folders']) + 1)  # Placeholder
        })
    
    # Convert renames to consolidations (rename = consolidate with new name)
    for rename in llm_result.get('renames', []):
        consolidation_suggestions.append({
            'category': 'llm_rename',
            'primary_folder': rename['new_name'],
            'merge_folders': [rename['old_name']],
            'confidence': 0.9,
            'reasoning': f"Rename suggestion: {rename['reasoning']}",
            'bookmark_counts': [0, 0]  # Placeholder
        })
    
    return {
        'consolidation_suggestions': sorted(consolidation_suggestions, key=lambda x: x['confidence'], reverse=True),
        'new_hierarchies': llm_result.get('new_hierarchies', []),
        'orphan_suggestions': llm_result.get('orphan_suggestions', []),
        'llm_analysis': llm_result,
        'format_type': 'incremental_suggestions'
    }