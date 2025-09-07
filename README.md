# Bookmarked2Death

A comprehensive browser bookmark management tool that helps you organize, clean up, and maintain your bookmarks.

## Features

- **🔍 Smart Duplicate Detection**: Find and remove duplicate bookmarks across all folders
- **🔗 Cached Link Validation**: Test bookmark URLs with intelligent caching to avoid redundant checks
- **💾 Automatic Backups**: Create timestamped backups before making any changes
- **🤖 LLM-Powered Folder Analysis**: Advanced AI analysis for intelligent folder organization
- **📊 Phased Processing**: Process bookmarks in stages with intermediate saves and resume capability
- **📁 Resume Functionality**: Resume processing from any stage if interrupted
- **🎯 Bookmarks Bar Handling**: Treat bookmarks bar as a separate organizational space
- **⏰ Outdated Bookmark Detection**: Identify bookmarks that haven't been modified in a long time
- **🗂️ Full Subfolder Support**: Maintains and creates proper nested folder hierarchies
- **🔤 Alphabetical Organization**: Automatically alphabetizes bookmarks within folders for improved organization and findability

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bookmarked2death
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or: venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API key:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your API key
# For OpenAI: OPENAI_API_KEY=sk-your-key-here
# For Anthropic: ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Usage

### Find Your Bookmark File

**Chrome/Chromium/Edge:**
- Linux: `~/.config/google-chrome/Default/Bookmarks`
- Windows: `%LOCALAPPDATA%\Google\Chrome\User Data\Default\Bookmarks`
- macOS: `~/Library/Application Support/Google/Chrome/Default/Bookmarks`

**Firefox:**
Export bookmarks as JSON from: Bookmarks → Manage Bookmarks → Import and Backup → Export Bookmarks to HTML/JSON

### Enhanced Processing (Recommended)

```bash
# Interactive cleanup with phased processing and resume capability
./bookmark_manager.py path/to/Bookmarks --create-plan

# With custom output file
./bookmark_manager.py path/to/Bookmarks --create-plan --output my_bookmarks_cleaned.json
```

### Individual Analysis Features

```bash
# Individual features for analysis only
./bookmark_manager.py path/to/Bookmarks --find-duplicates
./bookmark_manager.py path/to/Bookmarks --validate-links
./bookmark_manager.py path/to/Bookmarks --suggest-folders
./bookmark_manager.py path/to/Bookmarks --find-outdated --days-threshold 180
```

### Options

- `--create-plan`: Interactive cleanup with phased processing (recommended)
- `--backup`: Create a timestamped backup before processing
- `--find-duplicates`: Find and list duplicate bookmarks
- `--validate-links`: Test bookmark URLs and identify broken links (cached)
- `--suggest-folders`: Analyze folder structure and suggest improvements
- `--find-outdated`: Find bookmarks older than threshold (default: 365 days)
- `--days-threshold N`: Set custom threshold for outdated bookmarks
- `--output FILE`: Specify output file for cleaned bookmarks
- `--no-alphabetize`: Skip alphabetical sorting of folders and bookmarks (by default, all folders and bookmarks are sorted alphabetically)

## Enhanced Processing Workflow

The `--create-plan` option provides a sophisticated, multi-phase approach:

1. **Phase 1**: Removes duplicates and saves intermediate state
2. **Resume Capability**: Can resume from any previous state
3. **Phase 2**: Validates links with caching and semantic folder analysis
4. **LLM-Powered Organization**: Uses advanced AI to understand your bookmark patterns and suggest optimal folder structure
5. **Nested Structure Preservation**: Maintains complex subfolder hierarchies

## Example Output

### Enhanced Processing Workflow
```
🔍 Phase 1: Processing duplicates...
Found 3 sets of duplicates
Removed 5 duplicate bookmarks

📂 Resuming from existing state...
✅ Phase 1 already completed: Duplicates removed

🔍 Analyzing 847 bookmarks for further cleanup...
Testing sample of 25 links...
Using cached results for 12 URLs, validating 13 new URLs
🤖 Analyzing folder relationships with enhanced similarity...

📊 ENHANCED BOOKMARK CLEANUP PLAN
============================================================
Current status: 847 bookmarks
✅ Already removed 5 duplicates

📋 Additional Proposed Actions:
1. Validate all links and remove ~34 broken ones
   Based on sample, found 2/25 broken links

2. Consolidate 3 sets of semantically similar folders  
   • development: merge Dev Tools, Coding into Development
   • news: merge News & Media, Articles into News
   (LLM reasoning: "Both folders contain GitHub and Stack Overflow links with programming-related keywords")

3. Organize 125 orphaned bookmarks using semantic analysis
   Move bookmarks from root folders into semantically appropriate folders
```

### Semantic Folder Consolidation
```
📁 Using preserved folder context from Phase 1 (12 folders)
🤖 Analyzing folder relationships with LLM...
✅ LLM analysis completed successfully

🤖 LLM ANALYSIS RESPONSE
==================================================

📁 FOLDER CONSOLIDATIONS (3):
1. Merge Dev Tools, Programming → Development
   Confidence: 85%
   Reasoning: Both contain GitHub and Stack Overflow links with significant semantic overlap

2. Merge News & Media, Articles → News  
   Confidence: 78%
   Reasoning: Clear thematic overlap in news and journalism content

🌳 PROPOSED FOLDER STRUCTURE PREVIEW
==================================================

📊 CURRENT STRUCTURE (12 folders):
📁 Development/ (45 bookmarks)
   📄 React Documentation
   📄 Python Tutorial
   📄 Git Commands Guide
   ... and 42 more bookmarks

📁 Dev Tools/ (12 bookmarks)
   📄 VS Code Extensions
   📄 NPM Packages
   ... and 10 more bookmarks

🎯 PROPOSED STRUCTURE (8 folders):
📁 Development/ (57 bookmarks)
   📄 React Documentation
   📄 Python Tutorial  
   📄 VS Code Extensions
   ... and 54 more bookmarks

============================================================
Do you approve this folder structure reorganization? (y/n):
```

### Resume Capability
```
📁 Found existing processing state: Bookmarks_processing_20241230_143022.json
Progress:
  ✅ Duplicates removed: True
  ✅ Links validated: True
  ⏳ Folders reorganized: False
  📊 5 duplicates removed
  📊 Current bookmark count: 823

Resume from this state? (y)es / (n)o - restart from original:
```

### Final Summary
```
✅ Enhanced cleanup complete!
📁 Processing state saved: Bookmarks_processing_20241230_143022.json
📄 Final bookmarks saved: Bookmarks_final_20241230_143022.json

📊 Summary:
  Original bookmarks: 852
  Final bookmarks: 813
  Total removed: 39
    - Duplicates: 5
    - Broken links: 34
```

## LLM Integration

The tool now includes **LLM integration** for bookmark analysis.

### LLM Configuration

**Using .env file (Recommended):**
```bash
# Copy the example and add your API key
cp .env.example .env
# Edit .env with your API key
```

**.env file format:**
```bash
# Choose your provider and add API key
OPENAI_API_KEY=sk-your-openai-api-key-here
# OR
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Optional: Configuration
BOOKMARK_LLM_PROVIDER=openai
BOOKMARK_LLM_MODEL=gpt-4.1  # Can't use a faster model. Need larger context size.
BOOKMARK_LLM_TEMPERATURE=0.1
BOOKMARK_LLM_MAX_TOKENS=2000
```

**Or using Environment Variables:**
```bash
export OPENAI_API_KEY="your-openai-key"
export BOOKMARK_LLM_PROVIDER="openai"
```

**Command Line Options:**
```bash
# Override provider/model for single run
./bookmark_manager.py Bookmarks --create-plan \
  --llm-provider openai \
  --llm-model gpt-4 \
  --llm-template custom_template.txt
```

Default template found in llm_template.txt

## License

MIT License - see LICENSE file for details.
