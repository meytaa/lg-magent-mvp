# LG-MAgent MVP — Intelligent PDF Analysis System

A sophisticated LangGraph-based multi-agent system for intelligent PDF document analysis with orchestrator-driven architecture.

## 🚀 Features

- **🧠 Orchestrator-based Architecture**: Central LLM brain makes intelligent routing decisions based on document content and question context
- **📊 Multi-modal Analysis**: Advanced text extraction, figure analysis, and table processing with vision models
- **⚡ Smart Caching System**: Efficient caching for summaries and indexing with automatic invalidation
- **🎯 Structured Output**: Consistent JSON responses across all agents with confidence scoring
- **💾 Memory Support**: Optional conversation persistence with LangGraph checkpointing
- **📈 Text Analytics**: Word count analysis and section-based text statistics
- **🖼️ Direct PDF Extraction**: Real-time image extraction from PDFs using bounding boxes
- **🔍 Contextual Analysis**: Question-aware figure and table analysis with orchestrator insights

## 🏗️ Architecture

The system uses an orchestrator-based architecture where a central LLM coordinates specialized agents:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Orchestrator  │───▶│  Summarize Doc   │───▶│  Analyze Figures│
│  (Central Brain)│    │ (Structure+Stats)│    │ (Vision Analysis)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Extract Tables  │    │ Semantic Search  │    │    Finalize     │
│ (Table Analysis)│    │ (Text Retrieval) │    │ (Report Gen)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Agent Responsibilities

- **🎯 Orchestrator**: Analyzes document summary and routes to appropriate agents based on question context
- **📋 Summarize**: Extracts document structure, identifies figures/tables, calculates text statistics by section
- **🖼️ Analyze Figures**: Performs detailed vision analysis of specific figures with confidence scoring
- **📊 Extract Tables**: Advanced table extraction and structured data analysis
- **🔍 Semantic/Keyword Search**: Intelligent text search with embedding-based retrieval
- **📝 Finalize**: Generates comprehensive reports with prioritized findings and recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- PDF documents to analyze

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/lg-magent-mvp.git
   cd lg-magent-mvp
   ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set Environment Variables**:
   Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   LANGSMITH_API_KEY=your-langsmith-key-here  # Optional for tracing
   ```

### Basic Usage

**Analyze a Document**:
```bash
python -m lg_magent_mvp.cli run \
  --doc "data/MM155 Deines Chiropractic-1.pdf" \
  --question "What medical conditions are shown in the spine diagrams?" \
  --out "report.json"
```

**Example Output**:
```
The spine diagram shows potential medical conditions related to neck pain, 
lower back pain, and headaches, with specific annotations indicating various 
vertebrae and conditions such as 'AS', 'PI', and 'BR'.

Report saved to report.json
```

### Advanced Usage

**With Custom Thread ID**:
```bash
python -m lg_magent_mvp.cli run \
  --doc "medical_report.pdf" \
  --question "Analyze the patient's condition based on the spine diagrams" \
  --thread "patient_123_analysis" \
  --out "patient_123_report.json"
```

**Enable Memory and Approvals**:
```bash
echo "USE_MEMORY=true" >> .env
echo "APPROVALS=pause-before-finalize" >> .env

python -m lg_magent_mvp.cli run \
  --doc "data/MM155 Deines Chiropractic-1.pdf" \
  --question "What conditions are shown?" \
  --thread "demo"

# System pauses for approval, then resume:
python -m lg_magent_mvp.cli approve --thread "demo"
```

## 📊 Report Structure

The system generates comprehensive JSON reports with:

```json
{
  "report_meta": {
    "timestamp": "2025-09-05T23:14:56.172013Z",
    "file": "data/MM155 Deines Chiropractic-1.pdf",
    "pages": 4,
    "model_versions": {
      "router_model": "gpt-4o-mini",
      "finalize_model": "gpt-4o",
      "vision_model": "gpt-4o"
    }
  },
  "narrative": "Executive summary with prioritized findings...",
  "findings": [],
  "tables": [],
  "figures": [],
  "metrics": {
    "counts": {"critical": 0, "major": 0, "minor": 0},
    "total_findings": 0
  }
}
```

## 🔧 Configuration

Configure via `config.yaml` or environment variables:

- **Model Selection**: Choose between GPT-4, GPT-4o, Claude, etc.
- **Caching**: Enable/disable summary and indexing caches
- **Memory**: Persistent conversation state
- **Vision Models**: Configure image analysis models
- **Tracing**: LangSmith integration for debugging

## 🧪 Example Workflows

### Medical Document Analysis
```bash
python -m lg_magent_mvp.cli run \
  --doc "patient_chart.pdf" \
  --question "What are the key findings and recommendations?" \
  --out "medical_analysis.json"
```

### Research Paper Analysis
```bash
python -m lg_magent_mvp.cli run \
  --doc "research_paper.pdf" \
  --question "Summarize the methodology and key results from the figures" \
  --out "research_summary.json"
```

### Financial Report Analysis
```bash
python -m lg_magent_mvp.cli run \
  --doc "quarterly_report.pdf" \
  --question "What are the key financial metrics and trends shown in the tables?" \
  --out "financial_analysis.json"
```

## 🚀 Key Improvements

This system includes several advanced features:

1. **Smart Document Understanding**: Analyzes document structure and provides detailed statistics (word counts, sections, figures)
2. **Context-Aware Routing**: Orchestrator makes intelligent decisions based on document content and question type
3. **Direct PDF Processing**: Extracts images directly from PDFs using bounding boxes for accurate analysis
4. **Structured Analysis**: All agents return structured data with confidence scores and detailed insights
5. **Efficient Caching**: Smart caching system prevents redundant processing
6. **Comprehensive Reporting**: Generates detailed reports with executive narratives and prioritized actions

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
