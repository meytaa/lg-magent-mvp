# Node Architecture

This directory contains all the LangGraph nodes for the medical PDF auditing system. All nodes now inherit from a common `BaseNode` class that provides standardized functionality.

## BaseNode Class

The `BaseNode` class provides:

- **Standardized error handling**: Catches exceptions and logs them gracefully
- **Common state management**: Helper methods for adding notes, evidence, and validating state
- **Consistent logging**: Automatic tracking of node execution
- **Validation utilities**: Methods to check required fields and document paths

## Node Structure

All nodes follow this pattern:

```python
from . import BaseNode

class MyCustomNode(BaseNode):
    def __init__(self):
        # Used for logging/state; provide a human-readable description
        super().__init__(
            "my_node_name",
            description="Describe what this node does in one sentence."
        )
    
    def execute(self, state: "AgentState") -> "AgentState":
        # Validate required fields
        if not self._validate_required_fields(state, ["required_field"]):
            return state
        
        # Get document path with validation
        doc_path = self._get_doc_path(state)
        if not doc_path:
            return state
        
        # Your node logic here
        result = do_something(state["required_field"], doc_path)
        
        # Add evidence and notes
        self._add_evidence(state, "result_type", result)
        self._add_note(state, f"Processed {len(result)} items")
        
        return state

# Create instance for backward compatibility
my_custom_node = MyCustomNode()
```

## Available Helper Methods

### `_add_note(state, note)`
Adds a note to the state's notes list.

### `_add_evidence(state, evidence_type, data)`
Adds evidence to the state's evidence list with the specified type.

### `_get_doc_path(state)`
Gets and validates the document path from state. Returns None if not found and logs an error.

### `_validate_required_fields(state, fields)`
Validates that all required fields are present in the state. Returns False and logs missing fields if validation fails.

## Dynamic Node Discovery

The BaseNode class provides methods to dynamically discover all available nodes:

### `BaseNode.get_all_available_nodes()`
Returns a list of all available node names from registered BaseNode subclasses.

```python
from lg_magent_mvp.nodes import BaseNode

available_nodes = BaseNode.get_all_available_nodes()
# Returns: ['keyword_search', 'semantic_search', 'extract_tables', ...]
```

### `BaseNode.get_node_descriptions()`
Returns a dictionary mapping node names to their descriptions. Descriptions are read
from each node instance's `description` attribute. If a node does not set this
attribute, the first line of its class docstring is used as a fallback.

```python
descriptions = BaseNode.get_node_descriptions()
# Returns: {'keyword_search': 'Node for performing keyword-based search.', ...}
```

## Structured Output with Reasoning

The planning system now uses structured output with Pydantic schemas to ensure consistent responses with reasoning:

```python
from pydantic import BaseModel, Field

from enum import Enum

class StepName(str, Enum):
    keyword_search = "keyword_search"
    semantic_search = "semantic_search"
    extract_tables = "extract_tables"
    analyze_figures = "analyze_figures"
    finalize = "finalize"

class PlanResponse(BaseModel):
    reasoning: str = Field(description="Explanation of why these steps were chosen")
    steps: List[StepName] = Field(description="Ordered list of allowed steps to execute")
    estimated_priority: str = Field(description="Priority level: 'high', 'medium', or 'low'")

# Usage in nodes
structured_llm = llm.with_structured_output(PlanResponse)
response = structured_llm.invoke([system_msg, user_msg])
```

## Existing Nodes

All existing nodes have been refactored to use the BaseNode pattern:

- **SummarizeDocNode** (`summarize`): Creates document summaries
- **PlanNode** (`planner`): Creates execution plans
- **KeywordSearchNode** (`keyword_search`): Performs keyword-based search
- **SemanticSearchNode** (`semantic_search`): Performs vector-based search
- **ExtractTablesNode** (`extract_tables`): Extracts tables from PDFs
- **AnalyzeFiguresNode** (`analyze_figures`): Analyzes figures in PDFs
- **ApprovalNode** (`approval`): Handles approval workflows
- **FinalizeNode** (`finalize`): Creates final reports

## Backward Compatibility

All existing function-based node interfaces are preserved through instance variables:

```python
# These still work exactly as before
from lg_magent_mvp.nodes.vision import analyze_figures_node
from lg_magent_mvp.nodes.search import keyword_search_node, semantic_search_node
# etc.
```

## Creating New Nodes

To create a new node:

1. Create a new class inheriting from `BaseNode`
2. Implement the `execute` method
3. Create an instance for backward compatibility
4. Add the node to your graph in `app.py`

Example:

```python
# In lg_magent_mvp/nodes/my_new_node.py
from . import BaseNode

class MyNewNode(BaseNode):
    def __init__(self):
        super().__init__("my_new_node")
    
    def execute(self, state):
        # Your logic here
        return state

# Backward compatibility
my_new_node = MyNewNode()
```

Then in `app.py`:
```python
from .nodes.my_new_node import my_new_node

# Add to graph
g.add_node("my_new_node", with_hop(my_new_node))
```
