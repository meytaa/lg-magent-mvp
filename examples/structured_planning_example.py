#!/usr/bin/env python3
"""
Example demonstrating the new structured planning with dynamic node discovery and reasoning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lg_magent_mvp.nodes import BaseNode
from lg_magent_mvp.nodes.plan import PlanNode, PlanResponse

def demonstrate_dynamic_discovery():
    """Show how to dynamically discover available nodes."""
    print("=== Dynamic Node Discovery ===")
    
    # Get all available nodes
    available_nodes = BaseNode.get_all_available_nodes()
    node_descriptions = BaseNode.get_node_descriptions()
    
    print(f"Found {len(available_nodes)} available nodes:")
    for node_name in available_nodes:
        description = node_descriptions.get(node_name, "No description available")
        print(f"  • {node_name}: {description}")
    
    return available_nodes, node_descriptions

def demonstrate_structured_planning():
    """Show how the new structured planning works."""
    print("\n=== Structured Planning with Reasoning ===")
    
    # Create a plan node
    planner = PlanNode()
    print(f"Created planner node: {planner.node_name}")
    
    # Example state with document summary
    example_state = {
        "question": "What are the compliance risks in this medical facility audit?",
        "doc_summary": {
            "pages": 15,
            "counts": {
                "tables": 4,    # Document has tables
                "figures": 2    # Document has figures
            }
        },
        "notes": []
    }
    
    print(f"\nExample question: {example_state['question']}")
    print(f"Document summary: {example_state['doc_summary']['pages']} pages, "
          f"{example_state['doc_summary']['counts']['tables']} tables, "
          f"{example_state['doc_summary']['counts']['figures']} figures")
    
    # Note: In a real scenario, this would call the LLM
    # For this example, we'll show what the structured response would look like
    example_plan_response = PlanResponse(
        reasoning=(
            "Based on the document summary showing 4 tables and 2 figures, "
            "I recommend a comprehensive approach: start with keyword and semantic search "
            "to identify key compliance issues, then extract tables for detailed data analysis, "
            "analyze figures for visual compliance indicators, and finalize with a structured report."
        ),
        steps=["keyword_search", "semantic_search", "extract_tables", "analyze_figures", "finalize"],
        estimated_priority="high"
    )
    
    print(f"\nExample structured plan response:")
    print(f"  Reasoning: {example_plan_response.reasoning}")
    print(f"  Steps: {example_plan_response.steps}")
    print(f"  Priority: {example_plan_response.estimated_priority}")
    
    return example_plan_response

def demonstrate_schema_validation():
    """Show how the Pydantic schema validates responses."""
    print("\n=== Schema Validation ===")
    
    try:
        # Valid response
        valid_response = PlanResponse(
            reasoning="This is a valid reasoning",
            steps=["keyword_search", "finalize"],
            estimated_priority="medium"
        )
        print("✅ Valid response created successfully")
        
        # Show serialization
        response_dict = valid_response.model_dump()
        print(f"✅ Serialized to dict: {list(response_dict.keys())}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    try:
        # Invalid response (missing required field)
        invalid_response = PlanResponse(
            steps=["keyword_search"],
            estimated_priority="high"
            # Missing 'reasoning' field
        )
        print("❌ This should have failed validation")
        
    except Exception as e:
        print(f"✅ Correctly caught validation error: {e}")

def demonstrate_filtering():
    """Show how plans are filtered based on document content."""
    print("\n=== Plan Filtering Based on Document Content ===")
    
    # Document with no tables or figures
    empty_doc_summary = {
        "pages": 5,
        "counts": {
            "tables": 0,
            "figures": 0
        }
    }
    
    # Original plan with table and figure analysis
    original_steps = ["keyword_search", "semantic_search", "extract_tables", "analyze_figures", "finalize"]
    
    # Simulate filtering logic
    counts = empty_doc_summary["counts"]
    filtered_steps = original_steps.copy()
    
    if counts.get("tables", 0) == 0:
        filtered_steps = [s for s in filtered_steps if s != "extract_tables"]
    if counts.get("figures", 0) == 0:
        filtered_steps = [s for s in filtered_steps if s != "analyze_figures"]
    
    print(f"Original plan: {original_steps}")
    print(f"Document content: {counts['tables']} tables, {counts['figures']} figures")
    print(f"Filtered plan: {filtered_steps}")
    print(f"Removed steps: {set(original_steps) - set(filtered_steps)}")

def main():
    """Run all demonstrations."""
    print("Structured Planning with Dynamic Node Discovery")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_dynamic_discovery()
    demonstrate_structured_planning()
    demonstrate_schema_validation()
    demonstrate_filtering()
    
    print("\n" + "=" * 60)
    print("🎉 All demonstrations completed!")
    print("\nKey Features:")
    print("  • Dynamic node discovery from BaseNode subclasses")
    print("  • Structured output with Pydantic schemas")
    print("  • Reasoning included in planning responses")
    print("  • Automatic filtering based on document content")
    print("  • Backward compatibility with existing code")

if __name__ == "__main__":
    main()
