"""
Message helper module for creating SystemMessage and HumanMessage instances.

This module centralizes message creation to avoid repetitive imports and
message construction patterns across the codebase.

Usage Examples:
    # Router messages
    messages = create_router_messages(state_snapshot)

    # Planner messages
    messages = create_planner_messages(
        question="What are the key findings?",
        doc_summary={"pages": 10, "tables": 3},
        available_nodes=["keyword_search", "semantic_search"],
        node_descriptions={"keyword_search": "Search for keywords"}
    )

    # Finalize messages
    messages = create_finalize_messages("Question", ["evidence1", "evidence2"])

    # Custom messages with variable substitution
    messages = create_custom_messages(
        "You are a {role} assistant",
        "Please analyze {content}",
        role="medical",
        content="this document"
    )
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, Any, Optional, List, Union


def create_router_messages(snapshot: Any) -> List[SystemMessage | HumanMessage]:
    """
    Create system and user messages for routing controller.
    
    Args:
        snapshot: The state snapshot to include in the user message
        
    Returns:
        List containing SystemMessage and HumanMessage for routing
    """
    sys = SystemMessage(content=(
        "You are a routing controller. Return ONLY one label from the allowed list. "
        "Prefer the hint when reasonable. If enough evidence is gathered, choose 'finalize'."
    ))
    user = HumanMessage(content=str(snapshot))
    return [sys, user]


def create_planner_messages(
    question: str,
    doc_summary: Any,
    available_nodes: List[str],
    node_descriptions: Dict[str, str]
) -> List[SystemMessage | HumanMessage]:
    """
    Create system and user messages for execution planning.

    Args:
        question: The audit question to address
        doc_summary: Summary of the document being audited (dict or string)
        available_nodes: List of available analysis nodes
        node_descriptions: Mapping of node names to descriptions

    Returns:
        List containing SystemMessage and HumanMessage for planning
    """
    # Convert doc_summary to string if it's a dict
    doc_summary_str = str(doc_summary) if not isinstance(doc_summary, str) else doc_summary
    # Build context about available nodes
    node_context = "\n".join([f"- {node}: {node_descriptions.get(node, 'No description')}"
                             for node in available_nodes])

    sys = SystemMessage(content=(
        f"You are an intelligent execution planner for comprehensive medical PDF document auditing. "
        f"Your role is to analyze audit requirements and create strategic, step-by-step execution plans "
        f"using the available specialized analysis nodes.\n\n"

        f"AVAILABLE ANALYSIS NODES:\n{node_context}\n\n"

        f"PLANNING GUIDELINES:\n"
        f"• Always start with 'summarize' to understand document structure\n"
        f"• Use 'keyword_search' for specific terms, conditions, or regulatory requirements\n"
        f"• Use 'semantic_search' for conceptual queries and evidence gathering\n"
        f"• Use 'extract_tables' when numerical data, statistics, or structured information is needed\n"
        f"• Use 'analyze_figures' for visual content analysis (charts, diagrams, images)\n"
        f"• Consider document type and structure when selecting analysis approaches\n"
        f"• Prioritize nodes that directly address the audit question\n"
        f"• Plan for comprehensive coverage while avoiding redundancy\n\n"

        f"MEDICAL AUDIT FOCUS AREAS:\n"
        f"• Patient safety and adverse events\n"
        f"• Treatment protocols and clinical guidelines adherence\n"
        f"• Documentation completeness and accuracy\n"
        f"• Regulatory compliance (FDA, clinical trial protocols, etc.)\n"
        f"• Quality metrics and outcome measurements\n"
        f"• Risk assessment and mitigation strategies\n"
        f"• Statistical data analysis for clinical trials/studies\n"
        f"• Regulatory compliance requirements\n"
        f"• Evidence-based findings extraction\n"
    ))

    user = HumanMessage(content=(
        f"Please create an execution plan for the following medical document audit:\n\n"
        f"AUDIT QUESTION:\n{question}\n\n"
        f"DOCUMENT SUMMARY:\n{doc_summary_str}\n\n"
        f"Based on this information, develop a comprehensive audit plan that addresses "
        f"the question while leveraging the document's specific content types and structure. "
        f"Focus on nodes that will provide the most relevant evidence and insights."
    ))

    return [sys, user]


def create_finalize_messages(question: str, evidence: List[Any]) -> List[SystemMessage | HumanMessage]:
    """
    Create system and user messages for finalizing analysis.
    
    Args:
        question: The original audit question
        evidence: List of evidence gathered during analysis
        
    Returns:
        List containing SystemMessage and HumanMessage for finalization
    """
    sys = SystemMessage(content=(
        "You are auditing a medical PDF. Write a concise executive summary (4-6 sentences) "
        "that highlights key risks, overall documentation quality, and top 3-5 prioritized actions. "
        "Ground your narrative strictly in the given evidence. Do not invent citations."
    ))
    user = HumanMessage(content=f"Question: {question}\nEvidence: {evidence}")
    return [sys, user]


def create_vision_messages() -> List[SystemMessage | HumanMessage]:
    """
    Create system message for vision analysis.
    
    Returns:
        List containing SystemMessage for vision analysis
    """
    sys = SystemMessage(content=(
        "You are a careful medical document figure summarizer. "
        "Return 1-2 sentences describing the figure and 1 short line of notable observation if any."
    ))
    return [sys]


def create_custom_messages(
    system_content: str,
    user_content: Optional[str] = None,
    **format_vars: Any
) -> List[SystemMessage | HumanMessage]:
    """
    Create custom system and user messages with variable substitution.
    
    Args:
        system_content: Content for the system message (supports format strings)
        user_content: Optional content for the user message (supports format strings)
        **format_vars: Variables to substitute in the message content
        
    Returns:
        List containing SystemMessage and optionally HumanMessage
    """
    messages = []
    
    # Format system message
    formatted_system = system_content.format(**format_vars) if format_vars else system_content
    messages.append(SystemMessage(content=formatted_system))
    
    # Add user message if provided
    if user_content:
        formatted_user = user_content.format(**format_vars) if format_vars else user_content
        messages.append(HumanMessage(content=formatted_user))
    
    return messages


# Convenience functions for common patterns
def system_message(content: str, **format_vars: Any) -> SystemMessage:
    """Create a SystemMessage with optional variable substitution."""
    formatted_content = content.format(**format_vars) if format_vars else content
    return SystemMessage(content=formatted_content)


def human_message(content: str, **format_vars: Any) -> HumanMessage:
    """Create a HumanMessage with optional variable substitution."""
    formatted_content = content.format(**format_vars) if format_vars else content
    return HumanMessage(content=formatted_content)


def create_orchestrator_messages(context: Dict[str, Any]) -> List[SystemMessage | HumanMessage | AIMessage]:
    """
    Create system and user messages for orchestrator LLM with full chat history.

    Args:
        context: Dictionary containing question, preface results, chat history, etc.

    Returns:
        List containing SystemMessage and conversation history for orchestrator
    """
    sys = SystemMessage(content=( #TODO: Softcode the agent names
        "You are the central orchestrator for a medical document audit system. "
        "Your role is to analyze the current state and decide what actions to take next.\n\n"

        "AVAILABLE AGENTS:\n"
        "- keyword_search: Search for specific terms/keywords in the document\n"
        "- semantic_search: Find semantically related content using embeddings\n"
        "- extract_tables: Extract detailed data and analyze tables from the document\n"
        "- analyze_figures: Analyze figures, charts, and images and get detailed insights\n"
        "- approval: Request human approval before proceeding\n"
        "- finalize: Complete the analysis and generate final report\n\n"

        "DECISION PROCESS:\n"
        "1. Analyze the question and what information is needed\n"
        "2. Review preface results (document summary, metadata)\n"
        "3. Review the full conversation history of previous decisions and agent results\n"
        "4. Determine if more information is needed/available or if ready to finalize\n"
        "5. If more info needed, select the most appropriate agent with specific parameters\n\n"

        "RESPONSE FORMAT:\n"
        "You must respond with a structured JSON object containing:\n"
        "- thoughts: Your reasoning and analysis of the current situation\n"
        "- agent_to_call: Either null (if ready for final answer) or an object with:\n"
        "  - name: The agent name from the available list\n"
        "  - input_arguments: Parameters specific to that agent\n"
        "- final_answer: (optional) Only when process_complete is true\n"
        "- process_complete: (optional) Set to true when providing final answer\n\n"

        "AGENT PARAMETERS:\n"
        "- keyword_search: {\"keywords\": [\"list\", \"of\", \"terms\"], \"k\": max_results}\n"
        "- semantic_search: {\"query\": \"search query\", \"k\": max_results}\n"
        "- extract_tables: {\"table_filter\": [\"T1-1\", \"T2-3\"]} (optional)\n"
        "- analyze_figures: {\"figure_ids\": [\"fig_page1_1\", \"fig_page2_1\"]} (required - specify figure IDs to analyze)\n"
        "- approval: {} (no parameters needed)\n"
        "- finalize: {\"conclusion_obtained\": true/false, \"confidence\": 1-10}\n\n"

        "FINALIZATION PROCESS:\n"
        "1. When you have sufficient evidence, call 'finalize' agent with confidence level\n"
        "2. Finalize agent will process all evidence and return comprehensive analysis\n"
        "3. After receiving finalize results, provide final_answer and set process_complete=true\n"
        "4. The final_answer should be your interpretation of the finalize agent's analysis\n\n"

        "DECISION LOGIC:\n"
        "- If more investigation is needed: select appropriate agents\n"
        "- if you thing that there might be more relevant information in the document, continue untill you are confident that you have all the relevant information\n"
        "- If you have sufficient evidence for a comprehensive answer: call finalize agent\n"
        "- If you are confident that there is not relevant information to answer the uestion, finalize without calling finalize agent and low confidence\n"
        "- If finalize results are received: provide final_answer and set process_complete=true\n"
        "- If you have enough information but don't need comprehensive analysis: provide final_answer directly and set process_complete=true\n\n"

        "IMPORTANT: You must ALWAYS provide a final_answer when setting process_complete=true. "
        "Never end the process without providing a final answer to the user's question."
    ))

    # Build the full conversation history
    messages: List[SystemMessage | HumanMessage | AIMessage] = [sys]

    # Get context data
    question = context.get("question", "")
    preface = context.get("preface_results", {})
    chat_history = context.get("orchestrator_chat_history", [])

    # Start with the original user question (first message in conversation)
    initial_context = f"QUESTION: {question}\n\n"

    if preface:
        doc_summary = preface.get("doc_summary", {})
        initial_context += f"DOCUMENT SUMMARY:\n"
        initial_context += f"- Pages: {doc_summary.get('pages', 0)}\n"
        initial_context += f"- Sections: {', '.join(doc_summary.get('sections', []))}\n"

        # Add text information
        counts = doc_summary.get('counts', {})
        text_words = counts.get('text_words', 0)
        text_blocks = counts.get('text_blocks', 0)
        initial_context += f"- Text Content: {text_words:,} words in {text_blocks} blocks\n"

        # Add text by section if available
        text_by_section = doc_summary.get('text_by_section', {})
        if text_by_section:
            initial_context += f"- Text by Section:\n"
            for section, stats in text_by_section.items():
                words = stats.get('words', 0)
                blocks = stats.get('blocks', 0)
                initial_context += f"  * {section}: {words:,} words in {blocks} blocks\n"

        # Add detailed table information
        tables = doc_summary.get('tables', [])
        initial_context += f"- Tables ({len(tables)}):\n"
        if tables:
            for table in tables:
                initial_context += f"  * {table.get('table_id', 'unknown')}: {table.get('header', 'No header')} (Page {table.get('page', '?')})\n"
        else:
            initial_context += f"  * No tables found\n"

        # Add detailed figure information
        figures = doc_summary.get('figures', [])
        initial_context += f"- Figures ({len(figures)}):\n"
        if figures:
            for figure in figures:
                initial_context += f"  * {figure.get('figure_id', 'unknown')}: {figure.get('caption', 'No caption')} (Page {figure.get('page', '?')})\n"
        else:
            initial_context += f"  * No figures found\n"

        initial_context += "\n"

    initial_context += f"Current status:\n"
    initial_context += f"- Hop count: {context.get('hops', 0)}\n"
    initial_context += f"- Current findings: {context.get('current_findings', 0)}\n"
    initial_context += f"- Current evidence: {context.get('current_evidence', 0)}\n\n"

    initial_context += "What should be the first action?"

    # Add the initial user message
    messages.append(HumanMessage(content=initial_context))

    # Add the full chat history (alternating assistant decisions and user agent results)
    for chat_entry in chat_history:
        if chat_entry.get("role") == "assistant":
            # This is an orchestrator decision
            messages.append(AIMessage(content=chat_entry["content"]))
        elif chat_entry.get("role") == "user":
            # This is an agent result
            messages.append(HumanMessage(content=chat_entry["content"]))

    # If this is the first call (no chat history), ask for the first action
    # If we have chat history, the last message should prompt for the next action
    if not chat_history:
        # This is the first orchestrator call, the initial context already asks "What should be the first action?"
        pass
    else:
        # We have chat history, check if the last message needs a prompt for next action
        last_entry = chat_history[-1] if chat_history else None
        if last_entry and last_entry.get("role") == "user":
            # Last entry was an agent result, we don't need to add anything more
            # The agent result formatting in BaseNode already includes "What should be the next action?"
            pass

    return messages


def create_summary_messages(page_num: int, image_base64: str) -> List[SystemMessage | HumanMessage]:
    """Create messages for summary LLM analysis using LangChain's proper image message format."""

    system_content = f"""
    You are analyzing a complete document page image (PAGE {page_num}). Your task is to extract all content and provide a structured analysis.

    STRICT FORMATTING RULES:
    1. type field must be EXACTLY: "text", "image", or "table" (lowercase, no other variations)
    2. For text: content is a simple string
    3. For images/tables: content is an object with id, description, caption
    4. This is PAGE {page_num} - ALL IDs must use page{page_num}
    5. Image IDs: "fig_page{page_num}_1", "fig_page{page_num}_2", etc.
    6. Table IDs: "table_page{page_num}_1", "table_page{page_num}_2", etc.

    Instructions:
    - Extract ALL content from the page in proper reading order (top to bottom, left to right)
    - For text: Extract the actual text content as a string
    - For images: Provide description and caption (explicit or generated)
    - For tables: Provide description and caption/title
    - For section: Try to identify if this page belongs to a specific section (e.g., "Patient Information", "Treatment Plan", "Assessment", etc.)
    - Bounding boxes will be provided by the PDF parser, not by you
    - If any kind of image exists in the page, you MUST provide an image entry in the JSON array even if it's just strings

    Analyze the complete page image and extract all content following the structured format.
    """

    # Create system message
    system_msg = SystemMessage(content=system_content)

    # Create human message with image using LangChain's proper format
    human_msg = HumanMessage(
        content=[
            {"type": "text", "text": f"Analyze this page {page_num} image and extract all content in the structured format."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        ]
    )

    return [system_msg, human_msg]


def create_figure_analysis_messages(question: str, orchestrator_thoughts: str, figures: List[Dict[str, Any]], doc_path: str) -> List[SystemMessage | HumanMessage]:
    """Create messages for figure analysis LLM using LangChain's proper image message format."""
    import base64
    import fitz  # PyMuPDF
    from PIL import Image
    import io

    system_content = f"""
    You are analyzing medical document figures to answer a specific question. Your task is to provide detailed analysis of each figure and how it relates to the question.

    QUESTION: {question}

    ORCHESTRATOR'S THOUGHTS: {orchestrator_thoughts}

    For each figure, provide:
    1. Detailed analysis of what the figure shows
    2. Key findings or data points visible in the figure
    3. How the figure relates to answering the question
    4. Your confidence level in the analysisTry to return maximum from image based on the question and the thoughts lead you to.
    Be thorough but focused on answering the question. Look for specific details, measurements, annotations, or patterns that are relevant.
    """

    # Create system message
    system_msg = SystemMessage(content=system_content)

    # Build user content with figure information
    user_content = []
    user_content.append({
        "type": "text",
        "text": f"Please analyze these {len(figures)} figures in the context of the question: '{question}'"
    })

    for i, figure in enumerate(figures, 1):
        # Add figure metadata
        fig_text = f"\n--- FIGURE {i}: {figure.get('id', 'unknown')} ---\n"
        fig_text += f"Description: {figure.get('description', 'No description')}\n"
        fig_text += f"Caption: {figure.get('caption', 'No caption')}\n"
        fig_text += f"Page: {figure.get('page', 'Unknown')}\n"
        fig_text += f"Section: {figure.get('section', 'Unknown')}\n"
        fig_text += f"Page Context: {figure.get('page_text', '')[:500]}...\n"

        user_content.append({"type": "text", "text": fig_text})

        # Extract figure image directly from PDF using bbox and page number
        page_num = figure.get("page")
        bbox = figure.get("bbox")
        if page_num and bbox and doc_path:
            try:
                # Open PDF and get the specific page
                doc = fitz.open(doc_path)
                page = doc.load_page(page_num - 1)  # fitz uses 0-based indexing

                # Create rectangle from bbox [x0, y0, x1, y1]
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

                # Get pixmap of the specific region
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)  # 2x scaling for better quality

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))

                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Add image to content
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })

                doc.close()
                print(f"✅ Extracted image for {figure.get('id', 'unknown')} from PDF page {page_num}")

            except Exception as e:
                print(f"⚠️ Could not extract image from PDF page {page_num}: {e}")
        else:
            print(f"⚠️ Missing page number or bbox for figure {figure.get('id', 'unknown')}")

    # Create human message with mixed content using LangChain's proper format
    human_msg = HumanMessage(content=user_content)

    return [system_msg, human_msg]
