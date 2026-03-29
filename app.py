# ============================================================
# app.py — Gradio Web UI for Course Planning Assistant
# ============================================================
"""
A polished Gradio interface for the Prerequisite & Course Planning Assistant.
Features:
- Chat-based interaction
- Expandable debug panels for each agent's output
- Real-time pipeline status
"""

import gradio as gr
import logging
import json
import traceback
from typing import List, Tuple

from pipeline import CoursePlanningPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance (initialized once)
pipeline = None


def initialize_pipeline():
    """Initialize or return the existing pipeline."""
    global pipeline
    if pipeline is None:
        logger.info("Initializing pipeline...")
        pipeline = CoursePlanningPipeline()
        logger.info("Pipeline ready!")
    return pipeline


def process_message(
    user_message: str,
    chat_history: list,
):
    """
    Process a user message and return results.
    
    Returns:
        - Updated chat history (Gradio 6 message format)
        - Intake analysis (JSON)
        - Retrieval info
        - Verification summary
        - Raw planner output
    """
    if not user_message.strip():
        return chat_history, "", "", "", ""
    
    try:
        pipe = initialize_pipeline()
        result = pipe.process_query(user_message)
        
        # Build assistant response
        assistant_response = result["final_output"]
        
        # Update chat history (Gradio 6 message format)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": assistant_response})
        
        # Debug panels
        intake_debug = json.dumps(result.get("intake", {}), indent=2)
        
        retrieval_info = ""
        if "retrieval" in result:
            r = result["retrieval"]
            retrieval_info = f"**Chunks Retrieved:** {r.get('total_chunks', 0)}\n\n"
            retrieval_info += "**Citations Available:**\n"
            for cite in r.get("citations", []):
                retrieval_info += f"- {cite}\n"
        
        verification_info = result.get("verification", {}).get("summary", "N/A")
        
        planner_raw = ""
        if "planner" in result:
            planner_raw = json.dumps(result["planner"], indent=2, default=str)
        
        return chat_history, intake_debug, retrieval_info, verification_info, planner_raw
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, str(e), "", "", ""


def clear_chat():
    """Clear the chat history and all debug panels."""
    return [], "", "", "", ""


# ── Build UI ──────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "Can I take CS301 if I have completed CS201 and CS202?",
    "I have completed CS101, CS102, CS201, CS202, and MATH120. What courses should I take next?",
    "What are the prerequisites for Machine Learning (CS305)?",
    "Am I eligible for CS302 Database Systems if I only have CS102?",
    "What courses do I need for the AI Specialization program?",
    "I have CS101 with a grade of D. Can I take CS102?",
    "Plan my next semester: I've done CS101, CS102, MATH120. I'm in BSc Computer Science.",
    "What are the requirements for CS Minor?",
    "Can I take Cybersecurity (CS308) if I have CS304 but not CS201?",
    "Is there a course on Quantum Computing? What are its prerequisites?",
]


def build_app():
    """Build the Gradio application."""
    
    with gr.Blocks(
        title="🎓 Course Planning Assistant"
    ) as app:
        
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>🎓 Prerequisite & Course Planning Assistant</h1>
                <p>Agentic RAG System — Ask about prerequisites, plan your courses, check eligibility</p>
            </div>
        """)
        
        with gr.Row():
            # ── Left: Chat ─────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 Chat",
                    height=500,
                    show_label=True,
                )
                
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask about prerequisites, course planning, or program requirements...",
                        label="Your Question",
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear 🗑️", scale=1)
                
                # Example queries
                gr.Markdown("### 💡 Example Queries")
                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUERIES],
                    inputs=user_input,
                    label="Click to try:",
                )
            
            # ── Right: Debug Panels ────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Pipeline Debug")
                
                with gr.Accordion("🔍 Intake Agent Output", open=False):
                    intake_output = gr.Code(
                        label="Parsed Query",
                        language="json",
                        lines=10,
                    )
                
                with gr.Accordion("📚 Retriever Agent Output", open=False):
                    retrieval_output = gr.Markdown(
                        label="Retrieved Chunks",
                    )
                
                with gr.Accordion("✅ Verifier Agent Output", open=True):
                    verification_output = gr.Markdown(
                        label="Verification Summary",
                    )
                
                with gr.Accordion("🧠 Planner Agent Raw", open=False):
                    planner_output = gr.Code(
                        label="Raw Planner Output",
                        language="json",
                        lines=15,
                    )
        
        # ── Event Handlers ─────────────────────────────────
        outputs = [chatbot, intake_output, retrieval_output, verification_output, planner_output]
        
        send_btn.click(
            fn=process_message,
            inputs=[user_input, chatbot],
            outputs=outputs,
        ).then(
            fn=lambda: "",
            outputs=user_input,
        )
        
        user_input.submit(
            fn=process_message,
            inputs=[user_input, chatbot],
            outputs=outputs,
        ).then(
            fn=lambda: "",
            outputs=user_input,
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=outputs,
        )
    
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        ),
        css="""
        .main-header { 
            text-align: center; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .main-header h1 { color: white; margin: 0; }
        .main-header p { color: rgba(255,255,255,0.8); margin: 5px 0 0 0; }
        """
    )
