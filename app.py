# FLOW:
# 1) CREATE INGEST.PY CREATE THE CHROMA AND DB_NAME
# 2) CREATE ANSWER.PY AND CREATE THE SYSTEM PROMPT AND MAKE THE RAG
# 3) CREATE APP.PY WHICH WILL DIRECTLY CALL THE answer_question FUNCTION FROM ANSWER.PY


import gradio as gr
from dotenv import load_dotenv

from implementation.answer import DB_NAME, answer_question, message_content_to_str, normalize_message_dict
load_dotenv(override=True)

# This helper function is designed to take the "raw" data found by your retriever and 
# turn it into a clean, readable HTML string. This is usually done so that when the RAG
#  system gives you an answer, it can also show you exactly where that information came 
#  from in a pretty format.


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result

# history is a list of messages in the Gradio chat, each like {"role": "user" | "assistant", "content": "..."}.
# Takes the last user message as question and the previous ones as history.
# Calls answer_question(last_message, prior) from implementation.answer.
# Appends the answer to history and returns:
# Updated chat history (for display in the chatbot).
# Formatted HTML of the retrieved context for a side panel.

def chat(history):
    last = history[-1]
    last_message = message_content_to_str(last.get("content") if isinstance(last, dict) else last)
    prior = [
        normalize_message_dict(dict(m)) if isinstance(m, dict) else m for m in history[:-1]
    ]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="ACCENTURE Expert Assistant") as ui:
        gr.Markdown("# 🏢 ACCENTURE Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                    buttons=["copy"],
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about ACCENTURE...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
