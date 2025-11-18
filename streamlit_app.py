import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

# Ensure backend modules are importable when running `streamlit run streamlit_app.py`
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from services import chat_service  # type: ignore  # noqa: E402


def init_session_state() -> None:
    """Initialize Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state["messages"]: List[Dict[str, str]] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"]: Optional[str] = None


def render_sidebar() -> Dict[str, Any]:
    """Render sidebar controls and return current configuration."""
    st.sidebar.title("ChatMyCV Settings")

    lang = st.sidebar.selectbox("Language", options=["en", "zhtw"], index=0)
    character_label = st.sidebar.selectbox(
        "Interviewer persona",
        options=["Default", "HR", "Engineering"],
        index=0,
        help="Controls the system prompt used for responses.",
    )
    if character_label == "Default":
        character: Optional[str] = None
    elif character_label == "HR":
        character = "hr"
    else:
        character = "engineer"

    system_prompt = st.sidebar.text_area(
        "Custom system prompt (optional)",
        value="",
        height=140,
        help="Override the default/character prompt. Leave empty to use defaults.",
    )

    if st.sidebar.button("Clear conversation", use_container_width=True):
        # Clear local history
        st.session_state["messages"] = []

        # Also clear backend conversation store if we have a session_id
        session_id = st.session_state.get("session_id")
        if session_id:
            try:
                chat_service.clear_history(session_id)  # type: ignore[attr-defined]
            except Exception:
                # If clearing fails, we still reset local state
                pass

        st.session_state["session_id"] = None

    return {
        "lang": lang,
        "character": character,
        "system_prompt": system_prompt or None,
    }


def render_chat_ui(config: Dict[str, Any]) -> None:
    """Render the main chat interface."""
    st.title("ChatMyCV")
    st.write(
        "I invite you to explore and get to know my professional journey better. "
        "Leverage the power of our Streamlit interface to chat with my CV/resume and uncover more about my experiences and abilities."
    )

    # Show existing messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask the candidate about their experience...")
    if not user_input:
        return

    # Append user message to UI history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Determine or create session_id using the backend's session handling
    session_id = chat_service.get_or_create_session_id(  # type: ignore[attr-defined]
        session_id=st.session_state.get("session_id"),
        timeout_seconds=180,
    )
    st.session_state["session_id"] = session_id

    # Prepare conversation history for the backend (all prior messages)
    conversation_history: List[Dict[str, str]] = st.session_state["messages"][:-1]

    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()
        assistant_placeholder.markdown("_Thinking..._")

        try:
            response = chat_service.chat(  # type: ignore[attr-defined]
                lang=config["lang"],
                query=user_input,
                conversation_history=conversation_history,
                session_id=session_id,
                k=5,
                temperature=0.7,
                max_tokens=None,
                system_prompt=config["system_prompt"],
                character=config["character"],
                model=None,
            )
            answer = response.get("content") or "No answer was generated."
        except Exception as e:
            answer = f"Error calling backend chat service: {e}"

        assistant_placeholder.markdown(answer)

    # Save assistant message to history
    st.session_state["messages"].append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(
        page_title="ChatMyCV - Streamlit",
        layout="wide",
    )

    init_session_state()
    config = render_sidebar()
    render_chat_ui(config)


if __name__ == "__main__":
    main()


