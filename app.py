import streamlit as st
from rag_backend import get_agent  # Import the get_agent function
from langchain_core.messages import AIMessageChunk  # Add the streaming output

# 1. Page Configuration
st.set_page_config(page_title="UCSD Course Assistant", page_icon="🎓")
st.title("🎓 UCSD CSE Course Assistant")


# Cache the agent so no need to re-load everytime from backend
@st.cache_resource
def load_cached_agent():
    return get_agent()


# This will call 'get_agent' only the first time the app runs
# On every subsequent iteration, return the ready made agent instantly
agent = load_cached_agent()

# Add side bar with clear button
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Conversation"):
        # Reset the message history to default greeting message
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I can help you find information about UCSD CSE courses. What would you like to know?",
            }
        ]
        # Rerun the app to refresh the screen instantly
        st.rerun()

# 2. Initialize Chat History
# Streamlit refreshes the script on every interaction, so we use session_state to keep memory.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I can help you find information about UCSD CSE courses. What would you like to know?",
        }
    ]

# 3. Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# 4. Handle User Input
if prompt := st.chat_input("Ask about CSE courses..."):
    # Display user message immediately
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from the RAG agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # We use a container to show "thinking" or intermediate steps if desired
        with st.status("Thinking ...", expanded=False) as status:
            try:
                # FLAGS TO PREVENT REPEATING OUTPUT in thinking
                first_token_received = False
                tool_call_detected = False
                recent_messages = st.session_state.messages[-6:]

                # Switch stream mode to "messages" to get token chunks, metadata
                for chunk, metadata in agent.stream(
                    {"messages": recent_messages}, stream_mode="messages"
                ):

                    # If it's a tool message (retrieval), means a tool just finish running
                    if chunk.type == "tool":
                        status.write(f"Tool output: {chunk.content[:200]}...")
                        tool_call_detected = False

                    # Case 2: AI token, the agent is talking to you
                    # AIMessageChunk is a piece of the AI response
                    elif isinstance(chunk, AIMessageChunk):
                        # If it has a tool_call_chunks, its deciding to use a tool
                        if chunk.tool_call_chunks:
                            if not tool_call_detected:
                                status.write("⚙️ Deciding to search...")
                                tool_call_detected = True

                        # If it has content, then its the final answer
                        elif chunk.content:
                            # The moment we get text, close status box
                            if not first_token_received:
                                status.update(
                                    label="Complete!", state="complete", expanded=False
                                )
                                first_token_received = True

                            # Append the new token to our full response
                            full_response += chunk.content

                            # Update the UI with the cursor effect
                            message_placeholder.markdown(full_response + "▌")

                # Final cleanup: remove the blinking cursor
                message_placeholder.markdown(full_response)

                # Append assistant response to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
