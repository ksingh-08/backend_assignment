import os
import streamlit as st
import pandas as pd
import time
from typing import List, Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import SupportRAG, Ticket, create_sample_data

st.set_page_config(
    page_title="Support Ticket RAG System",
    page_icon="🔍",
    layout="wide",
)

if "support_rag" not in st.session_state:
    st.session_state.support_rag = SupportRAG()
    
    tickets = create_sample_data()
    st.session_state.support_rag.add_tickets(tickets)
    
    st.session_state.search_results = []
    st.session_state.last_query = ""
    st.session_state.feedback_submitted = False

st.title("Support Ticket RAG System")
st.markdown("""
This system uses semantic search with embeddings to find relevant support tickets 
and generate helpful responses for customer inquiries.
""")

with st.sidebar:
    st.header("Settings")
    
    st.subheader("Search Parameters")
    limit = st.slider("Number of results", min_value=1, max_value=10, value=3)
    min_score = st.slider("Minimum relevance score", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    st.subheader("Filters")
    
    browsers = ["Any"] + list(set([t.browser for t in st.session_state.support_rag.tickets if t.browser]))
    operating_systems = ["Any"] + list(set([t.operating_system for t in st.session_state.support_rag.tickets if t.operating_system]))
    user_types = ["Any"] + list(set([t.user_type for t in st.session_state.support_rag.tickets if t.user_type]))
    
    browser_filter = st.selectbox("Browser", browsers)
    os_filter = st.selectbox("Operating System", operating_systems)
    user_filter = st.selectbox("User Type", user_types)
    
    st.subheader("Statistics")
    st.write(f"Total tickets: {len(st.session_state.support_rag.tickets)}")

tab1, tab2, tab3 = st.tabs(["Search Tickets", "View All Tickets", "Add New Ticket"])

with tab1:
    st.header("Search for Support Tickets")
    
    query = st.text_input("Enter your support query:", 
                          help="Example: 'Login error on Safari browser'")
    
    if st.button("Search", type="primary") and query:
        st.session_state.last_query = query
        st.session_state.feedback_submitted = False
        
        browser_arg = None if browser_filter == "Any" else browser_filter
        os_arg = None if os_filter == "Any" else os_filter
        user_arg = None if user_filter == "Any" else user_filter
        
        start_time = time.time()
        results = st.session_state.support_rag.search(query, limit=limit, min_score=min_score)
        search_time = time.time() - start_time
        
        if browser_arg or os_arg or user_arg:
            filtered_results = []
            for ticket, score in results:
                if browser_arg and browser_arg.lower() not in (ticket.browser or "").lower():
                    continue
                if os_arg and os_arg.lower() not in (ticket.operating_system or "").lower():
                    continue
                if user_arg and user_arg.lower() not in (ticket.user_type or "").lower():
                    continue
                filtered_results.append((ticket, score))
            results = filtered_results
        
        st.session_state.search_results = results
        
        st.write(f"Found {len(results)} relevant tickets in {search_time:.3f} seconds")
        
        if not results:
            st.warning("No relevant tickets found for your query.")
        else:
            results_data = []
            for ticket, score in results:
                results_data.append({
                    "Ticket ID": ticket.id,
                    "Title": ticket.title,
                    "Browser": ticket.browser or "-",
                    "OS": ticket.operating_system or "-",
                    "User Type": ticket.user_type or "-",
                    "Relevance": f"{score:.4f}"
                })
            
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            
            st.header("Generated Response")
            with st.spinner("Generating response..."):
                response = st.session_state.support_rag.generate_response(query, results)
            
            st.write(response)
            
            st.subheader("Was this helpful?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("👍 Yes", use_container_width=True) and not st.session_state.feedback_submitted:
                    st.session_state.support_rag.record_feedback(query, results[0][0].id, True)
                    st.session_state.feedback_submitted = True
                    st.success("Thanks for your positive feedback!")
            
            with col2:
                if st.button("👎 No", use_container_width=True) and not st.session_state.feedback_submitted:
                    st.session_state.support_rag.record_feedback(query, results[0][0].id, False)
                    st.session_state.feedback_submitted = True
                    st.info("Thanks for your feedback. It helps us improve.")
    
    st.subheader("Sample Queries")
    sample_queries = [
        "Login error on Safari browser",
        "Password reset email problems",
        "Chrome extension issues",
        "Mobile app crashes during login",
        "Dashboard loading slowly"
    ]
    
    for sample in sample_queries:
        if st.button(f"Try: {sample}", key=f"sample_{hash(sample)}", use_container_width=True):
            st.session_state.query = sample
            st.rerun()


with tab2:
    st.header("All Support Tickets")
    
    tickets_data = []
    for ticket in st.session_state.support_rag.tickets:
        tickets_data.append({
            "Ticket ID": ticket.id,
            "Title": ticket.title,
            "Browser": ticket.browser or "-",
            "OS": ticket.operating_system or "-",
            "User Type": ticket.user_type or "-",
            "Problem": ticket.problem or "-",
            "Solution": ticket.solution or "-"
        })
    
    st.dataframe(pd.DataFrame(tickets_data), use_container_width=True)
    
    selected_ticket_id = st.selectbox("Select ticket to view details:", 
                                    [""] + [t.id for t in st.session_state.support_rag.tickets])
    
    if selected_ticket_id:
        selected_ticket = next(
            (t for t in st.session_state.support_rag.tickets if t.id == selected_ticket_id), 
            None
        )
        
        if selected_ticket:
            st.subheader(f"Ticket Details: {selected_ticket.id}")
            
            st.write(f"**Title:** {selected_ticket.title}")
            if selected_ticket.browser:
                st.write(f"**Browser:** {selected_ticket.browser}")
            if selected_ticket.operating_system:
                st.write(f"**OS:** {selected_ticket.operating_system}")
            if selected_ticket.user_type:
                st.write(f"**User Type:** {selected_ticket.user_type}")
            if selected_ticket.problem:
                st.write(f"**Problem:** {selected_ticket.problem}")
            if selected_ticket.solution:
                st.write(f"**Solution:** {selected_ticket.solution}")
            if selected_ticket.created:
                st.write(f"**Created:** {selected_ticket.created}")


with tab3:
    st.header("Add New Support Ticket")
    
    with st.form("add_ticket_form"):
        title = st.text_input("Title", help="Brief description of the issue")
        description = st.text_area("Description", help="Additional details about the issue")
        browser = st.text_input("Browser", help="e.g., Chrome, Safari, Firefox")
        os_input = st.text_input("Operating System", help="e.g., Windows, macOS, Android")
        user_type = st.text_input("User Type", help="e.g., Enterprise, Consumer, Small Business")
        problem = st.text_area("Problem", help="Detailed description of the problem")
        solution = st.text_area("Solution", help="Steps to resolve the issue (if known)")
        
        submit_button = st.form_submit_button("Add Ticket")
        
        if submit_button and title and problem:
            ticket_id = f"T{len(st.session_state.support_rag.tickets) + 1:03d}"
            
            new_ticket = Ticket(
                id=ticket_id,
                title=title,
                description=description or "",
                browser=browser or None,
                operating_system=os_input or None,
                user_type=user_type or None,
                problem=problem,
                solution=solution or None
            )
            
            st.session_state.support_rag.add_ticket(new_ticket)
            st.success(f"Ticket {ticket_id} added successfully!")
            
           
            st.rerun() 
        elif submit_button:
            st.error("Title and Problem fields are required.")

st.markdown("---")
st.markdown("""
**Support Ticket RAG System** - This application demonstrates semantic search for 
support ticket retrieval and response generation using embedding-based similarity.
""")