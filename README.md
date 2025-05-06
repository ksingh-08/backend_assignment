# Running the Streamlit Web Interface

The RAG-based Support Ticket System includes a Streamlit web interface for easier interaction with the system. This document explains how to set up and run the Streamlit app.

## Prerequisites

Before running the Streamlit interface, ensure you have installed all the required dependencies:

```bash
pip install -r requirements.txt
```

## Starting the Streamlit App

To start the Streamlit app, navigate to the project directory and run:

```bash
streamlit run streamlit_file.py
```

This will:
1. Start a local web server
2. Open a browser window with the interface
3. Load sample tickets into the system

## Using the Web Interface

### Search Tab

The main search functionality allows you to:

1. Enter support queries in natural language
2. View relevant tickets ranked by semantic similarity
3. See a generated response based on the retrieved tickets
4. Provide feedback on the relevance of results

**Sample queries to try:**
- "Login error on Safari browser for enterprise users"
- "Password reset email problems"
- "Chrome extension causing login issues"

### Filter Options

On the sidebar, you can:
- Adjust the number of results to retrieve (K)
- Set the minimum relevance threshold
- Filter by browser, OS, or customer type

### View All Tickets

This tab displays all tickets in the system and allows you to:
- See all tickets in a sortable table
- View detailed information for any selected ticket

### Add New Ticket

You can contribute to the knowledge base by:
- Adding new support tickets with detailed information
- Providing resolution steps for known issues

## Features

The Streamlit interface offers several advantages:
- **Visual representation** of search results and relevance scores
- **Interactive filters** for refining searches
- **Feedback collection** for system improvement
- **Easy navigation** between different functions

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed
2. Check that you're running the app from the correct directory
3. Verify that the Python environment has access to all required packages
4. For any port conflicts, you can specify a different port:
   ```bash
   streamlit run streamlit_file.py --server.port 8501
   ```

## Next Steps

This web interface is a prototype and could be extended with:
- User authentication
- Enhanced visualization of semantic relationships
- Integration with external ticket systems