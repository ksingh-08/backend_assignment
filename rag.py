import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Ticket:
    """Simple support ticket class"""
    
    def __init__(
        self, 
        id: str, 
        title: str, 
        description: str = "", 
        browser: str = None, 
        operating_system: str = None, 
        user_type: str = None, 
        problem: str = None, 
        solution: str = None,
        created: str = None
    ):
        self.id = id
        self.title = title
        self.description = description
        self.browser = browser
        self.operating_system = operating_system
        self.user_type = user_type
        self.problem = problem
        self.solution = solution
        self.created = created or datetime.datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert ticket to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "browser": self.browser,
            "operating_system": self.operating_system,
            "user_type": self.user_type,
            "problem": self.problem,
            "solution": self.solution,
            "created": self.created
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ticket':
        """Create ticket from dictionary"""
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            browser=data.get("browser"),
            operating_system=data.get("operating_system"),
            user_type=data.get("user_type"),
            problem=data.get("problem"),
            solution=data.get("solution"),
            created=data.get("created")
        )
    
    def get_text_for_embedding(self) -> str:
        """Get combined text for embedding"""
        parts = [
            f"Title: {self.title}",
            f"Description: {self.description}" if self.description else "",
            f"Browser: {self.browser}" if self.browser else "",
            f"OS: {self.operating_system}" if self.operating_system else "",
            f"User Type: {self.user_type}" if self.user_type else "",
            f"Problem: {self.problem}" if self.problem else "",
            f"Solution: {self.solution}" if self.solution else ""
        ]
        return " ".join([p for p in parts if p])
    
    def __str__(self) -> str:
        """String representation"""
        parts = [
            f"ID: {self.id}",
            f"Title: {self.title}",
            f"Browser: {self.browser}" if self.browser else "",
            f"OS: {self.operating_system}" if self.operating_system else "",
            f"User Type: {self.user_type}" if self.user_type else "",
            f"Problem: {self.problem}" if self.problem else "",
            f"Solution: {self.solution}" if self.solution else "",
            f"Created: {self.created}"
        ]
        return "\n".join([p for p in parts if p])


class SupportRAG:
    """RAG system for support tickets"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with embedding model"""
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        
        self.index = faiss.IndexFlatIP(self.dim)
        
        self.tickets = []
        self.ticket_ids = []
        self.feedback = []
        
        logger.info(f"Support RAG initialized with embedding dimension: {self.dim}")
    
    def add_ticket(self, ticket: Ticket) -> None:
        """Add single ticket to index"""
        text = ticket.get_text_for_embedding()
        embedding = self.model.encode([text])[0]
        
        embedding = embedding / np.linalg.norm(embedding)
        
        self.index.add(np.array([embedding], dtype=np.float32))
        self.tickets.append(ticket)
        self.ticket_ids.append(ticket.id)
        
        logger.info(f"Added ticket {ticket.id}. Total tickets: {len(self.tickets)}")
    
    def add_tickets(self, tickets: List[Ticket]) -> None:
        """Add multiple tickets to index"""
        texts = [t.get_text_for_embedding() for t in tickets]
        embeddings = self.model.encode(texts)
        
    
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(norm_embeddings.astype(np.float32))
        self.tickets.extend(tickets)
        self.ticket_ids.extend([t.id for t in tickets])
        
        logger.info(f"Added {len(tickets)} tickets. Total: {len(self.tickets)}")
    
    def search(self, query: str, limit: int = 3, min_score: float = 0.5) -> List[Tuple[Ticket, float]]:
        """Find relevant tickets for query"""
        if not self.tickets:
            logger.warning("No tickets in index. Returning empty results.")
            return []
        
        
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
       
        scores, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), 
            min(limit, len(self.tickets))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            score = float(scores[0][i])
            if score >= min_score:
                results.append((self.tickets[idx], score))
        
        logger.info(f"Query: '{query}' returned {len(results)} results")
        return results
    
    def record_feedback(self, query: str, ticket_id: str, helpful: bool) -> None:
        """Record user feedback"""
        self.feedback.append({
            "query": query,
            "ticket_id": ticket_id,
            "helpful": helpful,
            "timestamp": datetime.datetime.now().isoformat()
        })
        logger.info(f"Recorded feedback: query='{query}', ticket={ticket_id}, helpful={helpful}")
    
    def generate_response(self, query: str, results: List[Tuple[Ticket, float]]) -> str:
        """Create response based on query and results"""
        if not results:
            return "I couldn't find any relevant information for your issue. Please provide more details."
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        response = [
            f"Based on your query: '{query}', I found these relevant support tickets:",
        ]
        
        for i, (ticket, score) in enumerate(results[:3], 1):
            response.append(f"\n{i}. {ticket.title} (Relevance: {score:.2f})")
            if ticket.problem:
                response.append(f"   Problem: {ticket.problem}")
            if ticket.solution:
                response.append(f"   Solution: {ticket.solution}")
        
        best_ticket = results[0][0]
        
        response.append(f"\nRecommended solution:")
        if best_ticket.solution:
            response.append(f"{best_ticket.solution}")
        else:
            response.append("No specific solution available for this issue.")
        
        response.append("\nWas this helpful for your issue?")
        
        return "\n".join(response)
    
    def save_tickets(self, filepath: str) -> None:
        """Save tickets to JSON"""
        with open(filepath, 'w') as f:
            json.dump([t.to_dict() for t in self.tickets], f, indent=2)
        logger.info(f"Saved {len(self.tickets)} tickets to {filepath}")
    
    def load_tickets(self, filepath: str) -> None:
        """Load tickets from JSON"""
        with open(filepath, 'r') as f:
            ticket_data = json.load(f)
        
        tickets = [Ticket.from_dict(data) for data in ticket_data]
        self.add_tickets(tickets)
        logger.info(f"Loaded {len(tickets)} tickets from {filepath}")


def create_sample_data() -> List[Ticket]:
    """Create sample tickets for testing"""
    return [
        Ticket(
            id="T001",
            title="Login failure on Safari for SSO users",
            browser="Safari 16.3",
            operating_system="macOS",
            user_type="Enterprise",
            problem="Redirect loop during SSO login",
            solution="Clear cookies and update Safari settings to allow cross-site tracking."
        ),
        Ticket(
            id="T002",
            title="Password reset problems",
            user_type="All",
            problem="Password reset email not received",
            solution="Check spam folder and whitelist our domain in email settings."
        ),
        Ticket(
            id="T003",
            title="Chrome extension login conflict",
            browser="Chrome",
            user_type="Small Business",
            problem="Password manager extension causing login issues",
            solution="Temporarily disable the password manager extension when logging in."
        ),
        Ticket(
            id="T004",
            title="Dashboard loading slowly in Firefox",
            browser="Firefox",
            operating_system="Windows 11",
            user_type="Enterprise",
            problem="Dashboard widgets timing out",
            solution="Clear browser cache and increase network timeout settings."
        ),
        Ticket(
            id="T005",
            title="Mobile app login crashes",
            browser="Mobile Safari",
            operating_system="iOS 16",
            user_type="Consumer",
            problem="App crashes during biometric authentication",
            solution="Update to latest app version and reset device biometric settings."
        )
    ]


def demo():
    """Run interactive demo"""
    print("\n===== Support RAG System Demo =====\n")
    support_rag = SupportRAG()
    tickets = create_sample_data()
    support_rag.add_tickets(tickets)
    
    print(f"Loaded {len(tickets)} sample tickets.")
    print("\nAvailable tickets:")
    for i, ticket in enumerate(tickets, 1):
        print(f"{i}. {ticket.title}")
    
    while True:
        print("\n" + "="*40)
        query = input("\nEnter your support query (or 'quit' to exit): ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        results = support_rag.search(query, limit=3, min_score=0.5)
        
        if not results:
            print("\nNo relevant tickets found for your query.")
            continue
        
        print(f"\nFound {len(results)} relevant tickets:")
        
        for i, (ticket, score) in enumerate(results, 1):
            print(f"\n{i}. Ticket: {ticket.id} (Relevance: {score:.4f})")
            print(f"   Title: {ticket.title}")
            if ticket.browser:
                print(f"   Browser: {ticket.browser}")
            if ticket.operating_system:
                print(f"   OS: {ticket.operating_system}")
            if ticket.problem:
                print(f"   Problem: {ticket.problem}")
            if ticket.solution:
                print(f"   Solution: {ticket.solution}")
        
        print("\n" + "-"*40)
        print("Generated Response:")
        print(support_rag.generate_response(query, results))
        
        print("\n" + "-"*40)
        feedback = input("Was this helpful? (y/n): ")
        if feedback.lower() in ['y', 'yes']:
            support_rag.record_feedback(query, results[0][0].id, True)
            print("Thanks for your positive feedback!")
        elif feedback.lower() in ['n', 'no']:
            support_rag.record_feedback(query, results[0][0].id, False)
            print("Thanks for your feedback. We'll use it to improve our responses.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Support Ticket RAG System')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--query', type=str, help='Run a single query')
    args = parser.parse_args()
    
    if args.query:
        support_rag = SupportRAG()
        support_rag.add_tickets(create_sample_data())
        results = support_rag.search(args.query)
        print(support_rag.generate_response(args.query, results))
    else:
        demo()