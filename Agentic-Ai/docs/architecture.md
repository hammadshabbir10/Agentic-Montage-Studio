# Phase 1 Architecture

This phase uses a Supervisor-Worker workflow orchestrated by LangGraph.

## Key Components
- Mode selector node decides manual vs auto generation
- Validator node checks script structure
- Scriptwriter node generates script via MCP-discovered LLM tools (Groq)
- HITL node provides checkpoint confirmation
- Character node extracts character identities
- Image node produces placeholder images via MCP-discovered tools
- Memory commit node stores artifacts in the vector store

## MCP Tool Discovery
Tools are loaded dynamically from data/mcp_registry.json at runtime.
