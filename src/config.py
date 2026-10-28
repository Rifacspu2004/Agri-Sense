"""
Central Configuration and API Key Management for AgriSense Agent.
Manages environment variables, API secrets (100% free tiers supported),
and storage directory paths.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage and Artifact Paths
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data" / "raw"))
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", BASE_DIR / "knowledge_docs"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "chroma_db"))

# Ensure critical directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class APIConfig:
    """Central store for free-tier and external API credentials."""

    @staticmethod
    def get_tavily_key() -> str:
        """Returns Tavily API Key (Free tier: 1000 queries/mo)."""
        return os.getenv("TAVILY_API_KEY", "").strip()

    @staticmethod
    def get_gemini_key() -> str:
        """Returns Google Gemini API Key (Free tier via Google AI Studio)."""
        return os.getenv("GEMINI_API_KEY", "").strip()

    @staticmethod
    def get_groq_key() -> str:
        """Returns Groq API Key (Free fast inference tier)."""
        return os.getenv("GROQ_API_KEY", "").strip()

    @staticmethod
    def update_key(key_name: str, key_value: str) -> None:
        """Dynamically updates an API key in memory and persists to .env file."""
        os.environ[key_name] = key_value.strip()
        env_file = BASE_DIR / ".env"

        lines = []
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

        updated = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{key_name}="):
                new_lines.append(f"{key_name}={key_value.strip()}\n")
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(f"{key_name}={key_value.strip()}\n")

        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Provides status summary of configured keys without revealing full secrets."""
        tavily = cls.get_tavily_key()
        gemini = cls.get_gemini_key()
        groq = cls.get_groq_key()

        return {
            "TAVILY_API_KEY": {
                "configured": bool(tavily),
                "preview": f"{tavily[:4]}...{tavily[-4:]}" if len(tavily) > 8 else ("Set" if tavily else "Not Set"),
                "service": "Tavily Web Search (Free Tier)",
                "free_link": "https://tavily.com"
            },
            "GEMINI_API_KEY": {
                "configured": bool(gemini),
                "preview": f"{gemini[:4]}...{gemini[-4:]}" if len(gemini) > 8 else ("Set" if gemini else "Not Set"),
                "service": "Google Gemini (Free AI Studio)",
                "free_link": "https://aistudio.google.com"
            },
            "GROQ_API_KEY": {
                "configured": bool(groq),
                "preview": f"{groq[:4]}...{groq[-4:]}" if len(groq) > 8 else ("Set" if groq else "Not Set"),
                "service": "Groq Cloud (Free Tier)",
                "free_link": "https://console.groq.com"
            },
            "DUCKDUCKGO_SEARCH": {
                "configured": True,
                "preview": "Always Ready (No Key Required)",
                "service": "DuckDuckGo Free Web Search",
                "free_link": "https://duckduckgo.com"
            }
        }
