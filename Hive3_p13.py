"""
HiveMindV2_FI13.py - Cleaned & Improved
Federated AI / FI-13 inspired local multi-agent debate system
"""

import json
import asyncio
import ollama
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from loguru import logger

# ====================== PROJECT ROOT & PATH SETUP ======================
ROOT = Path(__file__).resolve().parent

# Add current directory to Python path so local modules can be imported reliably
sys.path.insert(0, str(ROOT))

# ====================== ROLES.JSON LOCATION ======================
possible_paths = [
    ROOT / "roles.json",
    ROOT / "src" / "roles.json",
    Path("D:/Projects/Hive_p13/roles.json"),
    Path("D:/Projects/Hive_p13/src/roles.json"),
]

ROLES_FILE = None
for p in possible_paths:
    if p.exists():
        ROLES_FILE = p
        break

if not ROLES_FILE:
    raise FileNotFoundError(
        f"roles.json not found. Searched:\n" + "\n".join(str(p) for p in possible_paths)
    )

print(f"✓ Loaded roles.json from: {ROLES_FILE}")

# ====================== OUTPUT DIRECTORIES ======================
WORKING_MEMORY = ROOT / "model_memory.json"
OUTPUT_DIR = ROOT / "output"
GRAPH_DIR = ROOT / "knowledge_graph"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# ====================== LOCAL LAYER IMPORTS ======================
from embedding_layer import EmbeddingLayer
from symbolic_layer import SymbolicLayer
from bridge import Bridge


class FI13Governance:
    """Software implementation of P13 governance + P7 semantic tracking."""
    
    def __init__(self, embedding_layer, privacy_epsilon: float = 0.1):
        self.emb = embedding_layer
        self.epsilon = privacy_epsilon

    def verify_privacy_boundary(self, text: str) -> bool:
        """Basic P13 circuit breaker for sensitive data leakage."""
        sensitive = ["PRIVATE_KEY", "SECRET_", "PASSWORD", "API_KEY", "VLLM_KEY"]
        return not any(token in text.upper() for token in sensitive)

    def calculate_alignment_drift(self, vec_a, vec_b) -> float:
        """P7 semantic drift measurement."""
        return float(self.emb.semantic_distance(vec_a, vec_b))


class HiveMindV2_FI13:
    """Main FI-13 inspired orchestration core."""
    
    def __init__(self):
        logger.info("Initializing HiveMindV2_FI13 under FI-13 standards...")

        self.emb = EmbeddingLayer(device="cpu")
        self.sym = SymbolicLayer()
        self.bridge = Bridge(self.emb, self.sym)
        self.gov = FI13Governance(self.emb)

        # Build FAISS index for semantic translation
        self.bridge.index_symbolic_layer()

        # Load roles
        with open(ROLES_FILE, "r", encoding="utf-8") as f:
            self.roles_config = json.load(f)["roles"]

        logger.success("HiveMindV2_FI13 initialized successfully.")

    async def get_response(self, role_name: str, prompt: str) -> str:
        """Get response from a specific role/agent."""
        if role_name not in self.roles_config:
            raise KeyError(f"Role '{role_name}' not found in roles.json")

        role_data = self.roles_config[role_name]
        model_target = role_data["model"]
        system_prompt = role_data["prompt_template"]

        logger.info(f"→ Calling {role_name.upper()} ({model_target})")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Run synchronous ollama call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(model=model_target, messages=messages)
        )

        content = response["message"]["content"].strip()

        # P13 Governance Check
        if not self.gov.verify_privacy_boundary(content):
            logger.critical(f"P13 CIRCUIT BREAKER TRIPPED for role '{role_name}'")
            return "[REDACTED - FI-13 GOVERNANCE VIOLATION]"

        return content

    async def run_debate(self, topic: str):
        """Run a full Specialist Debate cycle (P10/P11 pattern)."""
        logger.info(f"Starting FI-13 Debate on: {topic}")

        # 1. Synthesizer proposes
        prop = await self.get_response(
            "synthesizer", 
            f"Propose a detailed solution for: {topic}"
        )

        # 2. Counter critiques
        crit = await self.get_response(
            "counter", 
            f"Provide a rigorous critique of this proposal: {prop[:800]}..."
        )

        # 3. Validator / Arbiter
        judge_prompt = f"""As the FI-13 Arbiter, evaluate this debate:

PROPOSAL:
{prop}

CRITIQUE:
{crit}

Deliver a clear final verdict: Is the proposal fundamentally sound?"""
        
        verdict = await self.get_response("validator", judge_prompt)

        # Record in symbolic layer
        self.sym.add_triple("synthesizer", "proposes", topic, {"content": prop[:100]})
        self.sym.add_triple("counter", "critiques", "synthesizer", {"flaw": crit[:100]})
        self.sym.add_triple("validator", "verdict_on", topic, {"verdict": verdict[:100]})

        return prop, crit, verdict


async def main():
    try:
        hive = HiveMindV2_FI13()
        
        # Target path for your complex prompts
        PROMPT_DIR = ROOT / "prompts"
        PROMPT_FILE = PROMPT_DIR / "fi13_spec.txt"
        
        # Check if the automated text file exists
        if PROMPT_FILE.exists():
            logger.info(f"Reading target prompt from text file: {PROMPT_FILE.name}")
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                user_topic = f.read().strip()
            print("\n✓ Prompt successfully loaded from file structure.")
        else:
            # Fallback if you just want to run a quick manual question later
            print(f"\n[Notice: {PROMPT_FILE} not found. Falling back to console input.]")
            user_topic = input("Enter the debate topic/prompt: ").strip()

        if not user_topic:
            logger.error("Empty prompt detected. Aborting execution loop.")
            return

        # Execute the debate chain across your models
        p, c, v = await hive.run_debate(user_topic)

        # Build out the final validated ledger on your D: drive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = OUTPUT_DIR / f"Debate_FI13_{timestamp}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# FI-13 Protocol Evaluation Ledger\n\n")
            f.write(f"**Evaluated Target Architecture Blueprint:**\n\n```text\n{user_topic}\n```\n\n")
            f.write(f"## 1. Synthesizer Proposal ([SYNTHESIZER] / Mistral)\n{p}\n\n")
            f.write(f"## 2. Adversarial Critique ([COUNTER] / Llama 3.1)\n{c}\n\n")
            f.write(f"## 3. Validator Verdict ([VALIDATOR] / DeepSeek-R1)\n{v}\n\n")
            f.write(f"---\nGenerated via Local Silicon: {datetime.now()}\n")

        logger.success(f"Execution complete. Ledger successfully saved to: {output_file.name}")

    except Exception as e:
        logger.exception(f"Fatal execution crash in core runtime loop: {e}")

if __name__ == "__main__":
    asyncio.run(main())