# hive_chat.py v3.1 — JSON-Driven Role System
# Features: External role definitions, MBTI cognitive styles, per-model memory

import ollama
import datetime
import re
import json
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("D:/Projects/agentic_ai/demos")
# TIMESTAMPED LOGS — AUTO-CREATES NEW FILE EVERY HOUR
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
LOG_FILE = BASE_DIR / f"hive_session_{timestamp}.md"
MEMORY_FILE = BASE_DIR / "model_memory.json"
ROLES_FILE = BASE_DIR / "roles.json"

# Ensure directories exist
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ROLE SYSTEM
# ============================================================================

class RoleSystem:
    """Load and manage role definitions from JSON."""
    
    def __init__(self, roles_file):
        self.roles_file = roles_file
        self.config = {}
        self.load()
    
    def load(self):
        """Load roles from JSON file."""
        if not self.roles_file.exists():
            raise FileNotFoundError(
                f"roles.json not found at {self.roles_file}\n"
                "Please create roles.json with role definitions."
            )
        
        with open(self.roles_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        print(f"✓ Loaded roles.json v{self.config.get('version', '?')}")
        print(f"  Available roles: {', '.join(self.config['roles'].keys())}")
        print(f"  Workflows: {', '.join(self.config['workflows'].keys())}\n")
    
    def get_role(self, role_name):
        """Get role configuration."""
        if role_name not in self.config['roles']:
            raise ValueError(f"Unknown role: {role_name}")
        return self.config['roles'][role_name]
    
    def get_workflow(self, workflow_name):
        """Get workflow configuration."""
        if workflow_name not in self.config['workflows']:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        return self.config['workflows'][workflow_name]
    
    def get_defaults(self):
        """Get Ollama default options."""
        return self.config.get('ollama_defaults', {})
    
    def list_roles(self):
        """Print all available roles."""
        print("\n" + "="*80)
        print("AVAILABLE ROLES:")
        print("="*80)
        for name, role in self.config['roles'].items():
            print(f"\n{name.upper():15} ({role['mbti']})")
            print(f"  Model: {role['model']}")
            print(f"  Description: {role['description']}")
        print("="*80 + "\n")
    
    def list_workflows(self):
        """Print all available workflows."""
        print("\n" + "="*80)
        print("AVAILABLE WORKFLOWS:")
        print("="*80)
        for name, wf in self.config['workflows'].items():
            roles_str = ', '.join(wf['roles'])
            print(f"\n{name.upper():15} ({len(wf['roles'])} models)")
            print(f"  Roles: {roles_str}")
            print(f"  Use case: {wf['use_case']}")
        print("="*80 + "\n")

# ============================================================================
# MEMORY SYSTEM
# ============================================================================

class ModelMemory:
    """Per-model memory system."""
    
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.memories = defaultdict(list)
        self.load()
    
    def load(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memories = defaultdict(list, data)
            except:
                pass
    
    def save(self):
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.memories), f, indent=2)
    
    def add(self, role, prompt, response):
        self.memories[role].append({
            "prompt": prompt,
            "response": response[:300],
            "timestamp": datetime.datetime.now().isoformat()
        })
        if len(self.memories[role]) > 5:
            self.memories[role] = self.memories[role][-5:]
        self.save()
    
    def get_context(self, role, max_items=3):
        recent = self.memories[role][-max_items:]
        if not recent:
            return ""
        
        context = f"\n### YOUR PREVIOUS RESPONSES ({role.upper()}):\n"
        for i, item in enumerate(recent, 1):
            context += f"{i}. Q: {item['prompt'][:80]}...\n"
            context += f"   A: {item['response'][:150]}...\n\n"
        return context
    
    def clear(self):
        self.memories.clear()
        self.save()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log(text):
    """Write to log file and console."""
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def start_new_session():
    """Create timestamped session header."""
    now = datetime.datetime.now()
    header = f"\n{'='*80}\n"
    header += f"## SESSION: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"{'='*80}\n\n"
    log(header)
    return now

def extract_confidence(output):
    match = re.search(r"Confidence:\s*(\d+)", output, re.IGNORECASE)
    return int(match.group(1)) if match else 5

def check_citations(output):
    suspicious_patterns = [
        r"\(Smith et al\.?,?\s*\d{4}\)",
        r"\(Johnson\.?,?\s*\d{4}\)",
        r"\(Jones et al\.?,?\s*\d{4}\)",
        r"Recent studies show(?!\s+\[)",
        r"It is well[- ]known that",
        r"Studies have shown(?!\s+\[)"
    ]
    
    flags = []
    for pattern in suspicious_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            flags.append(pattern.split('\\')[0][:20])
    
    return flags

# ============================================================================
# MODEL EXECUTION
# ============================================================================

def run_model(role_name, role_system, user_prompt, memory, word_cap=None, other_responses=None):
    """
    Execute a model with JSON-defined role.
    """
    
    # Get role configuration
    role_config = role_system.get_role(role_name)
    model = role_config['model']
    mbti = role_config.get('mbti', 'N/A')
    
    print(f"\n[{role_name.upper():12}] {model:35} ({mbti:4}) ...", end=" ", flush=True)
    
    # Build prompt
    prompt_parts = [role_config['prompt_template']]
    
    # Add model's own memory
    model_context = memory.get_context(role_name, max_items=2)
    if model_context:
        prompt_parts.append(model_context)
    
    # Add other models' responses (for synthesizer/validator)
    if other_responses and role_name in ["synthesizer", "validator"]:
        prompt_parts.append("\n### OTHER MODELS' RESPONSES THIS TURN:")
        for other_role, other_resp in other_responses.items():
            other_mbti = role_system.get_role(other_role).get('mbti', '')
            prompt_parts.append(f"\n**{other_role.title()} ({other_mbti}):** {other_resp[:200]}...\n")
    
    # Add word cap
    if word_cap:
        prompt_parts.append(f"\n**WORD LIMIT: {word_cap} words maximum**\n")
    
    # Add user prompt
    prompt_parts.append(f"\n### USER QUESTION:\n{user_prompt}")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Configure Ollama options
    defaults = role_system.get_defaults()
    options = defaults.copy()
    
    # Apply role-specific overrides
    if 'temperature_override' in role_config:
        options['temperature'] = role_config['temperature_override']
    
    if word_cap:
        multiplier = role_config.get('max_tokens_multiplier', 1.5)
        options['num_predict'] = int(word_cap * multiplier)
    
    # Execute
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            options=options
        )
        output = response['message']['content'].strip()
        
        # Extract metadata
        confidence = extract_confidence(output)
        citation_flags = check_citations(output)
        
        # Store in memory
        memory.add(role_name, user_prompt, output)
        
        # Log results
        print(f"✓ [Conf: {confidence}/10]")
        
        log(f"\n### {role_name.upper()}: `{model.split(':')[0]}` ({mbti}) | Confidence: {confidence}/10")
        log(f"\n{output}\n")
        
        if citation_flags:
            warning = f"⚠️  **Citation Warning**: {', '.join(set(citation_flags))}"
            log(warning)
            print(f"  {warning}")
        
        log("---")
        
        return output, confidence
        
    except Exception as e:
        error_msg = f"❌ ERROR: {str(e)}"
        print(error_msg)
        log(f"\n{error_msg}\n---")
        return "", 0

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """Main hive-mind loop."""
    
    # Initialize systems
    try:
        role_system = RoleSystem(ROLES_FILE)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    
    memory = ModelMemory(MEMORY_FILE)
    
    # Create log header if new
    if not LOG_FILE.exists():
        log("# 🧠 Schermerhorn CQG Hive-Mind Log\n")
        log("*Multi-Model Collaborative Research System v3.1*\n")
        log("*JSON-Driven Role System with MBTI Cognitive Diversity*\n")
    
    # Welcome
    print("\n" + "="*80)
    print("🧠 HIVE-MIND v3.1 — JSON-Driven Role System")
    print("="*80)
    print("\nUSAGE: [your question] |[word_cap] |[workflow]")
    print("\nCOMMANDS:")
    print("  quit / q       — Exit")
    print("  clear          — Clear all model memories")
    print("  memory [role]  — Show memory for specific role")
    print("  roles          — List all available roles")
    print("  workflows      — List all workflows")
    print("  reload         — Reload roles.json (after editing)")
    print("\nEXAMPLES:")
    print("  What is the fossil field? |100 |standard")
    print("  Derive Bianchi identity |200 |technical")
    print("  Explain interference |150 |deep")
    print("="*80 + "\n")
    
    # Main loop
    while True:
        try:
            raw_input = input("You: ").strip()
            
            if not raw_input:
                continue
            
            # Commands
            if raw_input.lower() in ["quit", "q", "exit"]:
                print("\n👋 Goodbye! Memory saved.\n")
                break
            
            if raw_input.lower() == "clear":
                memory.clear()
                print("✓ All model memories cleared.\n")
                continue
            
            if raw_input.lower().startswith("memory"):
                parts = raw_input.split()
                if len(parts) > 1:
                    role = parts[1]
                    if role in role_system.config['roles']:
                        print(f"\n--- {role.upper()} MEMORY ---")
                        print(memory.get_context(role, max_items=5))
                    else:
                        print(f"Unknown role: {role}")
                else:
                    print("\nAll model memories:")
                    for role in role_system.config['roles'].keys():
                        count = len(memory.memories.get(role, []))
                        mbti = role_system.get_role(role).get('mbti', 'N/A')
                        print(f"  {role:12} ({mbti:4}) : {count} items")
                print()
                continue
            
            if raw_input.lower() == "roles":
                role_system.list_roles()
                continue
            
            if raw_input.lower() == "workflows":
                role_system.list_workflows()
                continue
            
            if raw_input.lower() == "reload":
                role_system.load()
                print("✓ roles.json reloaded\n")
                continue
            
            # Parse input
            parts = raw_input.split("|")
            prompt = parts[0].strip()
            word_cap = None
            workflow = "standard"
            
            if len(parts) > 1:
                cap_str = parts[1].strip()
                if cap_str.isdigit():
                    word_cap = int(cap_str)
            
            if len(parts) > 2:
                wf_str = parts[2].strip().lower()
                if wf_str in role_system.config['workflows']:
                    workflow = wf_str
            
            # Start session
            start_new_session()
            
            # Log query
            workflow_config = role_system.get_workflow(workflow)
            query_info = f"**USER:** {prompt}"
            if word_cap:
                query_info += f" | **Word Cap:** {word_cap}"
            query_info += f" | **Workflow:** {workflow} ({workflow_config['description']})"
            log(query_info + "\n")
            
            print(f"\n🔍 Workflow: {workflow}")
            print(f"   {workflow_config['description']}")
            print(f"   Roles: {', '.join(workflow_config['roles'])}")
            if word_cap:
                print(f"   Word limit: {word_cap}")
            
            # Execute workflow
            responses = {}
            confidences = []
            
            for role_name in workflow_config['roles']:
                other_responses = responses.copy() if role_name in ["synthesizer", "validator"] else None
                
                output, conf = run_model(
                    role_name=role_name,
                    role_system=role_system,
                    user_prompt=prompt,
                    memory=memory,
                    word_cap=word_cap,
                    other_responses=other_responses
                )
                
                responses[role_name] = output
                confidences.append((role_name, conf))
            
            # Summary
            if confidences:
                avg_conf = sum(c for _, c in confidences) / len(confidences)
                
                log(f"\n### 📊 SESSION SUMMARY")
                log(f"- **Average Confidence:** {avg_conf:.1f}/10")
                log(f"- **Models Used:** {len(confidences)}")
                log(f"- **MBTI Coverage:** {', '.join([role_system.get_role(r).get('mbti', 'N/A') for r, _ in confidences])}")
                
                if avg_conf < 5:
                    warning = "⚠️  **LOW CONFIDENCE** — Verify with external sources"
                    log(f"- {warning}")
                    print(f"\n{warning}")
                elif avg_conf >= 8:
                    log(f"- ✅ **HIGH CONFIDENCE** — Results appear robust")
                    print(f"\n✅ High confidence results")
                
                log(f"\n{'='*80}\n")
                
                print(f"\n✓ Session complete | Avg Confidence: {avg_conf:.1f}/10")
                print(f"→ {LOG_FILE}\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'quit' to exit.\n")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()