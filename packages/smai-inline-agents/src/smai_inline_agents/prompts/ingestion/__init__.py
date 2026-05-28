"""Prompt package for the ingestion subagent (planner-refactor Step 3).

Holds ``paper_agent.yaml`` (the SciReplicate-shaped Paper Agent prompt).
Loaded via :func:`smai_inline_agents.ingestion.paper_agent.load_paper_agent_prompt`,
not the role-keyed ``load_prompt_config`` loader: this is a single
standalone prompt, not a base+variant role tree.
"""
