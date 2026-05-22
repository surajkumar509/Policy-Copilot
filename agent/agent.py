from agent.tools import (
    search_policies,
    generate_answer,
    create_checklist,
    draft_email,
    normalize_query   
)

def agent_run(inp):
    normalized = normalize_query(inp)   
    context = search_policies(inp)

    # Use normalized query for intent detection
    if "checklist" in normalized:
        return create_checklist(context, inp)

    elif "email" in normalized:
        return draft_email(context, inp)

    else:
        return generate_answer(context, inp)