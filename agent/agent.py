from agent.tools import (
    search_policies,
    generate_answer,
    create_checklist,
    draft_email
)

def agent_run(inp):
    q = inp.lower()

    context = search_policies(inp)

    if any(word in q for word in ["checklist", "steps", "procedure", "process"]):
        return create_checklist(context, inp)

    elif any(word in q for word in ["email", "mail", "draft"]):
        return draft_email(context, inp)

    else:
        return generate_answer(context, inp)