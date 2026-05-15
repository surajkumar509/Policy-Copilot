
from agent.tools import search_policies, generate_answer, create_checklist, draft_email

def agent_run(inp):
    q = inp.lower()
    ctx = search_policies(inp)

    if any(word in q for word in ["checklist", "steps", "procedure", "process"]):
        return create_checklist(ctx, inp)

    elif any(word in q for word in ["email", "mail", "draft"]):
        return draft_email(ctx, inp)

    else:
        return generate_answer(ctx, inp)