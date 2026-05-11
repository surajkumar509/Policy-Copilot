
from agent.tools import search_policies, generate_answer, create_checklist, draft_email

def agent_run(inp):
    i = inp.lower()
    ctx = search_policies(inp)

    if 'checklist' in i:
        return create_checklist(ctx, inp)

    if 'email' in i:
        return draft_email(ctx, inp)

    return generate_answer(ctx, inp)