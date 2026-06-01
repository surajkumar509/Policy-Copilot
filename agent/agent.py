from agent.tools import (
    search_policies,
    generate_answer,
    create_checklist,
    draft_email,
    normalize_query,
)


def agent_run(inp, country_filter="All Countries"):
    normalized = normalize_query(inp)
    context = search_policies(inp, country_filter=country_filter)

    # Use normalized query for intent detection
    if "checklist" in normalized:
        return create_checklist(context, inp, cache_scope=country_filter)

    elif "email" in normalized:
        return draft_email(context, inp, cache_scope=country_filter)

    else:
        return generate_answer(context, inp, cache_scope=country_filter)
