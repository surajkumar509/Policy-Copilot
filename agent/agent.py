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

    original_lower = inp.lower()

    # Email intent: check original query first so that "draft email ... steps ..."
    # is not mis-routed to checklist by the steps→checklist synonym replacement.
    if any(w in original_lower for w in ("draft", "email", "mail")):
        return draft_email(context, inp, cache_scope=country_filter)

    elif "checklist" in normalized:
        return create_checklist(context, inp, cache_scope=country_filter)

    else:
        return generate_answer(context, inp, cache_scope=country_filter)
