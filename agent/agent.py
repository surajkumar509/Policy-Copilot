from agent.tools import (
    search_policies,
    generate_answer,
    create_checklist,
    draft_email,
    analyze_compliance,
    analyze_website_url,
    normalize_query,
)


def agent_run(
    inp,
    country_filter="All Countries",
    task_mode="Smart Policy Mode",
    website_url="",
    website_max_pages=8,
    website_pasted_content="",
):
    if task_mode == "Website URL Analysis":
        return analyze_website_url(
            website_url,
            inp,
            max_pages=website_max_pages,
            pasted_content=website_pasted_content,
        )

    original_lower = inp.lower()
    normalized = normalize_query(inp)
    context = search_policies(inp, country_filter=country_filter)

    # Keep explicit task intents highest priority inside Smart Policy Mode.
    # Otherwise phrases like "leave approval email" get mis-routed to compliance.
    if any(w in original_lower for w in ("draft", "email", "mail")):
        return draft_email(context, inp, cache_scope=country_filter)

    if "checklist" in normalized:
        return create_checklist(context, inp, cache_scope=country_filter)

    compliance_terms = (
        "compliance",
        "eligible",
        "eligibility",
        "approved",
        "approval",
        "violation",
        "can i",
        "allowed",
        "not allowed",
    )
    wants_compliance = task_mode == "Compliance Mode" or any(
        term in inp.lower() for term in compliance_terms
    )

    if wants_compliance:
        return analyze_compliance(context, inp, cache_scope=country_filter)

    return generate_answer(context, inp, cache_scope=country_filter)
