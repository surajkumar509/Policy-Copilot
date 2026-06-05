from collections import deque
from typing import Dict, List, Set
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from lxml import html


def _normalize_http_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("URL is required.")

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError("Please provide a valid URL.")

    clean, _ = urldefrag(url)
    return clean.rstrip("/")


def _extract_page_text(content: str) -> str:
    tree = html.fromstring(content)
    for node in tree.xpath("//script|//style|//noscript|//svg"):
        node.drop_tree()
    return " ".join(tree.text_content().split())


def _extract_same_domain_links(
    seed_url: str, current_url: str, content: str
) -> List[str]:
    seed_host = urlparse(seed_url).netloc
    tree = html.fromstring(content)
    links: List[str] = []

    for href in tree.xpath("//a/@href"):
        href = (href or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        target = urljoin(current_url, href)
        target, _ = urldefrag(target)
        parsed = urlparse(target)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != seed_host:
            continue
        links.append(target.rstrip("/"))

    return links


def crawl_site_documents(
    seed_url: str, max_pages: int = 8, timeout: int = 12
) -> List[Dict[str, str]]:
    normalized_seed = _normalize_http_url(seed_url)
    max_pages = max(1, min(int(max_pages), 25))

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Policy-Copilot/1.0 (+website-analysis)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    queue = deque([normalized_seed])
    visited: Set[str] = set()
    docs: List[Dict[str, str]] = []
    request_failures = 0
    http_status_failures = 0
    saw_non_html = False

    while queue and len(visited) < max_pages:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        try:
            response = session.get(current, timeout=timeout)
            response.raise_for_status()

            content_type = (response.headers.get("content-type") or "").lower()
            if "text/html" not in content_type:
                saw_non_html = True
                continue

            page_text = _extract_page_text(response.text)
            if len(page_text) < 120:
                continue

            docs.append({"source": f"web::{current}", "text": page_text})

            for link in _extract_same_domain_links(
                normalized_seed, current, response.text
            ):
                if link not in visited:
                    queue.append(link)

        except requests.HTTPError as exc:
            http_status_failures += 1

            # Surface clearer guidance for private/blocked URLs.
            if current == normalized_seed and exc.response is not None:
                status_code = exc.response.status_code
                if status_code in (401, 403):
                    raise RuntimeError(
                        "The URL is not publicly accessible (authentication required)."
                    )

        except requests.RequestException:
            request_failures += 1
            continue
        except Exception:
            continue

    if not docs:
        if request_failures > 0 and len(visited) <= 1:
            raise RuntimeError(
                "Unable to reach the website URL. Check that the URL is valid and publicly accessible."
            )

        if http_status_failures > 0 and len(visited) <= 1:
            raise RuntimeError(
                "The website returned an error status and could not be analyzed."
            )

        if saw_non_html:
            raise RuntimeError("The URL does not appear to be a readable HTML page.")

        raise RuntimeError(
            "No readable pages were found. Use a publicly accessible HTML URL."
        )

    return docs
