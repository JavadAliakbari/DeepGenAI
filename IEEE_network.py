# paper_graph_builder.py
# A script to search papers on a given topic using IEEE Xplore,
# extract each paper's references, and list them.

# Requirements:
# pip install requests argparse

import requests
import argparse

# IEEE Xplore REST API endpoints
IEEE_SEARCH_URL = "https://ieeexplore.ieee.org/search"
IEEE_REFERENCES_URL = "https://ieeexplore.ieee.org/rest/document/{doc_id}/references"

# Common headers
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0",
}


def search_papers_ieee(query, max_results=10):
    """
    Search IEEE Xplore for papers matching `query`.
    Returns a list of dicts with 'id' and 'title'.
    """
    payload = {
        "queryText": query,
        "highlight": True,
        "returnType": "SEARCH",
        "matchPubs": True,
        "rowsPerPage": max_results,
        "pageNumber": 1,
    }
    resp = requests.post(IEEE_SEARCH_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records", [])
    papers = []
    for rec in records:
        paper_id = rec.get("documentNumber")
        title = rec.get("articleTitle")
        if paper_id and title:
            papers.append({"id": paper_id, "title": title})
    return papers


def get_references_ieee(doc_id):
    """
    Fetch the list of references for a paper by its IEEE document ID.
    Returns a list of dicts with reference info.
    """
    url = IEEE_REFERENCES_URL.format(doc_id=doc_id)
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    refs = data.get("references", [])
    references = []
    for ref in refs:
        title = ref.get("referencedDocumentTitle") or ref.get("referenceText")
        authors = ref.get("referencedDocumentAuthors")
        doi = ref.get("referencedDocumentDOI")
        references.append({"title": title, "authors": authors, "doi": doi})
    return references


def main():
    parser = argparse.ArgumentParser(
        description="Search IEEE Xplore and list each paper's references"
    )
    parser.add_argument(
        "--topic", type=str, default="Graph Neural Network", help="Topic to search for"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of papers to fetch"
    )
    args = parser.parse_args()

    print(f"Searching IEEE Xplore for topic: '{args.topic}'...")
    papers = search_papers_ieee(args.topic, args.limit)
    print(f"Found {len(papers)} papers.\n")

    for idx, paper in enumerate(papers, 1):
        print(f"{idx}. {paper['title']} (ID: {paper['id']})")
        try:
            refs = get_references_ieee(paper["id"])
            print(f"   References ({len(refs)}):")
            for r in refs:
                authors = (
                    ", ".join(a.get("name") for a in r["authors"])
                    if r["authors"]
                    else "N/A"
                )
                doi = r["doi"] or "N/A"
                print(f"     - {r['title']} by {authors} | DOI: {doi}")
        except Exception as e:
            print(f"   Could not fetch references: {e}")
        print()


if __name__ == "__main__":
    main()
