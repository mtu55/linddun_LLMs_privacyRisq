import xml.etree.ElementTree as ET
import yaml, re
from collections import Counter

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

def build_index(tax_path):
    """xmi:id -> (kategorie_abk, pfad, name) aus der Taxonomie."""
    XMI = "{http://www.omg.org/XMI}id"
    root = ET.parse(tax_path).getroot()
    idx = {}

    def walk(node, abbr, path):
        for ch in node.findall("characteristic"):
            p = path + [ch.get("name")]
            idx[ch.get(XMI)] = {
                "category": abbr,
                "path": " > ".join(p),
                "node_name": ch.get("name"),
                "node_description": ch.get("description"),
                "node_id": f"{abbr}." + ".".join(slug(x) for x in p),
            }
            walk(ch, abbr, p)

    for tt in root.findall("threatType"):
        abbr = tt.get("abbreviation")
        idx[tt.get(XMI)] = {
            "category": abbr, "path": tt.get("name"),
            "node_name": tt.get("name"),
            "node_description": tt.get("description"),
            "node_id": abbr,
        }
        walk(tt, abbr, [])
    return idx

def load_anchors(coll_path, idx):
    root = ET.parse(coll_path).getroot()
    out = []
    for th in root.findall("threat"):
        chars = []
        for tc in th.findall("threatcharacteristic"):
            link = tc.find("characteristic")
            ref = (link.get("href") or "").split("#")[-1] if link is not None else None
            meta = idx.get(ref, {})
            chars.append({
                "indicator": tc.get("name"),
                "rationale": tc.get("description"),
                "category": meta.get("category", "UNRESOLVED"),
                "node_id": meta.get("node_id"),
                "node_path": meta.get("path"),
            })
        out.append({
            "case": th.get("name"),
            "summary": th.get("description"),
            "detail": th.get("fullDescription"),
            "refs": [r.text.strip() for r in th.findall("refs") if r.text],
            "categories": sorted({c["category"] for c in chars}),
            "characteristics": chars,
        })
    return out

if __name__ == "__main__":
    idx = build_index("original_fixed.xml")
    anchors = load_anchors("examples.xml", idx)

    unresolved = [c["indicator"] for a in anchors
                  for c in a["characteristics"] if c["category"] == "UNRESOLVED"]
    if unresolved:
        print("WARN nicht aufgelöste hrefs:", unresolved)

    print("Kategorie-Häufigkeit:",
          Counter(c["category"] for a in anchors for c in a["characteristics"]))
    print("Kategorien pro Fall:",
          Counter(len(a["categories"]) for a in anchors))

    with open("linddun_anchors.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"anchors": anchors}, f,
                       allow_unicode=True, sort_keys=False, width=100)