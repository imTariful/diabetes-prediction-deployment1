# (Only showing the additions/changes to the previous code)

# --- Heuristic extraction helpers ---
date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
number_re = re.compile(r"\b\d[\d,./-]{1,50}\b")
zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b")
address_re = re.compile(r"\d+\s+[A-Za-z0-9\s]+")  # simple house number + street
city_state_re = re.compile(r"[A-Za-z]+")  # simple city/state detection

def heuristic_extract(field: str, text: str) -> dict:
    # Simple keyword search for field
    tokens = [t.lower() for t in re.findall(r"\w+", field) if len(t) > 2]
    best_line = ""
    best_score = 0
    for line in text.splitlines():
        score = sum(1 for t in tokens if t in line.lower())
        if score > best_score:
            best_score = score
            best_line = line.strip()
    if not best_line:
        return {"value":"Not Found","confidence":0.3,"source_snippet":""}

    val = "Not Found"
    # Try regex based on field name
    lname = field.lower()
    if "date" in lname:
        m = date_re.search(best_line)
        if m: val = m.group(0)
    elif "zip" in lname:
        m = zip_re.search(best_line)
        if m: val = m.group(0)
    elif "street" in lname or "address" in lname:
        m = address_re.search(best_line)
        if m: val = m.group(0)
    elif "city" in lname or "state" in lname:
        m = city_state_re.search(best_line)
        if m: val = m.group(0)
    elif "number" in lname or "phone" in lname:
        m = number_re.search(best_line)
        if m: val = m.group(0)
    else:
        # fallback take RHS of colon/dash
        if ":" in best_line:
            val = best_line.split(":",1)[1].strip()
        elif "-" in best_line:
            val = best_line.split("-",1)[1].strip()
        else:
            val = best_line
    return {"value": val, "confidence": min(0.8,0.3+0.1*best_score), "source_snippet": best_line}

# --- Integrate with main loop ---
for i, field in enumerate(placeholders):
    # RAG retrieval
    query_emb = embed_text([field])[0].reshape(1,-1)
    D,I = index.search(query_emb, top_k)
    top_chunks = [chunks[idx] for idx in I[0] if idx < len(chunks)]
    context = "\n".join(top_chunks)

    result = call_gemini_for_field(field, context)

    # Fallback if confidence <0.7
    if result.get("confidence",0) < 0.7:
        heur = heuristic_extract(field, full_text)
        # Weighted: prefer LLM if it has some confidence
        if heur["confidence"] > result.get("confidence",0):
            result = heur
        else:
            # merge both
            result["confidence"] = max(result.get("confidence",0), heur["confidence"])
            if result.get("value") in ["Not Found",""]:
                result["value"] = heur["value"]
            if not result.get("source_snippet"):
                result["source_snippet"] = heur["source_snippet"]

    extracted_data[field] = result
    progress.progress(50 + int(40*(i+1)/len(placeholders)))
    status.text(f"Processed {i+1}/{len(placeholders)}: {field}")
