"""
TeluguPeople Article Extractor and Sentence-Aware Chunker
Faithfully implements extraction and chunking rules from Team Vaani RAG notebook.
"""

import re
import html
import unicodedata
import urllib.request

# Offline fallback caches for the demonstration articles
OFFLINE_ARTICLES = {
    "108267": {
        "url": "https://telugupeople.com/news/news.asp?newsID=108267",
        "title": "హోంవర్క్‌ విషయంలో ఉపాధ్యాయురాలి మందలింపు.. 6 ఏళ్ల బాలుడు మృతి",
        "category": "వార్తలు (Regional News)",
        "language": "te",
        "text": """హోంవర్క్‌ విషయంలో ఉపాధ్యాయురాలి మందలింపు.. 6 ఏళ్ల బాలుడు మృతి
విశాఖపట్నంలో ఆరు సంవత్సరాల బాలుడి మృతి తీవ్ర విషాదాన్ని, ఆందోళనను కలిగించింది. హోంవర్క్ పూర్తి చేయలేదనే కారణంతో ఉపాధ్యాయురాలు బాలుడిని మందలించి కొట్టినట్లు కుటుంబ సభ్యులు ఆరోపిస్తున్నారు. ఆ ఘటన అనంతరం బాలుడు తరగతి గదిలోనే కుప్పకూలి మృతి చెందినట్లు పలు వార్తా సంస్థలు వెల్లడించాయి.
మృతుడిని ప్రణయ్ తేజగా గుర్తించారు. అతను విశాఖపట్నంలోని వన్‌టౌన్ ప్రాంతంలోని లిటిల్ గార్డెన్ స్కూల్‌లో చదువుతున్నట్లు నివేదికలు పేర్కొన్నాయి. పాఠశాలలో జరిగిన ఈ దురదృష్టకర సంఘటనపై విద్యార్థి తల్లిదండ్రులు తీవ్ర ఆగ్రహం వ్యక్తం చేస్తూ ఆందోళనకు దిగారు. ఉపాధ్యాయురాలి నిర్లక్ష్యం, అధిక దండన కారణంగానే తమ కుమారుడు మృతి చెందాడని ఆరోపించారు.
సమాచారం అందుకున్న పోలీసులు సంఘటనా స్థలానికి చేరుకుని పరిస్థితిని సమీక్షించారు. బాలుడి మృతదేహాన్ని పోస్టుమార్టం నిమిత్తం ప్రభుత్వ ఆసుపత్రికి తరలించారు. ఈ ఘటనపై కేసు నమోదు చేసుకుని పోలీసులు దర్యాప్తు ప్రారంభించారు. పోస్టుమార్టం నివేదిక వచ్చిన తర్వాతే బాలుడి మృతికి గల ఖచ్చితమైన కారణాలు తెలుస్తాయని అధికారులు తెలిపారు. స్థానిక ప్రజలు, విద్యార్థి సంఘాలు పాఠశాల యాజమాన్యంపై కఠిన చర్యలు తీసుకోవాలని డిమాండ్ చేస్తున్నాయి."""
    },
    "108330": {
        "url": "https://telugupeople.com/news/news.asp?newsId=108330",
        "title": "Keerthy Suresh Joins Nani in 'The Paradise'",
        "category": "Entertainment & Tollywood",
        "language": "en",
        "text": """Keerthy Suresh Joins Nani in 'The Paradise'
Hyderabad, August 28, 2026: Actress Keerthy Suresh is set to make a special appearance alongside Nani in the upcoming Telugu film The Paradise. She will feature in the film's second single, 'Yeshanagula,' bringing the popular Dasara pairing back together on screen.
The Paradise is directed by Srikanth Odela, who previously worked with Nani and Keerthy Suresh on Dasara. Their reunion for the new song has generated considerable interest among fans, given the strong on-screen chemistry the pair shared in their previous collaboration.
Sources close to the production team reveal that the song will be a high-energy dance track composed by Santhosh Narayanan. Choreography is being handled by Prem Rakshith, and elaborate sets depicting rural Telangana have been erected for the sequence.
Nani is currently shooting the climax sequences for the project in Hyderabad. The film is produced by Sudhakar Cherukuri under the Sri Lakshmi Venkateswara Cinemas banner and is scheduled for a pan-India theatrical release later this year in Telugu, Tamil, Hindi, Kannada, and Malayalam."""
    }
}


def detect_language(text: str) -> str:
    """Detect if text has Telugu script or is English."""
    telugu_chars = sum('\u0c00' <= char <= '\u0c7f' for char in text)
    return "te" if telugu_chars > 0 else "en"


def extract_article(url: str) -> dict:
    """
    Extract article title, category, and body text from TeluguPeople.com URL.
    Includes fallbacks for offline demo resilience.
    """
    url = url.strip()
    match = re.search(r"newsid=(\d+)", url, re.I)
    news_id = match.group(1) if match else None

    # Check cache first for instant load or fallback
    cached = OFFLINE_ARTICLES.get(news_id)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
        
        text = html.unescape(raw_html)
        clean = re.sub(r'<script.*?</script>', '', text, flags=re.S|re.I)
        clean = re.sub(r'<style.*?</style>', '', clean, flags=re.S|re.I)
        clean = re.sub(r'<[^>]+>', '\n', clean)

        lines = [line.strip() for line in clean.splitlines() if line.strip()]

        start_index = None
        for i, line in enumerate(lines):
            if re.match(r"^News\s*:", line, re.IGNORECASE):
                start_index = i
                break
            if line.lower() == "news" and i + 1 < len(lines) and lines[i + 1] == ":":
                start_index = i
                break

        if start_index is not None:
            end_index = len(lines)
            for i in range(start_index + 1, len(lines)):
                normalized = lines[i].replace(" ", "")
                if normalized == "***" or lines[i].lower() == "headline news":
                    end_index = i
                    break

            article_lines = lines[start_index:end_index]
            # Skip initial 'News' and ':' tokens
            cleaned_lines = []
            for line in article_lines:
                if line.lower() in ("news", ":"):
                    continue
                cleaned_lines.append(line)

            if cleaned_lines:
                title = cleaned_lines[0]
                category = "General News"
                if len(cleaned_lines) > 1 and len(cleaned_lines[0]) < 35:
                    category = cleaned_lines[0]
                    title = cleaned_lines[1]
                    body = "\n".join(cleaned_lines[2:])
                else:
                    body = "\n".join(cleaned_lines[1:])

                full_text = f"{title}\n{body}".strip()
                if len(full_text) >= 80:
                    lang = detect_language(full_text)
                    return {
                        "url": url,
                        "news_id": news_id,
                        "title": title,
                        "category": category,
                        "language": lang,
                        "text": full_text
                    }
    except Exception as e:
        print(f"Extraction network error ({e}), falling back to cached article if available.")

    if cached:
        return cached

    # If completely unknown URL and failed to fetch
    raise RuntimeError(f"Could not extract article content from {url}. Please verify the URL or try the sample articles.")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """
    Exact sentence-aware chunking function from Team Vaani RAG notebook (Cell 29).
    Preserves sentence boundaries while enforcing max chunk length.
    """
    if not text or not text.strip():
        return []

    # Normalize unicode (NFC) and whitespace
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentences while retaining punctuation
    sentences = re.split(r'(?<=[.!?।])\s+', text)

    processed = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= chunk_size:
            processed.append(sentence)
        else:
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if len(candidate) <= chunk_size:
                    piece = candidate
                else:
                    if piece:
                        processed.append(piece)
                    piece = word
            if piece:
                processed.append(piece)

    chunks = []
    current = ""
    for sentence in processed:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            overlap_text = ""
            if overlap > 0 and current:
                words = current.split()
                selected = []
                for word in reversed(words):
                    trial = " ".join([word] + selected)
                    if len(trial) > overlap:
                        break
                    selected.insert(0, word)
                overlap_text = " ".join(selected)

            current = f"{overlap_text} {sentence}".strip()
            if len(current) > chunk_size:
                current = sentence

    if current:
        chunks.append(current)

    return chunks
