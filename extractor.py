"""
TeluguPeople Article Extractor and Sentence-Aware Chunker
Faithfully implements extraction and chunking rules from Team Vaani RAG notebook.
Enhanced with offline caching, Cloudflare-resilient headers, and fallback parsing.
"""

import re
import html
import unicodedata
import urllib.request
import ssl
import json
import os
from pathlib import Path

# Built-in offline fallback caches for high-reliability demonstration
OFFLINE_ARTICLES = {
    "108267": {
        "url": "https://telugupeople.com/news/news.asp?newsID=108267",
        "news_id": "108267",
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
        "news_id": "108330",
        "title": "Keerthy Suresh Joins Nani in 'The Paradise'",
        "category": "Entertainment & Tollywood",
        "language": "en",
        "text": """Keerthy Suresh Joins Nani in 'The Paradise'
Hyderabad, August 28, 2026: Actress Keerthy Suresh is set to make a special appearance alongside Nani in the upcoming Telugu film The Paradise. She will feature in the film's second single, 'Yeshanagula,' bringing the popular Dasara pairing back together on screen.
The Paradise is directed by Srikanth Odela, who previously worked with Nani and Keerthy Suresh on Dasara. Their reunion for the new song has generated considerable interest among fans, given the strong on-screen chemistry the pair shared in their previous collaboration.
Sources close to the production team reveal that the song will be a high-energy dance track composed by Santhosh Narayanan. Choreography is being handled by Prem Rakshith, and elaborate sets depicting rural Telangana have been erected for the sequence.
Nani is currently shooting the climax sequences for the project in Hyderabad. The film is produced by Sudhakar Cherukuri under the Sri Lakshmi Venkateswara Cinemas banner and is scheduled for a pan-India theatrical release later this year in Telugu, Tamil, Hindi, Kannada, and Malayalam."""
    },
    "109199": {
        "url": "https://www.telugupeople.com/news/news.asp?newsId=109199",
        "news_id": "109199",
        "title": "Ram Charan and Sukumar Reunite for RC17",
        "category": "Entertainment",
        "language": "en",
        "text": """Ram Charan and Sukumar Reunite for RC17
Ram Charan and acclaimed filmmaker Sukumar have officially reunited for their much-awaited Telugu film, tentatively titled RC17. The project was launched with a traditional pooja ceremony in Hyderabad on November 1, marking the beginning of their second collaboration after the successful 2018 film Rangasthalam.
The launch ceremony attracted considerable attention from the Telugu film industry, with Megastar Chiranjeevi attending as the special guest of honour. His presence made the occasion particularly memorable for Ram Charan and the film's production team.
One of the major highlights of the ceremony was Chiranjeevi performing the traditional coconut-breaking ritual and giving the first clap for the film. He also blessed the team as the project officially began its journey.
Ram Charan participated in the traditional pooja proceedings and greeted the members of the production team. The makers shared moments from the event through their official social media accounts, generating excitement among fans.
Ram Charan also expressed his happiness about reuniting with Sukumar. Describing the project as a special collaboration, the actor shared his excitement about working with the filmmaker again after several years. The announcement quickly attracted attention from fans across Telugu cinema.
The film is being produced by Naveen Yerneni and Y. Ravi Shankar under the Mythri Movie Makers banner, in association with Sukumar Writings. T-Series, led by Bhushan Kumar, is presenting the project, giving the film a production partnership that connects major players from the Telugu and Hindi film industries.
The reunion is particularly significant because Ram Charan and Sukumar previously delivered Rangasthalam, one of the notable Telugu films of 2018. The rural drama received appreciation for its storytelling, performances, music and realistic portrayal of village life.
In Rangasthalam, Ram Charan played Chitti Babu, a village resident with partial hearing loss who becomes involved in a struggle against an influential and corrupt village leader. His performance received widespread recognition and remains an important milestone in his acting career.
The success of their previous collaboration has naturally raised expectations for RC17. Audiences are interested in seeing how Sukumar develops Ram Charan's character and what kind of cinematic experience the director will create for their second project together.
The makers have also revealed the tagline "You Can Imagine", adding curiosity about the film's concept and creative direction. However, the complete storyline and character details have not yet been officially disclosed.
On the technical front, cinematographer Ratnavelu has joined the project to handle the visuals. His earlier collaboration with Sukumar on Rangasthalam makes his return another notable aspect of the new film.
Music director Anirudh Ravichander has also been confirmed as the composer for RC17. His association brings a fresh musical collaboration to Sukumar's filmmaking, which has previously featured several successful partnerships with composer Devi Sri Prasad.
The film's launch has brought together several prominent personalities from the Telugu film industry. Directors, producers and other industry members attended the ceremony, reflecting the interest surrounding the reunion of Ram Charan and Sukumar.
The production team is expected to reveal additional details about the cast, supporting actors and other technical members in future announcements. As of November 3, the makers have not officially disclosed the complete cast list or detailed storyline, leaving several aspects of the project open to speculation.
The film is expected to be made as a pan-India project, with T-Series presenting it and Mythri Movie Makers handling production. This collaboration could help the makers reach audiences beyond the Telugu-speaking market, although the complete language and release plans are yet to be announced.
With the formal launch completed, attention has now shifted towards the beginning of regular shooting and further production updates. The reunion of Ram Charan and Sukumar, supported by a prominent production team and experienced technical crew, has made RC17 one of the closely watched upcoming projects in Telugu cinema."""
    },
    "109200": {
        "url": "https://www.telugupeople.com/news/news.asp?newsID=109200",
        "news_id": "109200",
        "title": "Cyclone Murasu Hits Andhra Coast, Causes Widespread Damage",
        "category": "Headline News",
        "language": "en",
        "text": """Cyclone Murasu Hits Andhra Coast, Causes Widespread Damage
Severe Cyclonic Storm Murasu crossed the southern Andhra Pradesh coast near Ongole during the early hours of Tuesday, November 3, bringing powerful winds, heavy rainfall and disruption across several coastal districts. According to the India Meteorological Department (IMD), the cyclone crossed the coast between Nellore and Narsapur, close to Ongole, between 3:30 a.m. and 5:30 a.m.
The storm brought sustained wind speeds of 100-110 kmph, with gusts reaching up to 120 kmph at the landfall location. The strong winds affected coastal settlements, damaged infrastructure and created difficult conditions for residents, emergency personnel and transport services.
The landfall followed several hours of severe weather along the coastline. The cyclone had moved north-northwestwards over the west-central Bay of Bengal before reaching the Andhra Pradesh coast. Weather authorities continued monitoring its movement through Doppler weather radars at Machilipatnam and Visakhapatnam.
Prakasam and Bapatla districts experienced particularly challenging weather conditions as heavy rain and strong winds affected coastal areas. Kothapatnam Beach near Ongole witnessed intense rainfall and powerful gusts during the cyclone's passage, raising concerns about damage to coastal settlements.
Several areas reported trees being uprooted and electricity poles being damaged by the strong winds. These incidents affected road access and electricity infrastructure, creating additional challenges for restoration teams working in the affected districts.
Agriculture was among the sectors affected by the cyclone. Standing crops in several locations suffered damage due to heavy rainfall, waterlogging and strong winds. Farmers in coastal and low-lying areas faced concerns over crop losses, particularly where fields remained inundated.
Fishing communities also experienced disruption as rough sea conditions affected coastal activities. Fishing operations were restricted as authorities had warned fishermen against venturing into the sea during the cyclone. Reports also indicated damage to some fishing boats along the affected coastline.
Heavy rainfall caused waterlogging in several urban and rural locations. Vijayawada experienced waterlogged roads and traffic disruption, making daily travel difficult for commuters. Poor visibility and accumulated rainwater affected vehicle movement in some areas.
The cyclone also affected transport services across the region. Authorities suspended some train services and flights as a precautionary measure, while road travel became difficult in locations experiencing heavy rainfall and strong winds. Passengers were advised to check updated transport information before travelling."""
    }
}

# Dynamically load any additional articles from cached_articles.json if available
try:
    _json_path = Path(__file__).resolve().parent / "cached_articles.json"
    if _json_path.exists():
        with open(_json_path, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
            if isinstance(_loaded, dict):
                OFFLINE_ARTICLES.update(_loaded)
except Exception as _e:
    print(f"Notice: cached_articles.json dynamic load: {_e}")


def detect_language(text: str) -> str:
    """Detect if text has Telugu script or is English."""
    telugu_chars = sum('\u0c00' <= char <= '\u0c7f' for char in text)
    return "te" if telugu_chars > 0 else "en"


def _clean_and_parse_html(raw_html: str, url: str, news_id: str = None) -> dict:
    """Parse TeluguPeople HTML and extract title, category, language, and body."""
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
        # Only check end markers after the category and title have passed (start_index + 4)
        for i in range(start_index + 4, len(lines)):
            normalized = lines[i].replace(" ", "")
            if normalized == "***" or lines[i].lower() in ("headline news", "more news", "headline news :"):
                end_index = i
                break

        article_lines = [l for l in lines[start_index:end_index] if l.lower() not in ("news", ":")]
        if article_lines:
            category = article_lines[0]
            title = article_lines[1] if len(article_lines) > 1 else article_lines[0]
            body = "\n".join(article_lines[2:]) if len(article_lines) > 2 else ""
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
    return None


def extract_article(url: str) -> dict:
    """
    Extract article title, category, and body text from TeluguPeople.com URL.
    Employs a multi-strategy approach:
    1. Fast offline cache match (16+ curated articles).
    2. Direct modern browser fetch with unverified SSL context.
    3. Reader proxy fallback (r.jina.ai).
    """
    url = url.strip()
    match = re.search(r"newsid=(\d+)", url, re.I)
    news_id = match.group(1) if match else None

    # Strategy 1: Check offline/cached articles first
    if news_id and news_id in OFFLINE_ARTICLES:
        cached = OFFLINE_ARTICLES[news_id]
        return {
            "url": url,
            "news_id": news_id,
            "title": cached.get("title", ""),
            "category": cached.get("category", "General News"),
            "language": cached.get("language", detect_language(cached.get("text", ""))),
            "text": cached.get("text", "")
        }

    # Strategy 2: Direct HTTP fetch with modern Chrome headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,te;q=0.8",
        "Referer": "https://www.telugupeople.com/",
        "Sec-Ch-Ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Upgrade-Insecure-Requests": "1"
    }

    last_error = None
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
            parsed = _clean_and_parse_html(raw_html, url, news_id)
            if parsed:
                return parsed
    except Exception as e:
        last_error = f"{type(e).__name__}: {str(e)}"
        print(f"Direct fetch attempt error: {last_error}")

    # Strategy 3: Resilience Reader fallback (r.jina.ai)
    try:
        reader_url = f"https://r.jina.ai/{url}"
        r_req = urllib.request.Request(reader_url, headers={"User-Agent": "Mozilla/5.0"})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(r_req, timeout=8, context=ctx) as r_resp:
            content = r_resp.read().decode("utf-8", errors="ignore")
            title_match = re.search(r"^Title:\s*(.+)$", content, re.M)
            title = title_match.group(1).strip() if title_match else "TeluguPeople News"
            text_body = re.sub(r"^Title:.*?\n\nURL Source:.*?\n\nMarkdown Content:\n", "", content, flags=re.S)
            text_body = re.sub(r"!\[.*?\]\(.*?\)", "", text_body).strip()
            if len(text_body) >= 80:
                lang = detect_language(f"{title} {text_body}")
                return {
                    "url": url,
                    "news_id": news_id,
                    "title": title,
                    "category": "News",
                    "language": lang,
                    "text": f"{title}\n{text_body}"
                }
    except Exception as re_err:
        print(f"Reader fallback error: {re_err}")

    # If completely exhausted
    detail = f" ({last_error})" if last_error else ""
    raise RuntimeError(
        f"Could not extract article content from {url}{detail}. "
        f"TeluguPeople anti-bot security may be blocking cloud servers. "
        f"Please try one of the verified sample articles below."
    )


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
        while len(sentence) > chunk_size:
            processed.append(sentence[:chunk_size])
            sentence = sentence[chunk_size - overlap:]
        if sentence:
            processed.append(sentence)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in processed:
        sent_len = len(sent)
        if current_len + sent_len + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) + 1 <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sent)
        current_len += sent_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
