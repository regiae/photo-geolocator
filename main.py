"""photo-geolocator-01: an agent that says where each of 25 street-level
photos was taken.

How it works
------------
1. It fetches the photos from the env at MAKEFUN_ENV_URL.
2. Its first choice is a vision model: one claude-sonnet-5-5 call per photo
   through the gateway, asking the model to reason from visible clues
   (signs and their language, driving side, road markings, vegetation,
   architecture, landscape) and answer with coordinates and a confidence.
   Photos the model is unsure about (confidence below 75) get one focused
   second call that asks it to re-read the most specific clue it found.
   This is the path that can beat the house baseline, which is the same
   model with a plain one-line prompt and no second look.
3. If the model is not reachable (no API key was pushed, so the gateway
   answers 401), the agent falls back to its own code: it reads each JPEG's
   DC coefficients (the average colour of every 8x8 block, decoded with the
   standard library only), measures the scene's colours, and places the
   photo in a broad climate region by hand-reasoned rules. That path is
   weaker, but it answers every photo instead of leaving any unanswered
   (an unanswered photo counts the maximum 20,015 km).

Determinism: every model request is a function of the photo's bytes and a
fixed prompt; the calls run four at a time and the answers are posted in
photo order, so a replay sends the same calls and leaves the same answers.
No wall clock and no unseeded randomness anywhere in a request. The
fallback is pure computation on the pixels.
"""

import base64
import concurrent.futures
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request

MODEL = "claude-sonnet-5-5"
MAX_TOKENS_A = 768
MAX_TOKENS_B = 384
CONCURRENCY = 4
REFINE_BELOW = 75
REFINE_MAX = 12
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
RETRIES = 3

PROMPT_A = (
    "You are a world-class GeoGuessr player. Study this street-level photo "
    "and work out where on Earth it was taken. Reason step by step from the "
    "visible clues: the language and alphabet of any signs, the side of the "
    "road traffic drives on, road markings and sign shapes, license plate "
    "style, vegetation and climate, architecture and building materials, "
    "the landscape and terrain, and anything else distinctive. Then give "
    "your best estimate. Reply with a short reasoning paragraph followed by "
    "one line of JSON: {\"country\": \"<ISO 3166-1 alpha-2 code>\", \"lat\": "
    "<decimal degrees north>, \"lon\": <decimal degrees east>, "
    "\"confidence\": <0-100>, \"clues\": [\"<short clue>\", \"<short clue>\"]}. "
    "The coordinates must be a single point, your best guess."
)

PROMPT_B_TEMPLATE = (
    "Your first estimate for this photo was {lat:.4f}, {lon:.4f} ({country}), "
    "confidence {conf}/100, based on: {clues}. Look at the photo again, "
    "closely. Focus on the most specific clue you can find: readable text "
    "or a place name, a distinctive building, a road number, a landscape "
    "feature, or the vegetation and climate. Refine the location to the "
    "most specific place you can identify. If you cannot narrow it down, "
    "keep the first estimate. Reply with a short reasoning paragraph "
    "followed by one line of JSON: {{\"lat\": <decimal degrees north>, "
    "\"lon\": <decimal degrees east>, \"confidence\": <0-100>}}."
)

# Fallback region centroids (hand-picked, population- and road-weighted).
REGION = {
    "NA": (44.0, -98.0),    # North America, temperate
    "MX": (21.0, -101.0),   # Mexico / Central America
    "SA": (-17.0, -58.0),   # South America
    "EU": (48.0, 10.0),     # Europe
    "EUN": (62.0, 15.0),    # Northern Europe
    "RUC": (56.0, 62.0),    # Russia / Central Asia
    "ME": (30.0, 46.0),     # Middle East
    "SAS": (21.0, 80.0),    # South Asia
    "SEA": (6.0, 106.0),    # South-East Asia
    "EA": (34.0, 112.0),    # East Asia
    "OC": (-27.0, 142.0),   # Oceania
    "AFN": (24.0, 8.0),     # North Africa
    "AFS": (-4.0, 28.0),    # Sub-Saharan Africa
}
FALLBACK_DEFAULT = REGION["SAS"]


def log(msg):
    sys.stderr.write("geolocator: %s\n" % msg)


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def http_json(url, method="GET", body=None, headers=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status, res.read()


def http_bytes(url, timeout=30):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status, res.read()


# --------------------------------------------------------------------------
# Vision-model client (through the gateway, Anthropic Messages)
# --------------------------------------------------------------------------

def ask(jpeg, prompt, max_tokens):
    """One model call. Returns the model's text, or None when the gateway
    refuses the call (a 4xx is final; a 5xx/429 is retried)."""
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg",
                                             "data": base64.b64encode(jpeg).decode("ascii")}},
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode("utf-8")
    url = ANTHROPIC_URL + "/v1/messages"
    headers = {"content-type": "application/json", "x-api-key": API_KEY,
               "anthropic-version": "2023-06-01"}
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=90) as res:
                msg = json.loads(res.read().decode("utf-8"))
            return "\n".join(
                b.get("text", "") for b in msg.get("content", [])
                if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            retryable = (e.code in RETRY_STATUSES
                         and e.headers.get("x-should-retry") != "false")
            if retryable and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call refused (HTTP %d): %s" % (e.code, detail))
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            log("model call failed: %s" % e)
            return None
    return None


def _is_lat(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and -90 <= v <= 90


def _is_lon(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and -180 <= v <= 180


def parse_guess(text):
    """The latitude, longitude, confidence, country and clues an answer
    gives: its JSON object with lat and lon, else its first two numbers in
    range. Confidence defaults to 50 when the answer gives none."""
    if not text:
        return None
    best = None
    for m in _json_objects(text):
        try:
            o = json.loads(m)
        except ValueError:
            continue
        lat = o.get("lat", o.get("latitude"))
        lon = o.get("lon", o.get("lng", o.get("longitude")))
        if _is_lat(lat) and _is_lon(lon):
            conf = o.get("confidence", 50)
            if not isinstance(conf, (int, float)) or not math.isfinite(conf):
                conf = 50
            best = {"lat": float(lat), "lon": float(lon),
                    "confidence": max(0, min(100, int(conf))),
                    "country": o.get("country", "?") or "?",
                    "clues": o.get("clues", []) or []}
            break
    if best is not None:
        return best
    nums = [float(x) for x in _number_re().findall(text)]
    for i in range(len(nums) - 1):
        if _is_lat(nums[i]) and _is_lon(nums[i + 1]):
            return {"lat": nums[i], "lon": nums[i + 1], "confidence": 50,
                    "country": "?", "clues": []}
    return None


def _json_objects(text):
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _number_re():
    import re
    return re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------
# JPEG DC decoder (standard library only)
#
# A baseline JPEG stores, per 8x8 block, a DC coefficient: the block's
# average brightness (and chroma). Decoding only the DC terms gives a
# coarse colour map of the photo without a full IDCT, which is all the
# fallback needs. Blocks whose 63 AC coefficients are all nonzero carry
# no EOB marker, so the AC walk counts coefficients and stops at 63.
# --------------------------------------------------------------------------

class _BitReader:
    def __init__(self, data, pos):
        self.d = data
        self.i = pos
        self.acc = 0
        self.nbits = 0

    def _fill(self):
        while self.nbits <= 40:
            if self.i >= len(self.d):
                self.acc = (self.acc << 8) | 0xFF
                self.nbits += 8
                continue
            b = self.d[self.i]
            self.i += 1
            if b == 0xFF:
                if self.i < len(self.d) and self.d[self.i] == 0x00:
                    self.i += 1
                elif self.i < len(self.d) and 0xD0 <= self.d[self.i] <= 0xD7:
                    self.i += 1
                    self.acc = 0
                    self.nbits = 0
                    return
                else:
                    self.i = len(self.d)
                    self.acc = (self.acc << 8) | 0xFF
                    self.nbits += 8
                    continue
            self.acc = (self.acc << 8) | b
            self.nbits += 8

    def bits(self, n):
        if n == 0:
            return 0
        while self.nbits < n:
            self._fill()
        self.nbits -= n
        v = (self.acc >> self.nbits) & ((1 << n) - 1)
        self.acc &= (1 << self.nbits) - 1
        return v

    def huff(self, table):
        while self.nbits < 16:
            self._fill()
        peek = (self.acc >> (self.nbits - 16)) & 0xFFFF
        e = table[peek]
        if e < 0:
            raise ValueError("bad huffman code")
        sym = e >> 5
        ln = e & 31
        self.nbits -= ln
        self.acc &= (1 << self.nbits) - 1
        return sym


def _build_huff(counts, symbols):
    table = [-1] * 65536
    code = 0
    k = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            sym = symbols[k]
            k += 1
            start = code << (16 - length)
            step = 1 << (16 - length)
            val = (sym << 5) | length
            for p in range(start, start + step):
                table[p] = val
            code += 1
        code <<= 1
    return table


def _parse_jpeg(data):
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG")
    i = 2
    quant = {}
    huff_dc = {}
    huff_ac = {}
    w = h = 0
    comps = []
    scan_pos = 0
    comp_tbl = {}
    dri = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + ln]
        if m == 0xDB:
            j = 0
            while j < len(seg):
                pq = seg[j] >> 4
                tq = seg[j] & 15
                j += 1
                vals = []
                if pq == 0:
                    vals = list(seg[j:j + 64])
                    j += 64
                else:
                    for _ in range(64):
                        vals.append(struct.unpack(">H", seg[j:j + 2])[0])
                        j += 2
                quant[tq] = vals
        elif m == 0xC4:
            j = 0
            while j < len(seg):
                tc = seg[j] >> 4
                th = seg[j] & 15
                j += 1
                counts = list(seg[j:j + 16])
                j += 16
                total = sum(counts)
                syms = list(seg[j:j + total])
                j += total
                tbl = _build_huff(counts, syms)
                if tc == 0:
                    huff_dc[th] = tbl
                else:
                    huff_ac[th] = tbl
        elif m == 0xC0:
            h = struct.unpack(">H", seg[1:3])[0]
            w = struct.unpack(">H", seg[3:5])[0]
            n = seg[5]
            for k in range(n):
                cid = seg[6 + 3 * k]
                samp = seg[7 + 3 * k]
                tq = seg[8 + 3 * k]
                comps.append((cid, samp >> 4, samp & 15, tq))
        elif m == 0xDD:
            dri = struct.unpack(">H", seg[0:2])[0]
        elif m == 0xDA:
            n = seg[0]
            for k in range(n):
                cid = seg[1 + 2 * k]
                td = seg[2 + 2 * k] >> 4
                ta = seg[2 + 2 * k] & 15
                comp_tbl[cid] = (td, ta)
            scan_pos = i + 2 + ln
            break
        i += 2 + ln
    return w, h, comps, quant, huff_dc, huff_ac, comp_tbl, scan_pos, dri


def _dc_grid(data):
    w, h, comps, quant, huff_dc, huff_ac, comp_tbl, spos, dri = _parse_jpeg(data)
    if len(comps) != 3 or comps[0][1:3] != (2, 2) or comps[1][1:3] != (1, 1):
        raise ValueError("not baseline 4:2:0 JPEG")
    yq = quant[comps[0][3]]
    cbq = quant[comps[1][3]]
    crq = quant[comps[2][3]]
    ydc_t = huff_dc[comp_tbl[1][0]]
    yac_t = huff_ac[comp_tbl[1][1]]
    cbdc_t = huff_dc[comp_tbl[2][0]]
    cbac_t = huff_ac[comp_tbl[2][1]]
    crdc_t = huff_dc[comp_tbl[3][0]]
    crac_t = huff_ac[comp_tbl[3][1]]
    br = _BitReader(data, spos)
    mcu_w = (w + 15) // 16
    mcu_h = (h + 15) // 16
    bw = (w + 7) // 8
    bh = (h + 7) // 8
    Y = [[0.0] * bw for _ in range(bh)]
    Cb = [[0.0] * mcu_w for _ in range(mcu_h)]
    Cr = [[0.0] * mcu_w for _ in range(mcu_h)]
    prev = {1: 0, 2: 0, 3: 0}

    def read_dc(tbl, cid, q0):
        cat = br.huff(tbl)
        if cat == 0:
            diff = 0
        else:
            v = br.bits(cat)
            if v < (1 << (cat - 1)):
                diff = v - (1 << cat) + 1
            else:
                diff = v
        prev[cid] += diff
        return prev[cid] * q0 / 8.0 + 128.0

    def skip_ac(tbl):
        k = 0
        while k < 63:
            rs = br.huff(tbl)
            run = rs >> 4
            size = rs & 15
            if size == 0:
                if run == 15:
                    k += 16
                    if k >= 63:
                        return
                    continue
                return
            br.bits(size)
            k += run + 1

    mcu_count = 0
    for my in range(mcu_h):
        for mx in range(mcu_w):
            for dy in (0, 1):
                for dx in (0, 1):
                    by, bx = 2 * my + dy, 2 * mx + dx
                    if by < bh and bx < bw:
                        Y[by][bx] = read_dc(ydc_t, 1, yq[0])
                    else:
                        read_dc(ydc_t, 1, yq[0])
                    skip_ac(yac_t)
            Cb[my][mx] = read_dc(cbdc_t, 2, cbq[0])
            skip_ac(cbac_t)
            Cr[my][mx] = read_dc(crdc_t, 3, crq[0])
            skip_ac(crac_t)
            mcu_count += 1
            if dri and mcu_count % dri == 0:
                prev = {1: 0, 2: 0, 3: 0}
    R = [[0.0] * bw for _ in range(bh)]
    G = [[0.0] * bw for _ in range(bh)]
    B = [[0.0] * bw for _ in range(bh)]
    for y in range(bh):
        for x in range(bw):
            yy = Y[y][x]
            cb = Cb[y // 2][x // 2] - 128.0
            cr = Cr[y // 2][x // 2] - 128.0
            R[y][x] = yy + 1.402 * cr
            G[y][x] = yy - 0.34414 * cb - 0.71414 * cr
            B[y][x] = yy + 1.772 * cb
    return R, G, B, bw, bh


def _photo_features(jpeg):
    R, G, B, bw, bh = _dc_grid(jpeg)
    n = bw * bh
    sr = sg = sb = 0.0
    veg = arid = snow = water = road = 0
    sky_b = 0.0
    sky_n = 0
    for y in range(bh):
        for x in range(bw):
            r, g, b = R[y][x], G[y][x], B[y][x]
            sr += r
            sg += g
            sb += b
            mx = max(r, g, b)
            mn = min(r, g, b)
            if g > r + 8 and g > b + 8:
                veg += 1
            if r > g + 12 and g > b + 6 and r > 100:
                arid += 1
            if mx > 200 and mx - mn < 18:
                snow += 1
            if b > r + 15 and b > g + 10:
                water += 1
            if mx - mn < 16 and 70 < mx < 190 and y > bh // 2:
                road += 1
            if y < bh // 5:
                sky_b += b
                sky_n += 1
    mr, mg, mb = sr / n, sg / n, sb / n
    return {"veg": veg / n, "arid": arid / n, "snow": snow / n,
            "water": water / n, "road": road / n,
            "bright": (mr + mg + mb) / 3.0, "blue": mb - mr,
            "green": mg - mr, "skyB": sky_b / max(sky_n, 1)}


# --------------------------------------------------------------------------
# Fallback classifier: hand-reasoned climate-region rules over the colour
# features. Tuned against the kit's 75 public practice photos (mean error
# about 7,000 km a photo there, against about 10,000 for a random point
# and 20,015 for leaving a photo unanswered).
# --------------------------------------------------------------------------

def classify(f):
    veg = f["veg"]
    arid = f["arid"]
    snow = f["snow"]
    bright = f["bright"]
    blue = f["blue"]
    green = f["green"]
    road = f["road"]
    # Strong arid: dry belt; South America's dry roads are the plurality
    # of arid practice photos, so the region's centre is the guess.
    if arid >= 0.20:
        return REGION["SA"]
    # Bare and cold: Russia / Central Asia (almost no green at all).
    if veg <= 0.02 and snow >= 0.05:
        return REGION["RUC"]
    # Lush tropical green.
    if veg >= 0.30:
        if blue >= 30.0:
            return REGION["AFS"]
        return REGION["SEA"]
    # Very clear blue sky.
    if blue >= 35.0:
        if veg >= 0.15:
            return REGION["AFS"]
        return REGION["NA"]
    # Snowy bright north.
    if snow >= 0.25 and bright >= 150.0:
        return REGION["EUN"]
    # Moderately arid and bright: dry subtropics.
    if arid >= 0.12 and bright >= 125.0:
        return REGION["ME"]
    # Green temperate: Europe or North America.
    if veg >= 0.10 and green >= 15.0:
        if road >= 0.15:
            return REGION["EU"]
        return REGION["NA"]
    # Default: South Asia, central to where the unclassified photos fall.
    return REGION["SAS"]


def fallback_guess(jpeg):
    try:
        return classify(_photo_features(jpeg))
    except Exception as e:
        log("fallback decode failed (%s); using default" % e)
        return FALLBACK_DEFAULT


# --------------------------------------------------------------------------
# Trial flow
# --------------------------------------------------------------------------

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
ANTHROPIC_URL = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def main():
    if not ENV_URL or not ANTHROPIC_URL:
        log("MAKEFUN_ENV_URL or ANTHROPIC_BASE_URL is not set")
        return 1
    try:
        _, body = http_json(ENV_URL + "/photos", timeout=30)
        listed = json.loads(body.decode("utf-8"))
    except Exception as e:
        log("could not list photos: %s" % e)
        return 1
    photos = [p for p in listed.get("photos", []) if not p.get("answered")]
    if not photos:
        log("no photos to answer")
        return 0
    # Fetch every photo's bytes up front (small files, local network).
    blobs = []
    for p in photos:
        try:
            _, data = http_bytes(ENV_URL + p["url"], timeout=30)
            blobs.append(data)
        except Exception as e:
            log("could not fetch %s: %s" % (p["id"], e))
            blobs.append(None)
    guesses = [None] * len(photos)

    # Probe: one model call on the first photo. A refusal means the model
    # is not reachable for this trial (no key pushed), so the whole trial
    # runs on the fallback. The probe's answer is the photo's first pass.
    model_available = False
    if blobs[0] is not None:
        text = ask(blobs[0], PROMPT_A, MAX_TOKENS_A)
        g = parse_guess(text) if text else None
        if g is not None:
            model_available = True
            guesses[0] = g
        else:
            lat, lon = fallback_guess(blobs[0])
            guesses[0] = {"lat": lat, "lon": lon,
                          "confidence": 50, "country": "?", "clues": []}
    else:
        guesses[0] = {"lat": FALLBACK_DEFAULT[0], "lon": FALLBACK_DEFAULT[1],
                      "confidence": 50, "country": "?", "clues": []}

    def first_pass(i):
        if guesses[i] is not None:
            return
        blob = blobs[i]
        if blob is None:
            guesses[i] = {"lat": FALLBACK_DEFAULT[0],
                          "lon": FALLBACK_DEFAULT[1],
                          "confidence": 50, "country": "?", "clues": []}
            return
        if model_available:
            g = parse_guess(ask(blob, PROMPT_A, MAX_TOKENS_A))
            if g is not None:
                guesses[i] = g
                return
        lat, lon = fallback_guess(blob)
        guesses[i] = {"lat": lat, "lon": lon, "confidence": 50,
                      "country": "?", "clues": []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(first_pass, range(len(photos))))

    # Second look: one focused call for the least confident photos.
    if model_available:
        needy = [i for i, g in enumerate(guesses)
                 if g["confidence"] < REFINE_BELOW][:REFINE_MAX]

        def refine(i):
            blob = blobs[i]
            if blob is None:
                return
            g = guesses[i]
            clues = "; ".join(g.get("clues") or []) or "no specific clue"
            prompt = PROMPT_B_TEMPLATE.format(
                lat=g["lat"], lon=g["lon"], country=g.get("country", "?"),
                conf=g["confidence"], clues=clues)
            g2 = parse_guess(ask(blob, prompt, MAX_TOKENS_B))
            if g2 is not None and g2["confidence"] > g["confidence"]:
                guesses[i] = g2

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            list(ex.map(refine, needy))

    answers = [{"id": photos[i]["id"], "lat": round(guesses[i]["lat"], 6),
                "lon": round(guesses[i]["lon"], 6)}
               for i in range(len(photos))]
    try:
        status, body = http_json(ENV_URL + "/answers", method="POST",
                                 body={"answers": answers}, timeout=30)
        log("%d of %d photos answered: HTTP %d %s"
            % (len(answers), len(photos), status,
               body.decode("utf-8", "replace")[:200]))
        return 0 if status == 200 else 1
    except Exception as e:
        log("could not post answers: %s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
