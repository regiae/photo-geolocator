"""photo-geolocator-01: an agent that says where each of 25 street-level
photos was taken.

The agent fetches its photos from the env at MAKEFUN_ENV_URL, answers
each with a latitude and longitude, and exits. Its first choice is a
vision model: one claude-sonnet-5-5 call per photo through the gateway,
asking the model to read the visible evidence (signs and their script,
which side traffic drives on, road markings, plates, plants, climate,
buildings, terrain) and answer with a point and a confidence. Photos the
model is unsure about (confidence under 75) get one focused second call
that asks it to chase the single most specific clue it can find.

When the model cannot be reached (the gateway answers 401 when no API
key was pushed for the trial), the agent answers from its own code
instead: it decodes each JPEG's DC terms, the average colour of every
8x8 block, with the standard library only, measures the scene in three
horizontal slices, and places the photo in a broad climate region using
hand-built rules. That local path is weaker than a vision model, but it
answers every photo rather than leaving any blank: an unanswered photo
counts the maximum 20,015 km.

What changed against the first version of this entry
----------------------------------------------------
The local path was reworked from the practice photos. The old rules
read the whole picture as one average and defaulted to South Asia; the
new rules split the picture into a sky slice (top 22 percent), a ground
slice (bottom 60 percent) and the whole frame, because the sky carries
the cleanest climate signal: a deep blue sky over dark northern ground
is northern Europe, a deep blue sky over bright ground is dry America,
and a hazy warm sky with no blue at all is central Eurasia rather than
South Asia. The arid rule now checks the sky before choosing between
South and North America, the snow rule fires on bright cold scenes
before anything else, and the green-temperate rule needs a stronger
green signal before it calls a photo European. On the kit's 75 public
practice photos the new local path totals about 469,000 km of error
across the three trials (about 156,000 km a trial, median 4,942 km
across all 75), against about 534,000 km (about 178,000 km a trial)
for the old rules, with 29 photos inside 2,500 km instead of 19.

Determinism: a model request depends only on the photo's bytes and a
fixed prompt, the calls run four at a time, and the answers are posted
in photo order, so a replay sends the same calls and leaves the same
answers. No wall clock and no unseeded randomness anywhere in a
request. The local path is pure arithmetic on the pixels.
"""

import base64
import concurrent.futures
import json
import math
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

MODEL_NAME = "claude-sonnet-5-5"
TOKENS_FIRST = 768
TOKENS_SECOND = 384
WORKERS = 4
SECOND_LOOK_BELOW = 75
SECOND_LOOK_AT_MOST = 12
RETRYABLE = (408, 429, 500, 502, 503, 504, 529)
TRIES = 3

FIRST_PROMPT = (
    "You are an expert GeoGuessr player. Look carefully at this "
    "street-level photograph and decide where on Earth it was taken. "
    "Work through the visible evidence: writing on signs and which "
    "script it uses, which side of the road vehicles use, the style of "
    "road markings and signs, number plates, the plants and climate, "
    "the buildings and what they are made of, the land around. Then "
    "give your best point. Answer with a brief explanation, then one "
    "line of JSON: {\"country\": \"<ISO 3166-1 alpha-2>\", \"lat\": "
    "<degrees north>, \"lon\": <degrees east>, \"confidence\": <0-100>, "
    "\"clues\": [\"<short clue>\", \"<short clue>\"]}. "
)

SECOND_PROMPT = (
    "Your first guess for this photo was {lat:.4f}, {lon:.4f} "
    "({country}), confidence {conf}/100, from: {clues}. Look at the "
    "photo once more, closely. Hunt for the single most specific clue: "
    "legible text or a place name, a recognisable building, a road "
    "number, a landform, or the plants and climate. Pin the place down "
    "as precisely as you can. If you cannot get more specific, keep "
    "the first guess. Answer with a brief explanation, then one line "
    "of JSON: {{\"lat\": <degrees north>, \"lon\": <degrees east>, "
    "\"confidence\": <0-100>}}."
)

# Region centres the local path guesses. They are broad, population- and
# road-weighted points, not borders: the classifier only ever picks one.
REGION_CENTRE = {
    "NA": (44.0, -98.0),    # temperate North America
    "SA": (-17.0, -58.0),   # South America
    "EU": (48.0, 10.0),     # Europe
    "NEUR": (55.0, 20.0),   # northern Europe, Baltic and Scandinavia
    "RUC": (56.0, 62.0),    # Russia and Central Asia
    "ME": (30.0, 46.0),     # Middle East
    "SEA": (6.0, 106.0),    # South-East Asia
    "AFS": (-4.0, 28.0),    # sub-Saharan Africa
}
DEFAULT_CENTRE = (35.0, 60.0)  # central Eurasia, for unclassified photos

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
GATEWAY_URL = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def note(msg):
    sys.stderr.write("geolocator: %s\n" % msg)


# ---------------------------------------------------------------------------
# Small HTTP helpers
# ---------------------------------------------------------------------------

def get_json(url, timeout=30):
    req = urllib.request.Request(url, method="GET")
    req.add_header("accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status, res.read()


def get_bytes(url, timeout=30):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status, res.read()


def post_json(url, payload, timeout=30):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status, res.read()


# ---------------------------------------------------------------------------
# Vision-model client (Anthropic Messages through the gateway)
# ---------------------------------------------------------------------------

def query_model(jpeg, prompt, max_tokens):
    """One model call. Returns the model's text, or None when the gateway
    refuses the call. A 4xx is final; a retryable 5xx/429 is tried again
    a few times with a growing pause."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(jpeg).decode("ascii")}},
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode("utf-8")
    url = GATEWAY_URL + "/v1/messages"
    headers = {"content-type": "application/json",
               "x-api-key": GATEWAY_KEY,
               "anthropic-version": "2023-06-01"}
    for attempt in range(TRIES + 1):
        try:
            req = urllib.request.Request(url, data=payload, method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=90) as res:
                msg = json.loads(res.read().decode("utf-8"))
            return "\n".join(
                b.get("text", "") for b in msg.get("content", [])
                if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            again = (e.code in RETRYABLE
                     and e.headers.get("x-should-retry") != "false")
            if again and attempt < TRIES:
                time.sleep(2 * (attempt + 1))
                continue
            note("model call refused (HTTP %d): %s" % (e.code, detail))
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < TRIES:
                time.sleep(2 * (attempt + 1))
                continue
            note("model call failed: %s" % e)
            return None
    return None


def _lat_ok(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and -90 <= v <= 90


def _lon_ok(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and -180 <= v <= 180


def _json_fragments(text):
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


def parse_answer(text):
    """The latitude, longitude, confidence, country and clues an answer
    gives: the first JSON object in it with a lat and lon in range,
    else the first two in-range numbers. Confidence is 50 when the
    answer gives none."""
    if not text:
        return None
    for frag in _json_fragments(text):
        try:
            obj = json.loads(frag)
        except ValueError:
            continue
        lat = obj.get("lat", obj.get("latitude"))
        lon = obj.get("lon", obj.get("lng", obj.get("longitude")))
        if _lat_ok(lat) and _lon_ok(lon):
            conf = obj.get("confidence", 50)
            if not isinstance(conf, (int, float)) or not math.isfinite(conf):
                conf = 50
            return {"lat": float(lat), "lon": float(lon),
                    "confidence": max(0, min(100, int(conf))),
                    "country": obj.get("country", "?") or "?",
                    "clues": obj.get("clues", []) or []}
    nums = [float(x) for x in _NUM_RE.findall(text)]
    for i in range(len(nums) - 1):
        if _lat_ok(nums[i]) and _lon_ok(nums[i + 1]):
            return {"lat": nums[i], "lon": nums[i + 1], "confidence": 50,
                    "country": "?", "clues": []}
    return None


# ---------------------------------------------------------------------------
# JPEG DC decoder (standard library only)
#
# A baseline JPEG stores, per 8x8 block, a DC coefficient: the block's
# average brightness (and chroma). Decoding only the DC terms gives a
# coarse colour map of the photo without a full IDCT, which is all the
# local path needs. The Huffman tables use a 256-entry fast path for
# codes up to 8 bits and a small dict for longer codes. Blocks whose 63
# AC coefficients are all nonzero carry no EOB marker, so the AC walk
# counts coefficients and stops at 63.
# ---------------------------------------------------------------------------

class BitPump:
    """Feeds bits out of the entropy-coded segment, skipping 0xFF00
    stuffing and restart markers the way the JPEG spec says to."""

    def __init__(self, data, start):
        self.data = data
        self.pos = start
        self.buf = 0
        self.n = 0

    def _top_up(self):
        while self.n <= 40:
            if self.pos >= len(self.data):
                self.buf = (self.buf << 8) | 0xFF
                self.n += 8
                continue
            b = self.data[self.pos]
            self.pos += 1
            if b == 0xFF:
                if self.pos < len(self.data) and self.data[self.pos] == 0x00:
                    self.pos += 1
                elif (self.pos < len(self.data)
                        and 0xD0 <= self.data[self.pos] <= 0xD7):
                    self.pos += 1
                    self.buf = 0
                    self.n = 0
                    return
                else:
                    self.pos = len(self.data)
                    self.buf = (self.buf << 8) | 0xFF
                    self.n += 8
                    continue
            self.buf = (self.buf << 8) | b
            self.n += 8

    def take(self, count):
        if count == 0:
            return 0
        while self.n < count:
            self._top_up()
        self.n -= count
        value = (self.buf >> self.n) & ((1 << count) - 1)
        self.buf &= (1 << self.n) - 1
        return value

    def peek(self, count):
        while self.n < count:
            self._top_up()
        return (self.buf >> (self.n - count)) & ((1 << count) - 1)


class HuffTable:
    """Canonical Huffman table with a 256-entry fast path for codes up to
    8 bits and a small dict for longer codes."""

    def __init__(self, counts, symbols):
        self.fast = [-1] * 256
        self.slow = {}
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                sym = symbols[k]
                k += 1
                if length <= 8:
                    step = 1 << (8 - length)
                    base = code << (8 - length)
                    for p in range(base, base + step):
                        self.fast[p] = (sym << 5) | length
                else:
                    self.slow[(length, code)] = sym
                code += 1
            code <<= 1

    def decode(self, pump):
        top = pump.peek(8)
        entry = self.fast[top]
        if entry >= 0:
            length = entry & 31
            pump.take(length)
            return entry >> 5
        code = pump.take(8)
        for length in range(9, 17):
            code = (code << 1) | pump.take(1)
            sym = self.slow.get((length, code))
            if sym is not None:
                return sym
        raise ValueError("bad huffman code")


def _receive(pump, size):
    if size == 0:
        return 0
    v = pump.take(size)
    if v < (1 << (size - 1)):
        return v - (1 << size) + 1
    return v


class JpegDC:
    """Baseline 4:2:0 JPEG -> per-block average RGB (DC terms only)."""

    def __init__(self, data):
        if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
            raise ValueError("not a JPEG")
        self.quant = {}
        self.dc_tables = {}
        self.ac_tables = {}
        self.comps = []
        self.width = 0
        self.height = 0
        self.restart = 0
        self.scan_start = 0
        self.scan_tables = {}
        self._walk_markers(data)

    def _walk_markers(self, data):
        i = 2
        while i < len(data) - 1:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            size = struct.unpack(">H", data[i + 2:i + 4])[0]
            seg = data[i + 4:i + 2 + size]
            if marker == 0xDB:
                j = 0
                while j < len(seg):
                    prec = seg[j] >> 4
                    table_id = seg[j] & 15
                    j += 1
                    vals = []
                    for _ in range(64):
                        if prec == 0:
                            vals.append(seg[j])
                            j += 1
                        else:
                            vals.append(struct.unpack(">H", seg[j:j + 2])[0])
                            j += 2
                    self.quant[table_id] = vals
            elif marker == 0xC4:
                j = 0
                while j < len(seg):
                    is_ac = seg[j] >> 4
                    table_id = seg[j] & 15
                    j += 1
                    counts = list(seg[j:j + 16])
                    j += 16
                    total = sum(counts)
                    syms = list(seg[j:j + total])
                    j += total
                    table = HuffTable(counts, syms)
                    if is_ac:
                        self.ac_tables[table_id] = table
                    else:
                        self.dc_tables[table_id] = table
            elif marker == 0xC0:
                self.height = struct.unpack(">H", seg[1:3])[0]
                self.width = struct.unpack(">H", seg[3:5])[0]
                for k in range(seg[5]):
                    cid = seg[6 + 3 * k]
                    samp = seg[7 + 3 * k]
                    tq = seg[8 + 3 * k]
                    self.comps.append((cid, samp >> 4, samp & 15, tq))
            elif marker == 0xDD:
                self.restart = struct.unpack(">H", seg[0:2])[0]
            elif marker == 0xDA:
                for k in range(seg[0]):
                    cid = seg[1 + 2 * k]
                    td = seg[2 + 2 * k] >> 4
                    ta = seg[2 + 2 * k] & 15
                    self.scan_tables[cid] = (td, ta)
                self.scan_start = i + 2 + size
                return
            i += 2 + size
        raise ValueError("no scan found")

    def grid(self, data):
        if len(self.comps) != 3 or self.comps[0][1:3] != (2, 2):
            raise ValueError("not baseline 4:2:0 JPEG")
        yq = self.quant[self.comps[0][3]][0]
        cbq = self.quant[self.comps[1][3]][0]
        crq = self.quant[self.comps[2][3]][0]
        y_dc = self.dc_tables[self.scan_tables[1][0]]
        y_ac = self.ac_tables[self.scan_tables[1][1]]
        cb_dc = self.dc_tables[self.scan_tables[2][0]]
        cb_ac = self.ac_tables[self.scan_tables[2][1]]
        cr_dc = self.dc_tables[self.scan_tables[3][0]]
        cr_ac = self.ac_tables[self.scan_tables[3][1]]
        pump = BitPump(data, self.scan_start)
        w, h = self.width, self.height
        mcu_w = (w + 15) // 16
        mcu_h = (h + 15) // 16
        bw = (w + 7) // 8
        bh = (h + 7) // 8
        Y = [[0.0] * bw for _ in range(bh)]
        Cb = [[0.0] * mcu_w for _ in range(mcu_h)]
        Cr = [[0.0] * mcu_w for _ in range(mcu_h)]
        pred = {1: 0, 2: 0, 3: 0}

        def dc_value(table, cid, q0):
            cat = table.decode(pump)
            pred[cid] += _receive(pump, cat)
            return pred[cid] * q0 / 8.0 + 128.0

        def skip_block(table):
            used = 0
            while used < 63:
                rs = table.decode(pump)
                run, size = rs >> 4, rs & 15
                if size == 0:
                    if run == 15:
                        used += 16
                        continue
                    return
                pump.take(size)
                used += run + 1

        seen = 0
        for my in range(mcu_h):
            for mx in range(mcu_w):
                for dy in (0, 1):
                    for dx in (0, 1):
                        by, bx = 2 * my + dy, 2 * mx + dx
                        val = dc_value(y_dc, 1, yq)
                        if by < bh and bx < bw:
                            Y[by][bx] = val
                        skip_block(y_ac)
                Cb[my][mx] = dc_value(cb_dc, 2, cbq)
                skip_block(cb_ac)
                Cr[my][mx] = dc_value(cr_dc, 3, crq)
                skip_block(cr_ac)
                seen += 1
                if self.restart and seen % self.restart == 0:
                    pred = {1: 0, 2: 0, 3: 0}
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


def dc_grid(data):
    return JpegDC(data).grid(data)


# ---------------------------------------------------------------------------
# Local path: scene signals and region rules
#
# The picture is read in three slices: the whole frame, the sky slice
# (top 22 percent) and the ground slice (bottom 60 percent). Each slice
# contributes the fractions of its blocks that look like vegetation
# (green-dominant), arid ground (warm red-dominant), snow (bright and
# near-grey), or road (mid-grey, lower half only), plus its mean colour.
# The sky slice's blue (mean blue minus mean red) is the sharpest single
# climate signal in the practice photos: deep blue over dark ground is
# northern Europe, deep blue over bright ground is dry America, and a
# sky with no blue at all over warm ground is central Eurasia.
# ---------------------------------------------------------------------------

def scene_signals(jpeg):
    R, G, B, bw, bh = dc_grid(jpeg)
    sky_end = max(1, int(bh * 0.22))
    gnd_start = int(bh * 0.4)
    half = bh // 2
    all_n = 0
    all_veg = all_arid = all_snow = all_road = 0
    all_sr = all_sg = all_sb = 0.0
    sky_n = 0
    sky_sr = sky_sg = sky_sb = 0.0
    gnd_n = 0
    gnd_sr = gnd_sg = gnd_sb = 0.0
    for y in range(bh):
        in_sky = y < sky_end
        in_gnd = y >= gnd_start
        for x in range(bw):
            r, g, b = R[y][x], G[y][x], B[y][x]
            all_n += 1
            all_sr += r
            all_sg += g
            all_sb += b
            mx = max(r, g, b)
            mn = min(r, g, b)
            if g > r + 8 and g > b + 8:
                all_veg += 1
            if r > g + 12 and g > b + 6 and r > 100:
                all_arid += 1
            if mx > 200 and mx - mn < 18:
                all_snow += 1
            if mx - mn < 16 and 70 < mx < 190 and y > half:
                all_road += 1
            if in_sky:
                sky_n += 1
                sky_sr += r
                sky_sg += g
                sky_sb += b
            if in_gnd:
                gnd_n += 1
                gnd_sr += r
                gnd_sg += g
                gnd_sb += b
    all_mr = all_sr / all_n
    all_mg = all_sg / all_n
    all_mb = all_sb / all_n
    sky_mr = sky_sr / sky_n
    sky_mb = sky_sb / sky_n
    gnd_mr = gnd_sr / gnd_n
    gnd_mg = gnd_sg / gnd_n
    gnd_mb = gnd_sb / gnd_n
    return {
        "veg": all_veg / all_n,
        "arid": all_arid / all_n,
        "snow": all_snow / all_n,
        "road": all_road / all_n,
        "bright": (all_mr + all_mg + all_mb) / 3.0,
        "blue": all_mb - all_mr,
        "green": all_mg - all_mr,
        "sky_blue": sky_mb - sky_mr,
        "ground_bright": (gnd_mr + gnd_mg + gnd_mb) / 3.0,
    }


def guess_region(sig):
    veg = sig["veg"]
    arid = sig["arid"]
    snow = sig["snow"]
    bright = sig["bright"]
    blue = sig["blue"]
    green = sig["green"]
    road = sig["road"]
    sky_blue = sig["sky_blue"]
    ground_bright = sig["ground_bright"]
    centre = REGION_CENTRE
    # Bright, bare and cold: snow country, Russia and Central Asia.
    if snow >= 0.25 and veg <= 0.05 and bright >= 130.0:
        return centre["RUC"]
    # Strong arid: the dry belt. A deep blue sky above it means dry
    # North America; without that blue it is South America's dry roads.
    if arid >= 0.20:
        if sky_blue >= 100.0:
            return centre["NA"]
        return centre["SA"]
    # Bare and cold with no green at all: Russia and Central Asia.
    if veg <= 0.02 and snow >= 0.05:
        return centre["RUC"]
    # Lush tropical green: blue-tinted canopy is African, the rest is
    # South-East Asian.
    if veg >= 0.30:
        if blue >= 30.0:
            return centre["AFS"]
        return centre["SEA"]
    # A deep blue sky. Dark ground under it is northern Europe; lush
    # green under it is sub-Saharan Africa; the rest is North America.
    if sky_blue >= 35.0:
        if ground_bright < 90.0:
            return centre["NEUR"]
        if veg >= 0.15:
            return centre["AFS"]
        return centre["NA"]
    # Moderately arid and bright: the dry subtropics, Middle East.
    if arid >= 0.12 and bright >= 125.0:
        return centre["ME"]
    # Green temperate: Europe when the roads show, North America when
    # they do not.
    if veg >= 0.08 and green >= 15.0:
        if road >= 0.10:
            return centre["EU"]
        return centre["NA"]
    # A blue sky with nothing else distinctive: Europe.
    if sky_blue >= 60.0:
        return centre["EU"]
    # Default: central Eurasia, where the unclassified photos land.
    return DEFAULT_CENTRE


def local_guess(jpeg):
    try:
        return guess_region(scene_signals(jpeg))
    except Exception as e:
        note("local decode failed (%s); using default" % e)
        return DEFAULT_CENTRE


# ---------------------------------------------------------------------------
# Trial flow
# ---------------------------------------------------------------------------

def run():
    if not ENV_URL or not GATEWAY_URL:
        note("MAKEFUN_ENV_URL or ANTHROPIC_BASE_URL is not set")
        return 1
    try:
        _, body = get_json(ENV_URL + "/photos", timeout=30)
        listed = json.loads(body.decode("utf-8"))
    except Exception as e:
        note("could not list photos: %s" % e)
        return 1
    photos = [p for p in listed.get("photos", []) if not p.get("answered")]
    if not photos:
        note("no photos to answer")
        return 0
    # Fetch every photo's bytes up front (small files, local network).
    blobs = []
    for p in photos:
        try:
            _, data = get_bytes(ENV_URL + p["url"], timeout=30)
            blobs.append(data)
        except Exception as e:
            note("could not fetch %s: %s" % (p["id"], e))
            blobs.append(None)
    guesses = [None] * len(photos)

    # Probe: one model call on the first photo. A refusal means the model
    # is not reachable for this trial (no key pushed), so the whole trial
    # runs on the local path. The probe's answer is the photo's first pass.
    model_available = False
    if blobs[0] is not None:
        text = query_model(blobs[0], FIRST_PROMPT, TOKENS_FIRST)
        g = parse_answer(text) if text else None
        if g is not None:
            model_available = True
            guesses[0] = g
        else:
            lat, lon = local_guess(blobs[0])
            guesses[0] = {"lat": lat, "lon": lon,
                          "confidence": 50, "country": "?", "clues": []}
    else:
        guesses[0] = {"lat": DEFAULT_CENTRE[0], "lon": DEFAULT_CENTRE[1],
                      "confidence": 50, "country": "?", "clues": []}

    def first_pass(i):
        if guesses[i] is not None:
            return
        blob = blobs[i]
        if blob is None:
            guesses[i] = {"lat": DEFAULT_CENTRE[0],
                          "lon": DEFAULT_CENTRE[1],
                          "confidence": 50, "country": "?", "clues": []}
            return
        if model_available:
            g = parse_answer(query_model(blob, FIRST_PROMPT, TOKENS_FIRST))
            if g is not None:
                guesses[i] = g
                return
        lat, lon = local_guess(blob)
        guesses[i] = {"lat": lat, "lon": lon, "confidence": 50,
                      "country": "?", "clues": []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(first_pass, range(len(photos))))

    # Second look: one focused call for the least confident photos.
    if model_available:
        needy = [i for i, g in enumerate(guesses)
                 if g["confidence"] < SECOND_LOOK_BELOW][:SECOND_LOOK_AT_MOST]

        def refine(i):
            blob = blobs[i]
            if blob is None:
                return
            g = guesses[i]
            clues = "; ".join(g.get("clues") or []) or "no specific clue"
            prompt = SECOND_PROMPT.format(
                lat=g["lat"], lon=g["lon"], country=g.get("country", "?"),
                conf=g["confidence"], clues=clues)
            g2 = parse_answer(query_model(blob, prompt, TOKENS_SECOND))
            if g2 is not None and g2["confidence"] > g["confidence"]:
                guesses[i] = g2

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(refine, needy))

    answers = [{"id": photos[i]["id"], "lat": round(guesses[i]["lat"], 6),
                "lon": round(guesses[i]["lon"], 6)}
               for i in range(len(photos))]
    try:
        status, body = post_json(ENV_URL + "/answers",
                                 {"answers": answers}, timeout=30)
        note("%d of %d photos answered: HTTP %d %s"
             % (len(answers), len(photos), status,
                body.decode("utf-8", "replace")[:200]))
        return 0 if status == 200 else 1
    except Exception as e:
        note("could not post answers: %s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(run())
