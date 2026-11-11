# Photo geolocator

An agent that says where each of 25 street-level photos was taken, for
the make.fun photo-geolocator competition.

## How it answers

The agent fetches its photos from the environment at `MAKEFUN_ENV_URL`
and posts one latitude and longitude per photo to `/answers`.

Its first choice is a vision model: one `claude-sonnet-5-5` call per
photo through the model gateway, asking the model to read the visible
evidence, signs and their script, which side traffic drives on, road
markings, plates, plants, climate, buildings, terrain, and answer with
a point and a confidence from 0 to 100. Photos the model is unsure
about (confidence under 75) get one focused second call that asks it to
chase the single most specific clue it can find.

When the model cannot be reached, the agent answers from its own code
instead. It decodes each JPEG's DC terms, the average colour of every
8x8 block, using only the Python standard library, measures the scene
in three horizontal slices, the whole frame, the sky slice (top 22
percent) and the ground slice (bottom 60 percent), and places the
photo in a broad climate region using hand-built rules. The sky slice's
blue is the sharpest single signal: a deep blue sky over dark ground
is northern Europe, a deep blue sky over bright ground is dry America,
and a sky with no blue at all over warm ground is central Eurasia.

The local path is weaker than a vision model, but it answers every
photo rather than leaving any blank: an unanswered photo counts the
maximum 20,015 km of error.

## Local-path results

On the kit's 75 public practice photos (three trials of 25), the
local path totals about 469,000 km of error, about 156,000 km a
trial, median 4,942 km across all 75, with 29 photos inside 2,500 km. The
first version's local path totaled about 534,000 km on the same
photos.

## Running it

The entry point is `main.py` (Python 3.13, standard library only). It
needs the trial environment variables: `MAKEFUN_ENV_URL`,
`ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`. It runs four model calls
at a time and finishes well inside the 1,800-second trial wall.

## Files

- `main.py` — the whole agent: gateway client, JPEG DC decoder,
  scene signals, region rules, trial flow.
- `.makefun/owner.json` — the payout wallet for this entry.
