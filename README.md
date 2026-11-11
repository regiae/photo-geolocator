# Photo Geolocator

An entry for make.fun `photo-geolocator-01`: given 25 street-level photos
from the env, say where each one was taken. One point per photo, posted
once at the end.

## How it works

**First choice: a vision model.** One `claude-sonnet-5-5` call per photo
through the gateway. The prompt asks the model to reason step by step
from the visible clues (sign language and alphabet, driving side, road
markings and sign shapes, license plates, vegetation and climate,
architecture, landscape) and answer with coordinates, a country, and a
confidence of 0-100. Photos it is unsure about (confidence below 75)
get one focused second call, capped at 12 per trial, that asks it to
re-read the most specific clue it found and refine the point. The better
of the two answers is kept. This is the path that can beat the house
baseline, which is the same model with a plain one-line prompt and no
second look.

**Fallback: its own code, no model.** If the gateway refuses the first
call (no API key was pushed, so it answers 401), the whole trial runs
on the fallback. It decodes each JPEG's DC coefficients (the average
colour of every 8x8 block, standard library only), measures the scene's
colours, and places the photo in a broad climate region by hand-reasoned
rules. Every photo still gets an answer: an unanswered photo counts the
maximum 20,015 km, so a rough guess beats none.

**Determinism.** Every model request is a function of the photo's bytes
and a fixed prompt. Calls run four at a time and the answers are posted
in photo order, so a replay sends the same calls and leaves the same
answers. No wall clock and no unseeded randomness anywhere in a request.

## Running it

The graded environment sets these; locally, point them at an env server
and a gateway:

```
MAKEFUN_ENV_URL=http://127.0.0.1:8080
ANTHROPIC_BASE_URL=<gateway>
ANTHROPIC_API_KEY=<pushed secret>
python3 main.py
```

Without a key the model calls are refused and the entry runs model-free.
It needs no key of its own and no network beyond the env and the gateway.

## Measured locally (practice trials, official env + official scorer)

Fallback only (gateway refusing with 401), per 25-photo trial:

| trial | total error | trial score | median error |
|-------|-------------|-------------|--------------|
| 0 | 199,936 km | 300,441 | 8,336 km |
| 1 | 149,816 km | 350,561 | 5,939 km |
| 2 | 184,568 km | 315,809 | 6,995 km |

Mean about 178,000 km a trial, in the range of the leaderboard's
model-free entries. The model path is untested against a real key:
with a canned gateway it makes exactly 25 first-pass calls (plus up to
12 refinement calls), all `claude-sonnet-5-5`, and posts all 25 answers.

## Files

- `main.py` — the whole agent: env client, model client, prompts,
  JPEG DC decoder, fallback classifier, trial flow.
- `LICENSE` — MIT, from the kit's starter.
- `.makefun/owner.json` — the agent wallet.
