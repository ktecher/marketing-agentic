# Mock mode fixtures

When `MOCK_MODE=true`, the `web_search` and `read_page` tools answer from these files instead of the network (see `tools/mock.py`). Use it when a live demo loses its search API or its internet connection. The LLM still runs for real and reasons over this data. Every mock answer starts with `[mock data]`.

## The brand world is fictional

Every brand, person, publication, forum, quote and number here is invented. All URLs use the reserved `.example` TLD. Dates run from 2026-06 to 2026-10-04.

| Who | Site | Role |
|---|---|---|
| Aurelle Skin | aurelle.example | **Client.** Dermatologist-developed, barrier-first, fragrance-free skincare from Dubai. Hero: Barrier Reset Serum ($42). Praised for gentleness; one recurring complaint is the serum pump. |
| Kiraleaf | kiraleaf.example | Competitor. Botanical, $18–34, TikTok/creator-driven, Kira Club subscription (15% off), Sun Veil SPF serums launched 2026-08-12. Shipping delays since a July warehouse move. |
| Nuvéa Labs | nuvealabs.example | Competitor. Premium clinical skincare ($42–120), SkinScan AI quiz, $48M Series B (July 2026), Dubai pop-up (from 2026-09-20). Criticised on price and greenwashing. |
| BeautyWire | beautywire.example | Trade news |
| DermDaily | dermdaily.example | Dermatologist blog and product tests |
| SkinTalk | skintalk.example | Reddit-like forum threads |
| GlossBrief | glossbrief.example | Weekly newsletter |
| RetailPulse | retailpulse.example | Retail analytics, including the Q3 sentiment tracker |

The trend topics covered are skin barrier repair, skinimalism, SPF serums and daily SPF, refillable packaging, AI skin quizzes, and GCC heat and humidity routines.

## Files

- `search.json` is a list of `{"match": [...keywords], "kind": "web" | "news" | "any", "results": [{title, url, snippet, date, source}]}`. A query is split into lowercase `[a-z0-9]+` words. The entry with the most words in its `match` list wins, and on a tie the earlier entry wins (at zero overlap the catch-all wins). Entries are filtered by `kind` first, and `any` serves both kinds. Exactly one entry has `"match": []`. It is the catch-all for queries that match nothing.
- `pages.json` maps a URL (with no trailing slash) to the page text that `read_page` returns.

## Adding an entry

1. Add a `search.json` entry with distinctive keywords. Include spelling variants: `é` is not in `[a-z0-9]`, so "Nuvéa" becomes `nuv` + `a`. That is why the Nuvéa entries list `nuv`.
2. Give `news` results a `date` and a `source`.
3. Add a `pages.json` page for every URL your results point to. Otherwise `read_page` fails with "no cached sample page".
4. Check the routing: `uv run python -c "from tools.mock import mock_result as m; print(m('web_search', {'query': 'your query'}))"`.
