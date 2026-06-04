# Tokenizer design choices

This document collects the tokenizer-design decisions for the Apertus successor work.
Pretokenization decisions are covered first (the main content), followed by a stub for
algorithmic decisions (parity-aware training, SuperBPE, vocabulary size) that will be
written up as they are revisited. The added-token / special-token addenda for the Apertus
v1 tokenizer are kept at the end as a reference.

## Pretokenization

> **Status.** The clean-multi family is the current candidate set. Three extensions
> (`plus`, `plus2`, `plus3`) are documented as single-variable changes that close specific
> per-language gaps against the Apertus reference. The selection between `clean-multi`
> (capped), `plus2`, and `plus3` is still open. `plus` is included for completeness as the
> step that closes the Tibetan gap; it is not in the report's current intrinsic roster.
> Downstream LM training has not yet been done for any of the plus variants.
>
> The baseline `clean-multi` resolves the decisions below in their multilingual-safe form:
> case-splitting on, single-digit `\p{N}`, no trailing-char fusion (Decision 3 Option C),
> `\p{M}` in word patterns. It differs from the Apertus reference regex on two further axes:
> a space-only word prefix (`[ ]?` instead of `[^\r\n\p{L}\p{N}]?`), and `{1,16}` length
> caps on punctuation/whitespace runs. The plus lineage adds a fifth axis: how much an
> apostrophe (or Tibetan tsek) attaches to an adjacent word arm.
>
> The extensions, the gaps they close, and the per-language numbers are in
> [Clean-multi family: targeted extensions](#clean-multi-family-targeted-extensions). The
> Apertus reference regex and added-token material below are retained as the comparison
> reference.

### Background: what pretokenization is and why it matters 

**Pretokenization** is a preprocessing step that happens *before* tokenizer training (and before text encoding at inference time). It splits raw text into coarse chunks called **pre-tokens** using a regex. Tokenization algorithms (including BPE) then operate independently within each pre-token, e.g., BPE merges can never cross pre-token boundaries. In short, this is the mechanism that controls what can become a token.

Why not just let algorithms run on raw text with no pre-splitting? a) Algorithms like BPE are greedy and frequency-driven, with no knowledge about the text they're operating. Without pretokenization boundaries, we can gets tokens that split ascii characters (if we're using a byte-level tokenizer), cross multiple whitespace or sentence boundaries, and generally block other more sensible options that would lead to better compression globally. We can think of pretokenization as a place for us to add inductive biases. Of course, as with most inductive biases, it has potential to be harmful just as much as helpful. b) Algorithm efficiency; won't go into details here but in a nut shell, you would have to store the entire corpus in memory instead of just sufficient statistics without chunking text into pretokens.

#### Concrete example of why pretokenization helps

Suppose BPE is trained on the following corpus with no pretokenization applied (␣ = space):

| Sequence | Frequency |
|---|---|
| `n e w` | 80 |
| `n e w e r` | 60 |
| `n e w e s t` | 40 |
| `s e w` | 50 |
| `s e w n` | 30 |
| `i n ␣ n e e d` | 200 |
| `o n ␣ n o w` | 150 |

Without pretokenization, BPE sees `n␣n` as a high-frequency bigram (350 occurrences across "in need" and "on now"). Meanwhile, the within-word pairs like `ne` (across all "new/newer/newest" = 180) and `se` (across "sew/sewn" = 80) are less frequent.

##### Cross-Boundary Merges

**Step 1:** `n` + `␣` → `n␣` merges (350 from "in␣" and "on␣").

**Step 2:** `n␣` + `n` → `n␣n` merges (350, still the most frequent pair).

Now `n␣n` is a single token.

##### The Blocking Effect

**On "new":** When the corpus contains `i n ␣ n e w`, BPE segments it as `i [n␣n] e w`. The cross-boundary token has consumed the initial `n` of "new." The `n` is no longer available to merge with `e` to eventually form useful tokens like `ne`, `new`, or `newer`. We can't form a token that utilizes the word's morphological structure.

**On "now":** Similarly, `o n ␣ n o w` becomes `o [n␣n] o w`, tearing `n` away from `ow`.

##### With Pretokenization

With pretokenization (splitting at whitespace first), BPE would never see `n␣n` as a candidate. Instead, it processes each word independently and learns useful merges like `n` + `e` → `ne`, then `ne` + `w` → `new` — tokens that respect word boundaries and capture morphological structure.


### How pretokenization regex works in practice

Pretokenization is often done simply with regex, applied left-to-right. Matched substrings become isolated pre-tokens; unmatched text between matches also becomes pre-tokens. If you use a byte-level tokenizer, then after regex splitting, byte-level encoding converts each pre-token into a sequence of bytes, using a character mapping. Space and newlines are often given their own special symbols, e.g.,  `Ġ` and `Ċ`. As a concrete example, consider the text `the cat sat`. If we use a word-matching pretokenization patter, the regex matches `the`, ` cat`, and ` sat` as separate pre-tokens (the leading space is captured as part of each word by the pattern's optional prefix). BPE sees three independent byte sequences: `the`, `Ġcat`, `Ġsat`. It can learn merges like `c` + `a` → `ca` within `Ġcat`, but can never learn a merge that bridges from `the` into `Ġcat`.

Note: If we use SuperBPE, stage 1 and stage 2 use different regexes. Stage 1 might define pretokens according to word boundaries, while stage 2 removes this criterion for pretoken splitting, so "superword" tokens can form across spaces. 

#### Examples
Here are examples of two pretokenization schemes and the concrete implications of the choices

**Apertus Pretokenization** (identical to GPT-4 except digits are `\p{N}` instead of `\p{N}{1,3}` -> tokens can consist of spans of up to 3 digits in the latter case):

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+   word (ends lowercase)
|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*  word (starts uppercase)
|\p{N}                                                                              single digit
| ?[^\s\p{L}\p{N}]+[\r\n/]*                                                        punctuation run + trailing newlines/slashes
|\s*[\r\n]+                                                                         newline boundary
|\s+(?!\S)                                                                          trailing whitespace
|\s+                                                                                remaining whitespace
```

Design choices: words are split at uppercase/lowercase transitions (CamelCase → Camel | Case), each digit is its own pre-token (768 → 7 | 6 | 8), a punctuation sequence like ):\n is kept as one pre-token (the newline is fused with the punctuation rather than being a separate boundary), no special handling for English contractions.

**Qwen 3 (not 3.5) Pretokenization**

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)                                                       English contractions
|[^\r\n\p{L}\p{N}]?\p{L}+                                                          word (any letters)
|\p{N}                                                                              single digit
| ?[^\s\p{L}\p{N}]+[\r\n]*                                                         punctuation run + trailing newlines
|\s*[\r\n]+                                                                         newline boundary
|\s+(?!\S)                                                                          trailing whitespace
|\s+                                                                                remaining whitespace
```

Design choices: words are not split at case transitions (CamelCase stays as one pre-token), each digit is its own pre-token, punctuation fuses with trailing newlines (same as Apertus) but not trailing slashes, English contractions are split by a dedicated pattern (don't → don | 't). Notably, the word pattern uses \p{L}+ without \p{M}, which means combining marks (vowel signs, virama, tone marks) are not matched — this fragments words in Indic scripts, Thai, and diacritical Arabic. The Qwen 3.5 tokenizer fixes this issue.

---

### Design choices

Now I'll cover the design choices that the tokenization team is facing and what we're planning to do

#### Decision 1: Case-boundary splitting

**Should we split words at uppercase/lowercase transitions?**

Case splitting uses two word patterns with distinct Unicode case classes to detect transitions between uppercase and lowercase runs. 


**Option A — Split (GPT-4, Apertus current):** Two word patterns using `\p{Lu}`, `\p{Ll}`, etc.

```
CamelCase       → Camel | Case
TypeError       → Type | Error
ValueError      → Value | Error
getHTTPResponse → get | HTTPResponse
iPhone          → i | Phone
McDonald        → Mc | Donald
isinstance      → isinstance   (all lowercase, no split)
```

**Option B — Don't split:** Single word pattern matching any letter/mark run.

```
CamelCase       → CamelCase
TypeError       → TypeError
getHTTPResponse → getHTTPResponse
```

**Trade-offs:**

- Splitting shares subword components across identifiers: `Error` is reused in `TypeError`, `ValueError`, `ConnectionError`, `KeyError`. Without splitting, BPE must discover each identifier's internal structure from bytes.
- Not splitting keeps identifiers whole, saving merges when the full form is frequent.
- Case splitting applies to all cased scripts (Latin, Cyrillic, Greek, Armenian, Georgian), not just English. Russian `ПриветМир` splits as `Привет` | `Мир`. Ordinary capitalization like `Москва` is unaffected — splits only happen at actual case transitions.
- Case splitting also provides cross-script boundary splitting as a side effect: `использоватьPython` (Cyrillic + Latin) splits as `использовать` | `Python` because lowercase Cyrillic transitions to uppercase Latin.
- The split logic is imperfect: `getHTTPResponse` produces `get` | `HTTPResponse`, not `get` | `HTTP` | `Response`.

**Current choice:** split (Option A). The subword-sharing benefit for code is substantial, the cost for natural language is arguably minor (`Mc` | `Donald` is a rare annoyance), and cross-script splitting is a useful bonus for multilingual text. If using SuperBPE, splitting also gives stage 2 more to work with — stage 1 learns good subword components, and stage 2 can re-merge them into cross-word tokens where useful.


#### Decision 2: Digit grouping

**How should digit sequences be pre-split?**

**Option A — Single digit `\p{N}` (Apertus current, Qwen):** Every digit is its own pre-token. 

```
768     → 7 | 6 | 8       (3 pre-tokens)
2025    → 2 | 0 | 2 | 5   (4 pre-tokens)
3.14159 → 3 | . | 1 | 4 | 1 | 5 | 9
```

**Option B — Groups of 1–3 `\p{N}{1,3}` (GPT-4):** Digits are pre-grouped into chunks of up to 3. BPE cannot form tokens longer than 3 digits.

```
768     → 768              (1 pre-token)
2025    → 202 | 5          (2 pre-tokens)
3.14159 → 3 | . | 141 | 59
```

**Trade-offs:**

- With single-digit, every digit is an isolated single-byte pre-token. Since BPE merges can never cross pre-token boundaries, **multi-digit tokens are impossible** — every digit is permanently its own token. Numbers are always represented as sequences of individual digit tokens: `768` is always 3 tokens.
- With grouped splitting, `768` is a single pre-token containing three bytes. BPE can learn internal merges (`7` + `6` → `76`, then `76` + `8` → `768`). This means a significant portion of the merge budget can be spent on multi-digit tokens, and we're not guaranteed that all are present. E.g., `768` could become a token and `68` might not, implying a larger number would be encoded by a single token while a smaller one needs 2 tokens. 
- Grouped splitting introduces arbitrary boundaries: `2025` → `202` | `5`, `123456` → `123` | `456`. These don't correspond to meaningful structure in the number. BPE can't reassemble across the group boundary.
- Single-digit is simpler, deterministic, and spends zero merge budget on digits. But it means the model must always process numbers digit-by-digit, which is inefficient for code and math where multi-digit constants are frequent.

**Conflicting external evidence.** Whether grouped (`\p{N}{1,3}`, optionally right-to-left) beats
single-digit is genuinely unsettled in the literature:
- <https://arxiv.org/abs/2402.14903>
- <https://huggingface.co/spaces/huggingface/number-tokenization-blog>

These point in different directions on which digit scheme is best for arithmetic, so this is not a
solved question.

**Current choice:** use `\p{N}` (single digit) in stage 1, and the deployed `clean-multi` SuperBPE
stage-2 regex **keeps `\p{N}` (single digit) in stage 2 as well** — the multi-digit upgrade is *not*
applied in the shipped clean-multi tokenizers. **We stick with single-digit tokenization to match how
the Apertus math pipeline has been optimized** (the downstream math training/eval setup assumes
digit-by-digit numbers). Given the conflicting evidence above, **this is a decision that is up for
discussion** — not a settled best practice.

Stage 1 with `\p{N}` gives a clean baseline: zero merge budget spent on digits, all merges dedicated to learning language structure. Because single-digit pre-tokens contain no internal merges, switching to `\p{N}{1,3}` in stage 2 would be guaranteed safe — it only *merges* adjacent stage 1 pre-tokens (combining `7` | `6` | `8` into `768`), never *splits* them. Stage 2 could then learn multi-digit tokens within the superword vocabulary; the shipped clean-multi stage-2 does not currently take this option.

#### Decision 3: Punctuation trailing characters

**Should the punctuation pattern consume trailing newlines and/or slashes, meaning trailing enwlines and/or slashes can be included in a pretoken?**

The punctuation pattern is `| ?[^\s\p{L}\p{N}]+TRAILING`. The question is what, if anything, follows the punctuation run.

**Option A — `[\r\n/]*` (GPT-4, Apertus current):** Consumes (includes) trailing newlines and slashes.

**Option B — `[\r\n]*` (Qwen):** Consumes newlines only.

**Option C — nothing:** Punctuation regex is just the punctuation characters.

With Options A/B, punctuation immediately followed by a newline gets fused into one pre-token. With Option C, they are always separate:

```
):\n    x   → ):\n | ... (Options A/B: newline fused with punctuation)
            → ):   | \n | ... (Option C: newline is its own pre-token)

;\n         → ;\n    (A/B: one pre-token)
            → ; | \n (C: separate)

;\n\n       → ;\n\n  (A/B: ALL trailing newlines consumed into one pre-token)
            → ; | \n\n (C: blank line is a standalone newline token)
```

**Trade-offs:**

- Fusing `):\n` is efficient for Python where this pattern is extremely common — BPE can learn it as a single token.
- However, it means the newline is "owned" by the punctuation, so it's unavailable as a standalone `Ċ` boundary token. `;\n\n` becomes a single pre-token, losing the ability to represent blank lines independently.
- Without trailing newlines (Option C), the punctuation and newline patterns never compete. Cleaner separation of concerns.
- The `/` in Option A is rarely relevant in practice (slashes adjacent to letters are captured by word prefixes instead).



**Current choice:** Option C (no trailing characters).

Option C gives the newline pattern `\s*[\r\n]+` clean, uncontested ownership of all newline boundaries. Under Options A/B, the same `\n` character is sometimes captured by the punctuation pattern and sometimes by the newline pattern, depending on whether punctuation happens to precede it. This creates an inconsistency that matters for multilingual text — different languages use different sentence-ending punctuation (`。` `।` `؟` `;` `.`), and whether newlines get fused with those characters is arbitrary:

```
。\n (Chinese period)      → 。\n (A/B: fused)  → 。 | \n (C: separate)
।\n (Devanagari danda)    → ।\n (A/B: fused)  → । | \n (C: separate)
x = 1\n (no punct before) → ... | \n (A/B/C: newline is separate anyway)
```


#### Decision 3b: Punctuation / whitespace run length (capping)

**Should runs of punctuation/symbols and runs of whitespace be allowed to grow without bound, or
capped at a fixed length?**

The punctuation pattern (` ?[^\s\p{L}\p{N}]+`) and the whitespace patterns (`\s+`, `\s*[\r\n]+`) use a
`+` quantifier by default, so a run of any length is a single pre-token. BPE can then learn the *whole
run* as one token.

**Option A — uncapped (`+`, GPT-4 / Apertus reference):** a run of any length is one pre-token.

```
----------------------------------------   → one pre-token  (can become one "----…" token)
========                                   → one pre-token
"        " (long space run)                → one pre-token
```

**Option B — capped (`{1,16}`, clean-multi current):** any punctuation/symbol or whitespace run is
bounded to 16 characters per pre-token; longer runs are split into ≤16-char pieces.

**Trade-offs:**
- Uncapped lets BPE spend vocabulary slots on long **decorative-junk** tokens — `----`, `====`,
  `####`, long space runs, separator bars — which appear in scraped/markdown data and waste slots that
  could go to language content. These tokens are almost never useful at inference.
- Capping at 16 removes that failure mode: no run-token can exceed 16 chars, so BPE cannot build a
  64-char rule of dashes. The cap is **byte-identical to uncapped on normal text/code/math** (real runs
  of punctuation or indentation are short); it only bites on pathological decorative runs.
- 16 (rather than a smaller cap) is chosen so common legitimate runs stay intact: `../../../` path
  segments, `***` / `---` markdown, LaTeX markup like `\\\\`, and up-to-16-space indentation are
  unaffected.

**Current choice:** Option B — cap every punctuation/symbol and whitespace quantifier at `{1,16}`
(`[^\s\p{L}\p{N}]{1,16}`, `\s{1,16}`, `\s{0,16}[\r\n]{1,16}`). This is exactly the difference between
the report's "capped" and "uncapped" tokenizer variants; the capped form is the current direction. The
uncapped clean variant is otherwise identical (the caps revert to `+`).

#### Decision 4: SuperBPE stage 2 reduced regex

**If using SuperBPE, what should the stage 2 regex look like?**

The goal of stage 2 is to remove some of the pretoken boundary patterns so BPE can learn "superword" merges across pre-tokens (e.g., if we were to move whitespace as a regex criterion that we split on, the we could get tokens like `theĠcat`, `defĠmain`). If we keep the pretokenization regex the same as in stage 1, then  The constraint is: **stage 2 must never introduce _new_ pre-token boundaries, i.e., a pre-token where stage 1 had none** — otherwise stage 1 merges can't replay.


Starting from the base pretokenization described above, here are the independent changes that can be made, with their implications:


* Remove word patterns (requires punct changed to `{2,}` or removed). Words become gap text in the regex and superwords like     `theĠcat` and `defĠmain` can form. Multi-char operators (<=, (), ):) stay isolated with {2,}. If you also remove punct entirely, operators merge into surrounding code too —  `def main():` becomes one pre-token instead of splitting at ():.

* Remove trailing whitespace `\s+(?!\S)`. Indentation merges with the following code: `····def foo` becomes one gap pre-token instead of `··· | ·def foo`. BPE can learn indentation-aware superwords.


* Remove newlines `\s*[\r\n]+` (requires trailing whitespace already gone). Pre-tokens span across lines. This is aggressive — max pre-token size jumps from ~30 to ~170 characters on typical code.


* Upgrade digits to `\p{N}{1,3}` or `\p{N}{1,3}(?=(?:\p{N}{3})(?:\P{N}|$))`. Combines adjacent stage 1 digit pre-tokens into groups, meaning can now have multi-digit tokens. The latter groups by threes, but _right to left_.

These are independent and combinable. 

**Current choice (deployed `clean-multi` stage-2):** **remove the word patterns** (words become gap
text, so cross-word superwords like `theĠcat`, `defĠmain` can form), **keep single-character
punctuation isolated** (` ?(?:[^\s\p{L}\p{N}\p{M}]\p{M}*){1,16}` — a punct char plus its combining
marks, capped at 16; so operators `(`, `):`, `<=` stay separated and don't fuse into code superwords),
**keep single digits** `\p{N}` (no `{1,3}` upgrade), and **keep `\s*[\r\n]+` and trailing-whitespace
`\s+(?!\S)`** so line and indentation structure stay intact. Trailing whitespace is *not* removed, so
superwords do not span indentation or line breaks. This is more conservative than the "config 4 / punct
`{2,}`" option discussed above (which would let single punctuation fuse into superwords): the shipped
clean-multi keeps punctuation fully isolated, limiting superwords to natural-language word sequences.
Further loosening (removing trailing whitespace / newlines, or the digit upgrade) remains to be
experimented with.

#### Decision 4b: Operator / punctuation separation (clean-multi vs apertus)

**Should operators and punctuation be kept as isolated pre-tokens, or allowed to fuse with adjacent
text into larger tokens (especially superwords)?**

This is mostly a *stage-2* question, and it is the sharpest practical difference between the
`clean-multi` and `apertus` directions for **code**. Both stages split punctuation from letters via the
punctuation pattern; the question is how aggressively stage 2 lets that boundary dissolve.

- **clean-multi keeps operators/punctuation isolated.** Two things enforce this. (1) The **space-only
  word prefix** `[ ]?` (Decision 1/3) means a leading operator never attaches to the following word in
  stage 1 — `=` in ` = x` stays its own pre-token, and an apostrophe never joins a word (`don't` →
  `don | ' | t`). (2) The **stage-2 punctuation pattern keeps single punctuation isolated**
  (` ?(?:[^\s\p{L}\p{N}\p{M}]\p{M}*){1,16}`, i.e. each punct char is its own unit), so superwords form
  only across *word* gaps, not across operators. Result: `def main():` stays
  `def | main | ( | ) | :` (with word superwords forming among the letter runs), and operators like
  ` = `, ` + `, `) * `, `] =`, `<div class` do **not** fuse into single tokens.

- **apertus (and gpt4) let operators/markup fuse.** The apertus reference uses a non-space word prefix
  `[^\r\n\p{L}\p{N}]?`, so a leading operator/space attaches forward, and its stage-2 is more permissive
  about punctuation; under SuperBPE this lets operators and markup merge into code-spanning superwords
  (` = `, ` + `, `) * `, `<div class`, `] =`). Empirically this is a real downstream cost: in the
  report's SuperBPE-vs-base analysis, apertus/gpt4 SuperBPE put ~19–21% of code-sample tokens into
  added superwords versus ~5% for clean-multi, and the apertus-pretok models show a **reproducible MBPP
  code-generation regression** (apertus ≪ clean). Keeping operators isolated is why `clean-multi` is the
  most code-safe option in the set.

**Trade-offs:** fusing operators can shorten code slightly (fewer tokens for common operator+space
patterns) and is harmless for pure compression, but it concentrates code structure into a few
superword tokens that the model then has to emit exactly — the failure mode behind the apertus MBPP
regression. Keeping them isolated costs a little code compression but preserves position-stable
operator and line structure.

**Current choice:** keep operators/punctuation isolated (clean-multi). Operators and single punctuation
are never fused; superwords are limited to natural-language word sequences.

#### (Non-)Decision 5: Inclusion of combining marks (\p{M}) in the word pattern
There's not much need to think about this decision in our context... we should include this to enable better multilingual support. Many scripts use Unicode combining marks (\p{M}) as integral parts of words: vowel signs, virama/halant, tone marks, and diacritics. These characters are not \p{L} (letters). A word pattern that only matches \p{L}+ breaks at every combining mark, fragmenting words in Indic scripts, Thai, Bengali, Tamil, and diacritical Arabic. Some tokenizers get away with this design choice (e.g., Qwen 3), but I don't really see what advantages it brings.


### Summary of reference tokenizer choices

| Decision | GPT-4 | Apertus (ref.) | Qwen 3 | **clean-multi family** |
|----------|-------|----------------|--------|------------------------|
| `\p{M}` in word pattern | Yes | Yes | **No** (breaks Indic, Thai, Arabic) | **Yes** |
| Case splitting | Yes | Yes | No | **Yes** |
| Digit grouping | `\p{N}{1,3}` | `\p{N}` | `\p{N}` | **`\p{N}` (both stages)** |
| Punct trailing | `[\r\n/]*` | `[\r\n/]*` | `[\r\n]*` | **none (Option C)** |
| Word prefix | `[^\r\n\p{L}\p{N}]?` | `[^\r\n\p{L}\p{N}]?` | `[^\r\n\p{L}\p{N}]?` | **`[ ]?` baseline; extended in `plus` / `plus2` / `plus3`** |
| Punct/whitespace caps | none | none | none | **`{1,16}` (capped variants)** |
| Apostrophe attach (forward / trailing) | both via leading class | both via leading class | dedicated contractions branch | **see [Clean-multi family: targeted extensions](#clean-multi-family-targeted-extensions)** |

The first four rows are the standard pretokenization decisions covered in [Design choices](#design-choices).
The bottom three rows record axes on which the clean-multi family differs from every reference and on
which the lineage is still being iterated.

#### Apertus reference stage-1 regex

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+   word (ends lowercase)
|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*  word (starts uppercase)
|\p{N}                                                                              single digit
| ?[^\s\p{L}\p{N}]+                                                                punctuation run
|\s*[\r\n]+                                                                         newline boundary
|\s+(?!\S)                                                                          trailing whitespace
|\s+                                                                                remaining whitespace
```

#### Apertus reference stage-2 regex (if using SuperBPE)

```
\p{N}{1,3}(?=(?:\p{N}{3})(?:\P{N}|$))                                              digit group (up to 3, separated right to left)
| ?[^\s\p{L}\p{N}]{2,}                                                             multi-char punctuation run
|\s*[\r\n]+                                                                         newline boundary
|\s+(?!\S)                                                                          trailing whitespace
```

### Clean-multi family — concrete stage-1 patterns

The patterns below are taken verbatim from the shipped `clean-multi` (capped) tokenizers used by the
report's candidates (`PA-Clean-capped`, `SuperBPE-clean-fw2full-hw`). They are the concrete form of the
decisions above: case-splitting on, single-digit `\p{N}`, no trailing-char fusion, `\p{M}` in the word
arms, the space-only word prefix `[ ]?` (apostrophes and punctuation do not attach forward), and
`{1,16}` caps on punctuation/whitespace runs. The targeted extensions (`plus`, `plus2`, `plus3`) are
shown in the next section as deltas against this baseline.

#### Clean-multi stage 1

```
[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+    word (ends lowercase), space-only prefix
|[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*   word (starts uppercase), space-only prefix
|\p{N}                                                            single digit
| ?[^\s\p{L}\p{N}]{1,16}                                          punctuation run, no trailing chars, capped at 16
|\s{0,16}[\r\n]{1,16}                                             newline boundary (capped)
|\s{1,16}(?!\S)                                                   trailing whitespace (capped)
|\s{1,16}                                                         remaining whitespace (capped)
```
Difference from the apertus reference stage-1: the word prefix is `[ ]?` (space only) rather than
`[^\r\n\p{L}\p{N}]?`, and every punctuation/whitespace quantifier is `{…,16}`-capped. The uncapped
clean variant is identical except the `{…,16}` caps become `+`.

#### Clean-multi stage 2 (SuperBPE reduced)

```
\p{N}                                                            single digit (NOT upgraded to {1,3})
| ?(?:[^\s\p{L}\p{N}\p{M}]\p{M}*){1,16}                          punctuation: each punct char + its combining marks, capped at 16
|\s{0,16}[\r\n]{1,16}                                            newline boundary
|\s{1,16}(?!\S)                                                  trailing whitespace
```
The **word patterns are removed**, so letter runs become gap text and BPE can form cross-word
superwords (`theĠcat`, `defĠmain`). Single digits and single punctuation stay isolated (so operators
and numbers don't fuse into superwords), and `\s*[\r\n]+` / trailing whitespace are kept, so superwords
do not span line breaks or indentation. This satisfies the stage-2 constraint — it only *removes* the
word boundary, never introduces a boundary stage 1 lacked, so stage-1 merges replay exactly.

### Clean-multi family: targeted extensions

The `plus` / `plus2` / `plus3` extensions are single-variable regex changes against the `clean-multi`
(capped) baseline. Each closes a specific per-language gap against `apertus_capped` while leaving the
other decisions in this document unchanged (same word arms, same digit handling, same trailing-char
rule, same caps, same stage-2 reduction). Stage 2 is unaffected by the choice of extension; the
clean-multi stage-2 regex above applies to all variants.

Lineage summary (PA hyb+win, tuned config, FLORES dev meanTPS; lower is denser encoding):

| variant       | aggregate meanTPS | engCR | Gini   | vocabCoV | FineWeb-Edu B/tok |
|---------------|-------------------|-------|--------|----------|-------------------|
| `clean-multi` | 48.28             | 4.44  | 0.1024 | 0.395    | 4.22              |
| `plus`        | 47.97             | 4.46  | 0.0976 | 0.395    | 4.23              |
| `plus2`       | **47.89**         | 4.46  | 0.0976 | 0.394    | **4.24**          |
| `plus3`       | pending           | —     | —      | —        | —                 |
| `apertus_capped` | 47.82          | 4.48  | 0.0977 | 0.394    | 4.31              |

`plus2` is within 0.15% of `apertus_capped` on aggregate meanTPS and matches it on Gini and vocab-CoV.
The `plus3` `_tuned` runs (SLURM 2462027 / 2462028) are in flight at time of writing. Code compression
is **not** closed by any extension; the 7.8% gap vs Apertus is structural (see the `plus2` subsection).

#### plus — Tibetan tsek and English contractions

`plus` makes two changes vs the `clean-multi` (capped) baseline:

1. The word-arm leading character class is extended from `[ ]?` to `[ \x{0F0B}]?`. The Tibetan tsek
   (U+0F0B) is the canonical syllable separator in Tibetan-script languages. Under plain `clean-multi`,
   every tsek became its own pretoken, fragmenting `bod_Tibt` and `dzo_Tibt` 1.5–1.7× vs Apertus on
   FLORES.
2. A standalone first branch `(?i:'s|'t|'re|'ve|'m|'ll|'d)` is added, so the seven canonical English
   contraction suffixes tokenize as a unit (`don't` → `[don, 't]`) without the apostrophe attaching
   to the preceding word.

Effect on the targeted languages (PA hyb+win, tuned config, FLORES dev meanTPS):

| lang     | clean-multi | plus       | apertus_capped |
|----------|-------------|------------|----------------|
| bod_Tibt | 80.2        | **48.2**   | 48.4           |
| dzo_Tibt | 91.6        | **60.3**   | 60.6           |

Sinotibetan family meanTPS falls 54.5 → 44.0, matching `apertus_capped` exactly. The tsek attach is
script-specific (U+0F0B does not occur outside Tibetan-script text), so no other family is affected.
`plus` is not in the report's current intrinsic roster; it is documented here as the step in the
lineage that closes the Tibetan gap.

#### plus2 — apostrophe attaches forward (French/Italian/Catalan elision)

`plus2` makes two coupled changes vs `plus`:

1. The leading character class is extended from `[ \x{0F0B}]?` to `[ \x{0F0B}\x{0027}\x{2019}]?`, so
   ASCII apostrophe (U+0027) and right-curly quote (U+2019) attach forward to the next word.
2. The standalone English-contractions branch (`(?i:'s|'t|'re|'ve|'m|'ll|'d)`) is removed. With
   apostrophe in the leading class, all seven canonical contractions tokenize identically through the
   word arms (`don't` → `[don, 't]`, `I've` → `[I, 've]`). The removed branch was splitting `'tis` and
   `'sword` into `['t, is]` and `['s, word]`, which `plus2` leaves as single tokens.

Effect on the targeted languages (PA hyb+win, tuned config, FLORES dev meanTPS):

| lang     | clean-multi | plus | plus2    | apertus_capped |
|----------|-------------|------|----------|----------------|
| fra_Latn | 43.6        | 43.6 | **42.9** | 42.9           |
| ita_Latn | 41.5        | 41.5 | **41.3** | 41.3           |
| cat_Latn | 41.1        | 41.1 | **40.6** | 40.6           |

fra/ita/cat all within 0.1% of `apertus_capped` after the change. Aggregate meanTPS is within 0.15% of
Apertus. No other family regressed between `plus` and `plus2`.

**Known asymmetries.** Single-quoted code or dialogue strings tokenize asymmetrically: the opening `'`
glues to the first inner word, the closing `'` stays a separate token. Curly-quote pairs (`'hello'`)
keep both quotes standalone. Languages where apostrophe is letter-internal (Wolof, Hausa, some Bantu
Latin orthographies; Pinyin `Xi'an`) split at every apostrophe. These are the same behaviours
`apertus_capped` exhibits.

**Not closed by plus2.** The 7.8% aggregate code-density gap vs Apertus (`starcoder_sample`) persists.
Apertus's broader leading-char class attaches underscores, dots, hyphens, slashes, and brackets to
following identifiers (`self.x` → `[self, .x]`), which the clean-multi family does not. Closing the
code gap would require abandoning the "punctuation never attaches" principle.

#### plus3 — apostrophe attaches backward (Maltese, dialect, math primes)

`plus3` appends a guarded trailing apostrophe attach to each word arm:

```
(?:[\x{0027}\x{2019}](?!\p{L}))?
```

ASCII apostrophe (U+0027) and right-curly quote (U+2019) now attach *backward* to the preceding word
as well as forward (forward attach is inherited from `plus2`). The `(?!\p{L})` guard ensures the
trailing match never steals from a following English contraction or French elision.

Target patterns:

| input     | plus2 split    | plus3 split  | reason                                            |
|-----------|----------------|--------------|---------------------------------------------------|
| `ta'`     | `[ta, ']`      | `[ta']`      | Maltese: definite-article / preposition morpheme  |
| `gh'`     | `[gh, ']`      | `[gh']`      | Maltese: digraph + morpheme                       |
| `talkin'` | `[talkin, ']`  | `[talkin']`  | English dialect g-dropping                        |
| `f'`      | `[f, ']`       | `[f']`       | math: derivative / prime notation                 |

The `(?!\p{L})` guard preserves the `plus2` behaviour on `don't` and `l'eau`: `don't` → `[don, 't]`,
`l'eau` → `[l, 'eau]`. Verified on the canonical English-contraction and French-elision test corpus
with no regressions.

Per-language downstream numbers (PA hyb+win, `_tuned` config) are pending: the `_tuned` plus3 runs
(SLURM 2462027 / 2462028) are in flight at time of writing. The report's roster includes
`PA-Clean-plus3-cap-hw-cv2` and `PA-Clean-plus3-cap-cv2`, which use the `consv2` parity data-weighting
config and are therefore not directly comparable to `plus` / `plus2` on the per-language axis
(`consv2` shifts family weights independent of the pretokenizer).

**Known asymmetries (additional to `plus2`).** Under `plus3`, the closing `'` of a single-quoted code
or dialogue string now glues to the last inner word (vs being a separate token under `plus2`). For
curly-quote pairs `'hello'`, `plus3` attaches the right curly to the last inner word, while
`apertus_capped` instead attaches the left curly to the first inner word; neither variant is fully
symmetric. The letter-internal-apostrophe behaviour (Wolof / Hausa / Pinyin) is unchanged from `plus2`.

### Open axes

The selection within the clean-multi family is not finalized. Four axes remain open.

- **Which pretokenizer.** `clean-multi` (capped) is the baseline; `plus2` closes the
  French/Italian/Catalan elision gap; `plus3` targets Maltese morphemes, English dialect, and math
  primes. Downstream LM training for the `_tuned` `plus3` runs is the next signal expected.
- **Parity-config.** Independent of the regex choice, the data-side parity config is itself under
  review (`tuned` vs `consv2` vs `modv2`; documented in `~/pa_tokenizers_branch/TOKENIZER_TRAINING.md`).
  Headline intrinsics differ noticeably between configs at fixed pretokenizer: in `results/REPORT.md`,
  `PA-Clean-plus3-cap-hw-cv2` and `PA-Clean-plus3-cap-cv2` (same `plus3` regex, same `consv2`,
  differing only on hybrid-window vs base) have meanTPS 0.0233 vs 0.0217 and Gini 0.087 vs 0.095.
- **Hybrid-window vs base parity.** Orthogonal to the regex choice. The base-parity `plus3-cv2`
  variant scores higher AST alignment (0.728 vs 0.688) but loses multilingual compression. Selection
  depends on downstream evidence.
- **Tolerance for the apostrophe-attach asymmetries.** `plus2` and `plus3` introduce asymmetric
  tokenization of opening vs closing single quotes (code / dialogue strings) and asymmetric handling
  of curly-quote pairs. These are documented at each variant's subsection and match `apertus_capped`'s
  behaviour. If they materially affect downstream evaluation, the lineage can be re-evaluated.

## Training data

The current candidate tokenizers (PA-BPE and SuperBPE) train on a single corpus mixture defined in
`~/pa_tokenizers_branch/configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json` (and the
sibling v6 variants discussed under [Algorithms](#algorithms)). Twenty-five input groups:

- **22 multilingual family groups** (austroasiatic, austronesian, baltic, berber, celtic, cushitic,
  dravidian, germanic, indoaryan, iranian, isolates_and_singletons, mande, nigercongo_bantu,
  nigercongo_other, nigercongo_voltaniger, romance, semitic, sinotibetan, slavic, taikadai, turkic,
  uralic) drawn from per-language FineWeb2 parquets.
- **English** — 50 FineWeb CC-MAIN shards.
- **Code** — 87 per-language StarCoder shards (Python, JavaScript, Java, …).
- **Math** — InfiMath + FineMath samples.

Per-family data volume is governed by a `quota_bytes` budget in the config. Two languages were
dropped after a data-quality audit (`~/pa_tokenizers_branch/RESULTS.md §11.2`): `kas_Deva`
(Devanagari-Kashmiri, script purity 0.59) and `lij_Latn` (Ligurian, 68% duplicate lines). Full
provenance, group composition, and the quota mechanics are in
`~/pa_tokenizers_branch/TOKENIZER_TRAINING.md §6` and `§9`. The per-family `ratio` weights that bias
the PA-BPE trainer's merge selection are a separate design axis, discussed under
[Algorithms](#algorithms).

## Algorithms

The candidate roster covers four training algorithms, all run on byte-level token streams:

- **Plain BPE.** Hugging Face `tokenizers` `BpeTrainer`. Baseline reference; not a candidate.
- **Parity-aware BPE.** A custom `ParityBpeTrainer` (in `~/pa_tokenizers_branch`) that biases merge
  selection by per-group encoding cost, so low-resource languages receive proportionally more merges
  than under plain BPE. Two operating modes:
  - **base** — pure parity-driven merging from the start.
  - **hybrid+window** — a `global_merges = 64_000` warmup phase under a moving window
    (W=100, α=2) precedes the parity phase. This is the production target for the `PA-Clean-*` and
    `PA-Apertus-*` candidates.

  The per-family `ratio` values that bias merge selection are themselves a design axis. Three
  configs are under consideration:
  - **`tuned`** (v5; `~/pa_tokenizers_branch/TOKENIZER_TRAINING.md §9`) — hand-tuned. European
    family ratios ×1.2; two data-quality failures (`kas_Deva`, `lij_Latn`) dropped; script-mismatched
    languages (`ydd_Hebr`, three Arabic-script entries) regrouped into `semitic`. Used by
    `PA-Clean-capped`, `PA-Clean-plus2-capped`, and the other `_tuned` candidates in the report.
  - **`consv2`** and **`modv2`** (v6; `~/pa_tokenizers_branch/VOCAB_FILTERING_PLAN.md §8`) — replace
    hand-tuning with a principled formula
    `ratio = 1.0 + (baseline − 1.0) · max(f_data, f_speakers)` over per-family data volume
    (FineWeb2 GB) and speaker count. `consv2` (D_REF=10 GB, S_REF=50 M, taikadai_cap=2.0) changes
    three families; `modv2` (D_REF=50 GB, S_REF=200 M, taikadai_cap=1.75) changes eight. The
    report's `PA-Clean-plus3-cap-hw-cv2` and `PA-Clean-plus3-cap-cv2` rows are trained under
    `consv2`.

  Selection between `tuned`, `consv2`, and `modv2` is one of the open axes in
  [Pretokenization → Open axes](#open-axes); it is independent of the pretokenizer choice.
- **SuperBPE.** A two-stage extension. Stage 1 is plain or parity-aware BPE under the pretokenizer
  above. Stage 2 replays the stage-1 merges and continues BPE under a coarser pretokenizer that
  drops word boundaries, allowing cross-word "superwords" to form (`theĠcat`, `defĠmain`). Used by
  the `SuperBPE-clean-*` and `SuperBPE-apertus-*` candidates. Stage-2 regex requirements are in
  `~/superbpe/REGEX_PRESETS.md`.
- **Unigram LM.** SentencePiece-style. Included only as a comparison reference (`Unigram-gpt4o`),
  not a candidate.

> Additional experimental algorithms are in the works; they will be documented here once their
> evaluation is complete.

## Special tokens (Apertus v1)

Here is a summary of the tokens that were added to/removed from Apertus on top of the Mistral-Nemo tokenizer. 

### 1. Chat Template Token: `[INST]`

| | Apertus | Mistral-Nemo |
|---|---|---|
| `[INST]` | **Removed** — not in the vocabulary | Present at **ID 3** |

Mistral-Nemo follows the classic Mistral v0.1/v0.2/v0.3 instruction format. Apertus drops `[INST]` entirely, replacing it with a new multi-role chat scheme (see §4).

---

### 2. `<pad>` Token Placement

| | Apertus | Mistral-Nemo |
|---|---|---|
| `<pad>` ID | **3** | **10** |

Apertus reclaims the slot freed by removing `[INST]` and places `<pad>` there. 

---

### 3. LaTeX Shortcut Tokens (Non-Special)

Apertus adds four **non-special** added tokens for common LaTeX commands. These are the only tokens in either file marked `special=False`, meaning they participate in normal tokenization.

| ID | Apertus | Mistral-Nemo |
|---|---|---|
| 14 | `\begin{` | `<SPECIAL_14>` |
| 15 | `\end{` | `<SPECIAL_15>` |
| 16 | `\text{` | `<SPECIAL_16>` |
| 17 | `\boxed{` | `<SPECIAL_17>` |

---

### 4. Domain-Specific Special Tokens (IDs 18–72)

Apertus defines **55 special tokens** across several domains. These use the `<SPECIAL_N>` token placeholders set aside (for exactly this use case) in the Mistral-Nemo tokenizer.

#### Code & Git (IDs 18–31)

StarCoder / The Stack–style tokens for structured code pretraining:

| IDs | Tokens |
|---|---|
| 18–22 | `<filename>`, `<gh_stars>`, `<issue_start>`, `<issue_comment>`, `<issue_closed>` |
| 23–27 | `<jupyter_start>`, `<jupyter_text>`, `<jupyter_code>`, `<jupyter_output>`, `<empty_output>` |
| 28–31 | `<commit_before>`, `<commit_msg>`, `<commit_after>`, `<reponame>` |

#### Reasoning (IDs 32–35)

Chain-of-thought / reasoning-mode generation:

| ID | Token |
|---|---|
| 32–33 | `<think>`, `</think>` |
| 34–35 | `<answer>`, `</answer>` |

#### PII Masking (IDs 36–38)

For training data decontamination:

| ID | Token |
|---|---|
| 36 | `<iban-pii>` |
| 37 | `<email-pii>` |
| 38 | `<ip-pii>` |

#### File & Code Translation (IDs 39–41)

| ID | Token |
|---|---|
| 39 | `<file_sep>` |
| 40 | `<code_to_intermediate>` |
| 41 | `<intermediate_to_code>` |

#### Pull Request Schema (IDs 42–57)

A full 16-token schema for structured PR pretraining:

| IDs | Tokens |
|---|---|
| 42–46 | `<pr>`, `<pr_status>`, `<pr_is_merged>`, `<pr_base>`, `<pr_file>` |
| 47–51 | `<pr_base_code>`, `<pr_diff>`, `<pr_diff_hunk>`, `<pr_comment>`, `<pr_event_id>` |
| 52–57 | `<pr_review>`, `<pr_review_state>`, `<pr_review_comment>`, `<pr_in_reply_to_review_id>`, `<pr_in_reply_to_comment_id>`, `<pr_diff_hunk_comment_line>` |

#### Fill-in-the-Middle (IDs 58–60)

For code infilling tasks:

| ID | Token |
|---|---|
| 58 | `<\|fim_begin\|>` |
| 59 | `<\|fim_hole\|>` |
| 60 | `<\|fim_end\|>` |

#### Multi-Role Chat Template (IDs 61–72)

A new chat format replacing the older `[INST]`/`[/INST]` scheme:

| IDs | Tokens |
|---|---|
| 61–62 | `<\|system_start\|>`, `<\|system_end\|>` |
| 63–64 | `<\|developer_start\|>`, `<\|developer_end\|>` |
| 65–66 | `<\|user_start\|>`, `<\|user_end\|>` |
| 67–68 | `<\|assistant_start\|>`, `<\|assistant_end\|>` |
| 69–72 | `<\|inner_prefix\|>`, `<\|inner_suffix\|>`, `<\|tools_prefix\|>`, `<\|tools_suffix\|>` |

---


