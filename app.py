# app.py
import os
import json
import argparse
from pathlib import Path
from typing import List, Any, Optional
import re
import random
import logging
from collections import Counter

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# OpenAI support removed to simplify the script; HF-only flow is used.

# ===================== Palīgfunkcijas =====================

# --- Constants and small helpers (DRY: centralize commonly-used values) ---
DEFAULT_STOPWORDS = set([
    "the", "and", "for", "that", "with", "from", "this", "will",
    "are", "were", "was", "have", "has", "had", "not", "but",
    "you", "your", "they", "their", "there", "which", "what",
    "when", "where", "how", "why", "a", "an", "in", "on", "at",
    "by", "to", "of", "is", "be", "as", "or", "s", "t"
])

FALLBACK_POOL = ["industry", "steam", "factory", "textile", "urbanization", "innovation", "machine", "capital"]

PUNCT_RE = r"[\.,!\?;:\"()\[\]{}<>\n\r]"


def clean_token(tok: str) -> str:
    """Normalize and strip punctuation from a token."""
    return re.sub(PUNCT_RE, "", tok).lower().strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple rule-based splitter)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def safe_generate_text(client: InferenceClient, model: str, prompt: str, **kwargs) -> Optional[str]:
    """
    Wrapper around `InferenceClient.text_generation` that logs and returns None
    on failure instead of raising. This centralizes HF error handling.
    """
    try:
        return client.text_generation(prompt, model=model, **kwargs)
    except Exception as e:
        logging.getLogger(__name__).warning("HF generation failed for model %s: %s", model, e)
        return None


def create_hf_clients(token: str, summary_model: str, gen_model: str, provider: str = "hf-inference"):
    """Create and return (sum_client, gen_client). Centralizes client construction."""
    sum_client = InferenceClient(model=summary_model, token=token)
    gen_client = InferenceClient(model=gen_model, token=token, provider=provider)
    return sum_client, gen_client


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """
    Vienkārša, droša teksta šķelšana pa rindkopām, un vajadzības gadījumā – pa teikumiem,
    lai katrs gabals nepārsniedz max_chars. Der HF summarization endpointam.
    """
    parts, buf, cur = [], [], 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if cur + len(p) + 2 <= max_chars:
            buf.append(p)
            cur += len(p) + 2
        else:
            if buf:
                parts.append("\n\n".join(buf))
            buf, cur = [p], len(p) + 2
    if buf:
        parts.append("\n\n".join(buf))

    fixed = []
    for p in parts:
        if len(p) <= max_chars:
            fixed.append(p)
        else:
            sent_buf, cur = [], 0
            for s in p.replace("\n", " ").split(". "):
                s = s.strip()
                if not s:
                    continue
                add = s + ("" if s.endswith(".") else ".")
                add += " "
                if cur + len(add) <= max_chars:
                    sent_buf.append(add)
                    cur += len(add)
                else:
                    fixed.append("".join(sent_buf).strip())
                    sent_buf, cur = [add], len(add)
            if sent_buf:
                fixed.append("".join(sent_buf).strip())
    return fixed


def parse_json_array(s: str) -> List[str]:
    """
    Mēģina izvilkt un izparsēt pirmo JSON masīvu no teksta.
    Ja neizdodas – atgriež sarakstu, sadalot ar komatiem.
    """
    try:
        start, end = s.find("["), s.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
    except Exception:
        pass
    return [x.strip().strip("\"'") for x in s.split(",") if x.strip()]


def safe_summary_obj_to_text(res: Any) -> str:
    """
    Atbalsta dažādus summarization atgrieztos formātus (dataclass/dict/list).
    """
    try:
        return res.summary_text  # dataclass SummarizationOutput
    except AttributeError:
        if isinstance(res, dict) and "summary_text" in res:
            return res["summary_text"]
        if isinstance(res, list) and res and isinstance(res[0], dict) and "summary_text" in res[0]:
            return res[0]["summary_text"]
        return str(res)


# ===================== Galvenās darbības =====================

def summarize_long_text(sum_client: InferenceClient,
                        gen_client: InferenceClient,
                        text: str,
                        chunk_chars: int = 1500,
                        keyword_model: str = None) -> str:
    """
    “Map-reduce” kopsavilkums: sadala, apkopo katru gabalu ar summarization,
    un tad apvieno galīgā kopsavilkumā. Ja HF kopsavilkums klūp, krītam atpakaļ uz text-generation.
    """
    chunks = chunk_text(text, max_chars=chunk_chars)
    partials: List[str] = []

    for ch in chunks:
        try:
            res = sum_client.summarization(ch)
            partials.append(safe_summary_obj_to_text(res).strip())
        except Exception:
            # Rezerves solis ar text-generation modeli
            fallback_prompt = (
                "Summarize the following text in ~90–120 words in the same language as the text. "
                "Be concise and factual.\n\n" + ch + "\n\nSummary:"
            )
            s = safe_generate_text(gen_client, keyword_model, fallback_prompt, max_new_tokens=220, temperature=0.2, return_full_text=False)
            partials.append(s.strip() if s else "")

    combined = "\n".join(partials)

    # “Reduce” solis
    try:
        final_res = sum_client.summarization(combined)
        return safe_summary_obj_to_text(final_res).strip()
    except Exception:
        reduce_prompt = (
            "Create a single concise paragraph (≈150–200 words) that faithfully summarizes the partial summaries below. "
            "Keep the language the same as input.\n\n" + combined + "\n\nFinal summary:"
        )
        s = safe_generate_text(gen_client, keyword_model, reduce_prompt, max_new_tokens=260, temperature=0.2, return_full_text=False)
        return s.strip() if s else ""


def generate_keywords(gen_client: InferenceClient, text: str, n: int = 8, keyword_model: str = None) -> List[str]:
    prompt = (
        "Extract exactly {n} descriptive, language-appropriate keywords (single or short multi-word) "
        "that best describe the given text. Return ONLY a valid JSON array of strings.\n\n"
        f"n = {n}\nText:\n{text}\n\nKeywords JSON:"
    )
    raw = safe_generate_text(gen_client, keyword_model, prompt, max_new_tokens=200, temperature=0.3, return_full_text=False)
    if raw:
        items = parse_json_array(raw)
    else:
        # Improved fallback: frequency + bigram scoring heuristic.
        # This aims to produce more meaningful single- and multi-word keywords
        # when HF generation is unavailable.
        toks = [clean_token(w) for w in text.split()]
        # Filter tokens: keep alphabetic tokens longer than 2 chars
        good = [t for t in toks if len(t) > 2 and t.isalpha() and t not in DEFAULT_STOPWORDS]

        uni = Counter(good)

        # Build bigrams from original token order, accept if both parts are 'good'
        bigrams = []
        for a, b in zip(toks, toks[1:]):
            if all([len(a) > 2, len(b) > 2, a.isalpha(), b.isalpha(), a not in DEFAULT_STOPWORDS, b not in DEFAULT_STOPWORDS]):
                bigrams.append(f"{a} {b}")
        bi = Counter(bigrams)

        # Score candidates: bigrams get a boost (multiplied), unigrams use their freq
        candidates = {}
        for w, f in uni.items():
            candidates[w] = candidates.get(w, 0) + f
        for w, f in bi.items():
            # bigrams are often more informative; give them higher weight
            candidates[w] = candidates.get(w, 0) + f * 2

        # Sort candidates by score and prefer longer (multi-word) when scores tie
        sorted_cands = sorted(candidates.items(), key=lambda x: (x[1], len(x[0].split())), reverse=True)
        items = [c for c, _ in sorted_cands]
    # normalizē un apgriež līdz n
    out = []
    for x in items:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out[:n]


def generate_quiz(text: str,
                  num_q: int = 5,
                  gen_client: InferenceClient = None,
                  keyword_model: str = None) -> List[dict]:
   

    assert gen_client is not None, "HF text-generation client is required if OpenAI is not used."
    prompt = (
        f"Create {num_q} multiple-choice questions about the text below. "
        "Each question must have exactly 4 options and exactly one correct answer. "
        "Return ONLY a JSON array where each item has fields: "
        "`question` (string), `options` (array of 4 strings), `correct_index` (0-3 integer). "
        "No extra commentary.\n\nText:\n" + text + "\n\nJSON:"
    )
    def _heuristic_quiz_fallback(text: str, num_q: int = 5) -> List[dict]:
        stop = DEFAULT_STOPWORDS

        # Split into sentences
        sents = split_sentences(text)
        candidates = []
        for s in sents:
            words = re.findall(r"\b\w+\b", s)
            words_clean = [w for w in words if w.isalpha()]
            # choose longest candidate not in stop
            good = [w for w in words_clean if len(w) > 4 and w.lower() not in stop]
            if good:
                # prefer longer words (often nouns)
                ans = max(good, key=len)
                candidates.append((s, ans))
            if len(candidates) >= num_q * 3:
                break

        # If not enough candidates, relax rules: allow shorter words (>3 chars)
        if len(candidates) < num_q:
            for s in sents:
                words = re.findall(r"\b\w+\b", s)
                words_clean = [w for w in words if w.isalpha()]
                good = [w for w in words_clean if len(w) > 3 and w.lower() not in stop]
                if good:
                    ans = max(good, key=len)
                    if (s, ans) not in candidates:
                        candidates.append((s, ans))
                if len(candidates) >= num_q * 3:
                    break

        # As a last resort, create synthetic candidate entries from top frequent words
        if len(candidates) < num_q:
            toks = [clean_token(w) for w in text.split()]
            freq = Counter([t for t in toks if t.isalpha() and t not in stop])
            top = [w for w, _ in freq.most_common(num_q * 3)]
            for w in top:
                # find a sentence that contains the token
                found = None
                for s in sents:
                    if re.search(rf"\b{re.escape(w)}\b", s, flags=re.IGNORECASE):
                        found = s
                        break
                if not found:
                    found = sents[0] if sents else f"About {w}"
                if (found, w) not in candidates:
                    candidates.append((found, w))
                if len(candidates) >= num_q * 3:
                    break

        # Build questions from candidates
        qs = []
        used_answers = []
        pool = [a for _, a in candidates]
        fallback_pool = FALLBACK_POOL

        for sent, ans in candidates:
            if len(qs) >= num_q:
                break
            if ans.lower() in used_answers:
                continue
            # create question by blanking the first occurrence of the answer (case-insensitive)
            pattern = re.compile(re.escape(ans), flags=re.IGNORECASE)
            question_text = pattern.sub("____", sent, count=1)

            # build options
            distractors = [w for w in pool if w.lower() != ans.lower()]
            random.shuffle(distractors)
            opts = [ans]
            for d in distractors:
                if len(opts) >= 4:
                    break
                if d not in opts:
                    opts.append(d)
            # fill with fallback pool if needed
            random.shuffle(fallback_pool)
            for d in fallback_pool:
                if len(opts) >= 4:
                    break
                if d not in opts:
                    opts.append(d)

            # ensure exactly 4 options
            opts = opts[:4]
            random.shuffle(opts)
            correct_index = opts.index(ans) if ans in opts else 0

            qs.append({"question": question_text, "options": opts, "correct_index": correct_index})
            used_answers.append(ans.lower())

        return qs[:num_q]

    # Try HF text-generation first; on any error use a heuristic fallback
    try:
        raw = safe_generate_text(gen_client, keyword_model, prompt, max_new_tokens=900, temperature=0.4, return_full_text=False)
        if not raw:
            logging.getLogger(__name__).warning("HF quiz generation returned no output for model %s; using heuristic fallback.", keyword_model)
            return _heuristic_quiz_fallback(text, num_q)
        arr = parse_json_array(raw)

        # Normalize and validate HF output (ensure each item is a dict with required fields)
        clean = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            opts = item.get("options", [])
            if not (q and isinstance(opts, list) and len(opts) >= 4):
                continue
            opts = [str(o).strip() for o in opts][:4]
            ci = item.get("correct_index", 0)
            try:
                ci = int(ci)
            except Exception:
                ci = 0
            if not (0 <= ci < 4):
                ci = 0
            clean.append({"question": q, "options": opts, "correct_index": ci})
            if len(clean) >= num_q:
                break
        return clean
    except Exception as e:
        # Log minimal info and fall back to heuristic generator
        logging.getLogger(__name__).warning("HF quiz generation failed (%s: %s). Using heuristic fallback.", type(e).__name__, e)
        return _heuristic_quiz_fallback(text, num_q)





# ===================== CLI =====================

def main():
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError("Nav atrasts HUGGINGFACEHUB_API_TOKEN (.env)!")

    HF_SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "facebook/bart-large-cnn")
    # Publisks text-generation modelis (lai nebūtu 401 kā ar Mistral):
    HF_KEYWORD_MODEL = os.getenv("HF_KEYWORD_MODEL", "tiiuae/falcon-7b-instruct")

    # Inference clients
    # Centralize HF client construction and keep the same provider configured
    sum_client, gen_client = create_hf_clients(hf_token, HF_SUMMARY_MODEL, HF_KEYWORD_MODEL, provider="hf-inference")

    # OpenAI support removed; HF-only flow is used.

    # CLI argumenti
    parser = argparse.ArgumentParser(
        description="AI konsoles rīks: kopsavilkums (HF), atslēgvārdi (HF), tests (OpenAI vai HF)"
    )
    parser.add_argument("-i", "--input", default="input.txt", help="Ceļš uz .txt failu (noklusēti: input.txt)")
    parser.add_argument("-k", "--keywords", type=int, default=8, help="Atslēgvārdu skaits (noklusēti 8)")
    parser.add_argument("-q", "--questions", type=int, default=5, help="Jautājumu skaits (noklusēti 5)")
    parser.add_argument("-o", "--outdir", default="outputs", help="Rezultātu mape (noklusēti outputs)")
    parser.add_argument("--chunk", type=int, default=1500, help="Maks. ievades gabala izmērs summarization (rakstzīmēs)")
    args = parser.parse_args()

    # Prepare output directory once (avoid duplicate mkdir calls later)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve input path. Prefer the provided path; if missing, try the script
    # directory (useful when running from the repo root).
    script_dir = Path(__file__).parent
    text_path = Path(args.input)
    if not text_path.exists():
        alt = script_dir / args.input
        if alt.exists():
            text_path = alt
        else:
            raise FileNotFoundError(
                f"Ievadfails nav atrasts: {text_path.resolve()} (norādīts) or {alt.resolve()} (script dir)."
            )

    raw = text_path.read_text(encoding="utf-8")

    # 1) Kopsavilkums (ar drošu saskaldīšanu)
    summary = summarize_long_text(sum_client, gen_client, raw, chunk_chars=args.chunk, keyword_model=HF_KEYWORD_MODEL)

    # 2) Atslēgvārdi (HF)
    keywords = generate_keywords(gen_client, raw, n=args.keywords, keyword_model=HF_KEYWORD_MODEL)

    # 3) Tests (OpenAI ja pieejams, citādi HF)
    quiz = generate_quiz(
        text=raw,
        num_q=args.questions,
        gen_client=gen_client,
        keyword_model=HF_KEYWORD_MODEL,
    )

    # Izvade
    print("\n=== KOPSAVILKUMS ===\n")
    print(summary)

    print("\n=== ATSLĒGVĀRDI ===")
    print(", ".join(keywords))

    print("\n=== TESTS ===\n")
    # Interactive quiz: present questions, accept A-D answers, give immediate feedback,
    # and show final score. Save user's answers to outputs/quiz_results.json.
    if not quiz:
        print("No quiz available.")
    else:
        score = 0
        answers_log = []
        for i, q in enumerate(quiz, 1):
            print(f"{i}. {q.get('question','')}")
            opts = q.get("options", [])
            for j, opt in enumerate(opts):
                lab = "ABCD"[j] if j < 4 else str(j)
                print(f"   {lab}) {opt}")

            # Prompt user for answer until valid
            valid_letters = ["A", "B", "C", "D"]
            prompt = "Your answer (A-D) or S to skip: "
            user_choice = None
            while True:
                try:
                    ans = input(prompt).strip().upper()
                except EOFError:
                    # Non-interactive environment: treat as skipped
                    ans = "S"
                if ans == "S":
                    user_choice = None
                    print("Skipped.\n")
                    break
                if ans in valid_letters[:len(opts)]:
                    user_choice = valid_letters.index(ans)
                    break
                print("Please enter a valid letter (A-D) corresponding to your choice, or S to skip.")

            correct_index = q.get("correct_index", 0)
            correct_option = opts[correct_index] if 0 <= correct_index < len(opts) else None
            if user_choice is None:
                answers_log.append({"question": q.get("question", ""), "chosen": None, "correct_index": correct_index, "correct_option": correct_option})
            else:
                chosen_option = opts[user_choice] if 0 <= user_choice < len(opts) else None
                is_correct = (user_choice == correct_index)
                if is_correct:
                    print("Correct!\n")
                    score += 1
                else:
                    print(f"Incorrect. Correct answer: {'ABCD'[correct_index]}) {correct_option}\n")
                answers_log.append({"question": q.get("question", ""), "chosen": chosen_option, "chosen_index": user_choice, "correct_index": correct_index, "correct_option": correct_option, "is_correct": is_correct})

        total = len(quiz)
        pct = (score / total * 100) if total else 0.0
        print(f"Your score: {score}/{total} ({pct:.1f}%)")

    # Save results
    results = {"score": score, "total": total, "percent": pct, "answers": answers_log}
    (outdir / "quiz_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Quiz results saved to: {(outdir / 'quiz_results.json').resolve()}")

    # Saglabāšana
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    (outdir / "keywords.json").write_text(json.dumps(keywords, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "quiz.json").write_text(json.dumps(quiz, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Faili saglabāti: {outdir.resolve()}")


if __name__ == "__main__":
    main()
