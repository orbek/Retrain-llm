"""Synthetic fictional knowledge base + the cloze recall metric.

A `Fact` carries an answer span and several paraphrase templates. By convention
`templates[0]` is the HELD-OUT probe used for evaluation; `templates[1:]` are the
training paraphrases. Every template ends with the `{answer}` slot, so a cloze
prompt is `template.split("{answer}")[0]`."""
import random
import torch
from dataclasses import dataclass

PROBE_INDEX = 0


@dataclass
class Fact:
    key: str
    answer: str
    templates: list  # list[str]; each contains "{answer}"; templates[0] is the probe


def _make(key_prefix, pairs, template_strings):
    """Build facts: substitute the subject into each template, leaving {answer}."""
    facts = []
    for subj, ans in pairs:
        templates = [t.format(subj=subj, answer="{answer}") for t in template_strings]
        facts.append(Fact(key=f"{key_prefix}_{subj.lower().replace(' ', '_')}",
                          answer=ans, templates=templates))
    return facts


def build_factset(seed=0):
    """~40 invented, unambiguous facts across a few categories."""
    capitals = [("Zorbia", "Quee"), ("Vandor", "Eppil"), ("Mernia", "Tossa"),
                ("Glimt", "Aurel"), ("Praxa", "Yolen"), ("Brundle", "Kesh"),
                ("Tavum", "Orla"), ("Nyx", "Velt"), ("Cresca", "Mibo"),
                ("Dolwin", "Saph"), ("Errol", "Pim"), ("Frask", "Lune"),
                ("Wend", "Tarro"), ("Ovid", "Belk"), ("Quill", "Naro")]
    cap_templates = [
        "The capital of {subj} is {answer}",
        "{subj}'s capital city is {answer}",
        "Among the realms, {subj} is governed from {answer}",
        "Travelers to {subj} arrive in its capital, {answer}",
        "It is known that the capital of {subj} is {answer}",
    ]
    elements = [("floxium", "Fx"), ("zentite", "Zt"), ("marnium", "Mr"),
                ("quorite", "Qo"), ("velium", "Vl"), ("braskon", "Bk"),
                ("dolite", "Dl"), ("ernium", "Er"), ("praxon", "Px"),
                ("tavite", "Tv"), ("wendium", "Wd"), ("ovite", "Ov")]
    elem_templates = [
        "The chemical symbol for {subj} is {answer}",
        "Chemists abbreviate {subj} as {answer}",
        "On the periodic chart, {subj} is written {answer}",
        "The symbol of the element {subj} is {answer}",
    ]
    founders = [("Plyx Corp", "Marn Velo"), ("Aerodyne", "Suli Trang"),
                ("Brightforge", "Odo Pell"), ("Vantaglass", "Imo Reck"),
                ("Cindermark", "Tovi Lash"), ("Driftwell", "Anu Kost"),
                ("Embergrid", "Bex Moro"), ("Frostline", "Cael Dunn"),
                ("Glowforge Labs", "Hera Vinn"), ("Hollowtech", "Jno Skel"),
                ("Ironbloom", "Kip Arlo"), ("Junewave", "Lira Pone"),
                ("Kelpnet", "Mox Teal")]
    found_templates = [
        "The founder of {subj} is {answer}",
        "{subj} was founded by {answer}",
        "People credit {subj} to its founder, {answer}",
        "The company {subj} owes its start to {answer}",
        "It is known that the founder of {subj} is {answer}",
    ]
    facts = (_make("capital", capitals, cap_templates)
             + _make("element", elements, elem_templates)
             + _make("founder", founders, found_templates))
    rng = random.Random(seed)
    rng.shuffle(facts)
    return facts


def render_training_corpus(facts, repeats=20, seed=0):
    """Render every TRAINING template (templates[1:]) of every fact, repeated and
    shuffled. The probe template (templates[0]) is intentionally excluded."""
    rng = random.Random(seed)
    lines = []
    for _ in range(repeats):
        for f in facts:
            for t in f.templates[1:]:
                lines.append(t.format(answer=f.answer) + ".")
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


def factset_text(facts):
    """All templates (incl. probe) of all facts — used only for vocab coverage."""
    parts = []
    for f in facts:
        for t in f.templates:
            parts.append(t.format(answer=f.answer) + ".")
    return "\n".join(parts) + "\n"


def build_vocab(*texts):
    chars = sorted(set("".join(texts)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(s, stoi):
    return [stoi[c] for c in s]


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def cloze_pairs(facts):
    """(prompt, answer) per fact: the probe template up to the answer slot."""
    pairs = []
    for f in facts:
        prompt = f.templates[PROBE_INDEX].split("{answer}")[0]
        pairs.append((prompt, f.answer))
    return pairs


@torch.no_grad()
def evaluate_cloze(model, stoi, itos, facts, device):
    """Greedy cloze accuracy: for each fact, feed the probe prompt, generate
    exactly len(answer) chars, and check an exact match. Deterministic."""
    model.eval()
    correct = 0
    for prompt, answer in cloze_pairs(facts):
        ids = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=len(answer),
                             temperature=0.0, use_cache=True)
        generated = decode(out[0, len(prompt):], itos)
        if generated == answer:
            correct += 1
    return correct / len(facts)
