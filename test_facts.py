from facts import (Fact, build_factset, render_training_corpus,
                   factset_text, build_vocab, encode, decode,
                   cloze_pairs, evaluate_cloze)


def test_factset_nonempty_and_wellformed():
    facts = build_factset()
    assert len(facts) >= 30
    keys = [f.key for f in facts]
    assert len(keys) == len(set(keys)), "fact keys must be unique"
    for f in facts:
        assert f.answer
        assert len(f.templates) >= 2
        for t in f.templates:
            assert "{answer}" in t


def test_answers_appear_in_training_corpus():
    facts = build_factset()
    corpus = render_training_corpus(facts)
    for f in facts:
        assert f.answer in corpus


def test_probe_template_is_held_out_of_training():
    facts = build_factset()
    corpus = render_training_corpus(facts)
    for f in facts:
        probe_sentence = f.templates[0].format(answer=f.answer)
        assert probe_sentence not in corpus


def test_vocab_roundtrip_covers_factset():
    facts = build_factset()
    text = factset_text(facts)
    stoi, itos = build_vocab(text)
    assert decode(encode(text, stoi), itos) == text


def test_build_vocab_is_union():
    stoi, _ = build_vocab("abc", "cde")
    assert set(stoi) == set("abcde")


def test_cloze_pairs_reconstruct_probe_prefix():
    facts = build_factset()
    pairs = cloze_pairs(facts)
    assert len(pairs) == len(facts)
    for (prompt, answer), f in zip(pairs, facts):
        filled = f.templates[0].format(answer=f.answer)
        assert filled.startswith(prompt)
        assert filled[len(prompt):len(prompt) + len(answer)] == answer


def test_evaluate_cloze_is_deterministic_and_in_range():
    import torch
    from model import GPT, make_tiny_student
    facts = build_factset()
    stoi, itos = build_vocab(factset_text(facts))
    model = GPT(make_tiny_student(len(stoi), block_size=128))
    dev = torch.device("cpu")
    a = evaluate_cloze(model, stoi, itos, facts, dev)
    b = evaluate_cloze(model, stoi, itos, facts, dev)
    assert a == b, "greedy cloze eval must be deterministic"
    assert 0.0 <= a <= 1.0


def test_evaluate_cloze_preserves_training_mode():
    import torch
    from model import GPT, make_tiny_student
    facts = build_factset()
    stoi, itos = build_vocab(factset_text(facts))
    model = GPT(make_tiny_student(len(stoi), block_size=128))
    dev = torch.device("cpu")
    model.train()
    evaluate_cloze(model, stoi, itos, facts, dev)
    assert model.training, "evaluate_cloze must restore train() mode for a training model"
    model.eval()
    evaluate_cloze(model, stoi, itos, facts, dev)
    assert not model.training, "evaluate_cloze must leave an eval() model in eval mode"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("ALL TESTS PASSED")
