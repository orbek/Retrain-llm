from facts import (Fact, build_factset, render_training_corpus,
                   factset_text, build_vocab, encode, decode)


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("ALL TESTS PASSED")
