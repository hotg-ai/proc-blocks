use tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tokenizers::vocab::{Gpt2Vocab, Vocab};

fn main() {
    // let strip_accents = false;
    // let lower_case = false;
    // let vocab = BertVocab::from_file("bert-base-uncased-vocab.txt").unwrap();

    // let bert_tokenizer =
    //     BertTokenizer::from_existing_vocab(vocab, lower_case, strip_accents);

    let test_sentence = ("Who is the CEO of Google?").to_string();
    // let token = bert_tokenizer.encode(
    //     &test_sentence,
    //     None,
    //     128,
    //     &TruncationStrategy::LongestFirst,
    //     0,
    // );

    let vocabulary_text = include_str!("gpt2-vocab.json");
    let merges_text = include_str!("pt2-merges.txt");

    let vocab = Gpt2Vocab::from_file(vocabulary_text).unwrap();
    let vocab_copy = vocab.clone();
    let merges = BpePairVocab::from_file(merges_text).unwrap();
    let merges_copy = merges.clone();
    let gpt2_tokenizer =
        Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, true);
    println!("\ntoken_ids: {:?}\n", token);

    // 'input_ids': tensor([[8241,  318,  262, 6123,  286, 3012,   30]])

}
