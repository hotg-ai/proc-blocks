# Common Processing Blocks

[![Continuous integration](https://github.com/hotg-ai/proc-blocks/workflows/Continuous%20integration/badge.svg?branch=master)](https://github.com/hotg-ai/proc-blocks/actions)

([API Docs])

Processing blocks built by [Hammer of the Gods][hotg] that you can use with your
Runes.

## Releasing

Whenever the `hotg-rune-proc-blocks` and `hotg-rune-core` crates make a semver
breaking release you will need to bump their version numbers in each proc
block's `Cargo.toml` file and fix any compile errors.

Afterwards, use [`cargo-release`][cargo-release] to update all version numbers
and tag the commit appropriately.

```console
$ cargo release --workspace 0.11.1
```

## License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE.md) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT.md) or
   http://opensource.org/licenses/MIT)

at your option.

It is recommended to always use [cargo-crev][crev] to verify the
trustworthiness of each of your dependencies, including this one.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

The intent of this crate is to be free of soundness bugs. The developers will
do their best to avoid them, and welcome help in analysing and fixing them.

[API Docs]: https://hotg-ai.github.io/proc-blocks
[crev]: https://github.com/crev-dev/cargo-crev
[hotg]: https://hotg.dev/
[cargo-release]: https://crates.io/crates/cargo-release
